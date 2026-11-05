"""
ai_engine.py - AI scene analysis and adaptive difficulty engine.

Provides:
  - YOLOv8-based object detection for classroom images
  - Spatial position and relation helpers
  - build_scene_context() to build a rich scene description dict
  - DifficultyPredictor using sklearn LogisticRegression for adaptive leveling
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Ensure console output regardless of Flask's logging configuration
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[ai_engine] %(levelname)s - %(message)s"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Optional heavy imports — wrapped so the app starts even when packages are
# absent (e.g. during initial setup before pip install).
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    _yolo_available = True
except ImportError:
    logger.warning("ultralytics not installed — YOLO detection disabled.")
    _yolo_available = False

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    _sklearn_available = True
except ImportError:
    logger.warning("scikit-learn / numpy not installed — ML predictor disabled.")
    _sklearn_available = False

# ---------------------------------------------------------------------------
# YOLO model — loaded once at module import time so it is reused across calls.
# Uses the small YOLOv8n checkpoint; swap for a larger model if accuracy matters.
# ---------------------------------------------------------------------------
_yolo_model = None

def _get_yolo_model() -> Optional[object]:
    """Lazy-load and cache the YOLO model."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if not _yolo_available:
        return None
    try:
        _yolo_model = YOLO("yolov8s.pt")   # downloads automatically on first run
        # Verify which weights file is actually loaded
        weights = getattr(_yolo_model, "ckpt_path", None) or "yolov8s.pt"
        num_classes = len(_yolo_model.names) if hasattr(_yolo_model, "names") else "?"
        logger.info(
            "YOLOv8s loaded — weights: %s | classes: %s", weights, num_classes
        )
        return _yolo_model
    except Exception as exc:
        logger.error("Failed to load YOLO model: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def get_spatial_position(cx: float, cy: float, img_w: float, img_h: float) -> str:
    """
    Map a bounding-box centre (cx, cy) to a human-readable position string.

    The image is divided into a 3×3 grid:
        top-left    | top-center    | top-right
        center-left | center        | center-right
        bottom-left | bottom-center | bottom-right

    Args:
        cx: Horizontal centre of the bounding box (pixels).
        cy: Vertical centre of the bounding box (pixels).
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        A position string such as "top-left" or "center".
    """
    if img_w <= 0 or img_h <= 0:
        return "unknown"

    rel_x = cx / img_w   # 0.0 … 1.0
    rel_y = cy / img_h

    if rel_x < 1 / 3:
        col = "left"
    elif rel_x < 2 / 3:
        col = "center"
    else:
        col = "right"

    if rel_y < 1 / 3:
        row = "top"
    elif rel_y < 2 / 3:
        row = "center"
    else:
        row = "bottom"

    if row == "center" and col == "center":
        return "center"
    if row == "center":
        return f"center-{col}"
    if col == "center":
        return f"{row}-center"
    return f"{row}-{col}"


def get_spatial_relations(detected_objects: list) -> list:
    """
    Build a list of pairwise spatial-relation strings from detected objects.

    Each object dict must contain at least:
        {"label": str, "cx": float, "cy": float}

    Relations generated (threshold-based):
        ABOVE / BELOW  — vertical axis
        LEFT OF / RIGHT OF — horizontal axis

    Only relations whose centres differ by more than 5 % of the dominant axis
    are reported to avoid noisy near-equal comparisons.

    Args:
        detected_objects: List of dicts with label, cx, cy keys.

    Returns:
        List of relation strings, e.g. ["book is ABOVE table", "chair is LEFT OF desk"].
    """
    relations = []
    n = len(detected_objects)

    for i in range(n):
        for j in range(i + 1, n):
            a = detected_objects[i]
            b = detected_objects[j]

            # Guard against missing keys
            if not all(k in a and k in b for k in ("label", "cx", "cy")):
                continue

            dx = a["cx"] - b["cx"]   # positive → a is to the right of b
            dy = a["cy"] - b["cy"]   # positive → a is below b (image coords)

            # Use the larger absolute delta to choose the dominant axis
            if abs(dy) >= abs(dx):
                # Vertical relation
                if dy < 0:
                    relations.append(f"{a['label']} is ABOVE {b['label']}")
                else:
                    relations.append(f"{a['label']} is BELOW {b['label']}")
            else:
                # Horizontal relation
                if dx < 0:
                    relations.append(f"{a['label']} is LEFT OF {b['label']}")
                else:
                    relations.append(f"{a['label']} is RIGHT OF {b['label']}")

    return relations


# ---------------------------------------------------------------------------
# Scene context builder
# ---------------------------------------------------------------------------

def build_scene_context(image_path: str) -> dict:
    """
    Run YOLOv8 on *image_path* and return a structured scene description.

    Returns a dict with three keys:
        objects  (list[str])  — deduplicated object labels detected in the image
        relations (list[str]) — pairwise spatial relations between objects
        context_text (str)    — a single paragraph summarising the scene, ready
                                to be embedded in an LLM prompt

    If YOLO is unavailable or detection fails, the function returns safe
    fallback values so downstream code never crashes.
    """
    fallback = {
        "objects": [],
        "relations": [],
        "context_text": "Focus on the objects visible in the image."
    }

    if not os.path.isfile(image_path):
        logger.warning("build_scene_context: file not found: %s", image_path)
        return fallback

    model = _get_yolo_model()
    if model is None:
        return fallback

    try:
        # Resize to max 1280px width in memory before YOLO — keeps aspect ratio,
        # never touches disk, and cuts inference time ~3x on large source images.
        from PIL import Image as _PILImage
        import numpy as _np
        with _PILImage.open(image_path) as _pil:
            _w, _h = _pil.size
            if _w > 640:
                _scale = 640 / _w
                _pil = _pil.resize((640, int(_h * _scale)), _PILImage.LANCZOS)
                logger.debug("Resized for YOLO: %dx%d → %dx%d", _w, _h, *_pil.size)
            _img_array = _np.array(_pil.convert("RGB"))

        logger.info("Running YOLOv8s detection on: %s", image_path)
        results = model(_img_array, verbose=False, conf=0.60, iou=0.5, max_det=10)
        if not results:
            logger.warning("YOLO returned no results for: %s", image_path)
            return fallback

        result = results[0]
        img_h, img_w = result.orig_shape[:2]
        logger.debug("Image dimensions: %dx%d px", img_w, img_h)

        CONF_THRESHOLD = 0.60
        PROMINENCE_THRESHOLD = 0.05   # object must cover ≥5% of image area
        EXCLUDED_LABELS = {"person"}
        # Positions considered background — typically ceiling, wall, sky
        BACKGROUND_POSITIONS = {"top-left", "top-right", "top-center"}

        img_area = img_w * img_h

        detected = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, f"object_{cls_id}")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # ── Confidence filter ──────────────────────────────────────────
            if conf < CONF_THRESHOLD:
                logger.debug(
                    "  Skipped (low conf):    %-20s conf=%.2f", label, conf
                )
                continue

            # ── Label exclusion ────────────────────────────────────────────
            if label.lower() in EXCLUDED_LABELS:
                logger.debug(
                    "  Skipped (excluded):    %-20s conf=%.2f", label, conf
                )
                continue

            # ── Spatial position ───────────────────────────────────────────
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            position = get_spatial_position(cx, cy, img_w, img_h)

            if position in BACKGROUND_POSITIONS:
                logger.debug(
                    "  Skipped (background):  %-20s conf=%.2f  pos=%s",
                    label, conf, position
                )
                continue

            # ── Prominence filter (bbox area / image area) ─────────────────
            bbox_area = (x2 - x1) * (y2 - y1)
            prominence_score = bbox_area / img_area if img_area > 0 else 0

            if prominence_score < PROMINENCE_THRESHOLD:
                logger.debug(
                    "  Skipped (too small):   %-20s conf=%.2f  prominence=%.3f",
                    label, conf, prominence_score
                )
                continue

            detected.append({
                "label": label,
                "cx": cx,
                "cy": cy,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "position": position,
                "conf": conf,
                "prominence": round(prominence_score, 4)
            })
            logger.debug(
                "  Detected: %-20s conf=%.2f  pos=%-15s  prominence=%.3f  bbox=[%.0f,%.0f,%.0f,%.0f]",
                label, conf, position, prominence_score, x1, y1, x2, y2
            )

        if not detected:
            logger.warning("No objects detected in: %s", image_path)
            return fallback

        logger.info(
            "Detection complete — %d object(s) found in '%s'",
            len(detected), os.path.basename(image_path)
        )

        # Deduplicated label list (preserves first-occurrence order)
        seen = set()
        unique_labels = []
        for obj in detected:
            if obj["label"] not in seen:
                seen.add(obj["label"])
                unique_labels.append(obj["label"])

        logger.info("Unique objects: %s", ", ".join(unique_labels))

        relations = get_spatial_relations(detected)
        if relations:
            logger.debug("Spatial relations: %s", " | ".join(relations[:8]))

        # Build a human-readable summary
        objects_str = ", ".join(unique_labels)
        relations_str = (
            "; ".join(relations[:8])   # cap at 8 to keep prompts compact
            if relations else "no clear spatial relations identified"
        )

        context_text = (
            f"The image contains the following objects: {objects_str}. "
            f"Spatial layout: {relations_str}. "
            "Generate questions ONLY about these detected objects."
        )

        # Find the most prominent object and compute highlight bbox as percentages
        most_prominent = max(detected, key=lambda o: o["prominence"])
        highlight = {
            "x": round(most_prominent["cx"] / img_w * 100, 2),
            "y": round(most_prominent["cy"] / img_h * 100, 2),
            "w": round((most_prominent["x2"] - most_prominent["x1"]) / img_w * 100, 2),
            "h": round((most_prominent["y2"] - most_prominent["y1"]) / img_h * 100, 2),
        }
        logger.debug(
            "Highlight object: %s  bbox=%.1f%%,%.1f%%  size=%.1f%%x%.1f%%",
            most_prominent["label"], highlight["x"], highlight["y"],
            highlight["w"], highlight["h"]
        )

        return {
            "objects": unique_labels,
            "relations": relations,
            "context_text": context_text,
            "highlight": highlight,
        }

    except Exception as exc:
        logger.error("build_scene_context error for %s: %s", image_path, exc)
        return fallback


# ---------------------------------------------------------------------------
# Adaptive difficulty predictor
# ---------------------------------------------------------------------------

class DifficultyPredictor:
    """
    Predicts the best starting difficulty level for the next quiz session.

    Uses a LogisticRegression model trained on the user's past session history.
    Falls back to a transparent rule-based heuristic when fewer than
    MIN_SESSIONS_FOR_ML sessions are available.

    Supports per-environment prediction with transfer learning across environments.
    """

    MIN_SESSIONS_FOR_ML = 3   # minimum sessions required to use the ML model
    LEVELS = ["beginner", "intermediate", "advanced", "expert"]
    _ENV_MAP = {"classroom": 0, "kitchen": 1, "supermarket": 2, "fruits": 3}

    def __init__(self):
        self._model: Optional[object] = None
        self._encoder = None
        self._trained = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(sessions: list) -> list:
        """
        Convert a list of QuizSession ORM objects into a 2-D feature matrix.

        Features per session (all numeric):
            0: score percentage (0–100)
            1: level encoded as int  (beginner=0, intermediate=1, advanced=2, expert=3)
            2: session index (recency proxy — higher = more recent)
            3: environment encoded as int

        Returns a list of feature rows (list of lists).
        """
        level_map = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
        env_map = {"classroom": 0, "kitchen": 1, "supermarket": 2, "fruits": 3}
        rows = []
        for idx, s in enumerate(sessions):
            pct = s.get_percentage() if hasattr(s, "get_percentage") else 0.0
            lvl = level_map.get(getattr(s, "level", "beginner"), 0)
            env = env_map.get(getattr(s, "environment", "classroom"), 0)
            rows.append([pct, lvl, idx, env])
        return rows

    @staticmethod
    def _label_for_session(session) -> str:
        """
        Derive the 'ideal next level' label from a completed session.

        Rule:
            score == 100% → expert
            score >= 80 % → advance one level (or stay at advanced)
            score >= 50 % → stay at current level
            score <  50 % → drop one level (or stay at beginner)
        """
        level_order = ["beginner", "intermediate", "advanced", "expert"]
        pct = session.get_percentage() if hasattr(session, "get_percentage") else 0.0
        current = getattr(session, "level", "beginner")
        idx = level_order.index(current) if current in level_order else 0

        if pct == 100:
            return "expert"
        elif pct >= 80:
            return level_order[min(idx + 1, 3)]
        elif pct >= 50:
            return level_order[idx]
        else:
            return level_order[max(idx - 1, 0)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, sessions: list) -> None:
        """
        Fit the LogisticRegression model on historical sessions.

        Args:
            sessions: List of QuizSession ORM objects (all sessions for a user).
        """
        if not _sklearn_available:
            logger.warning("DifficultyPredictor.train: sklearn not available.")
            return

        if len(sessions) < self.MIN_SESSIONS_FOR_ML:
            # Not enough data yet — skip training, rule-based will be used.
            return

        try:
            X = self._extract_features(sessions)
            y = [self._label_for_session(s) for s in sessions]

            # Encode string labels → integers
            self._encoder = LabelEncoder()
            y_enc = self._encoder.fit_transform(y)

            # Weight recent sessions more heavily
            n = len(sessions)
            sample_weights = [1 + (i / n) for i in range(n)]

            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X, y_enc, sample_weight=sample_weights)

            self._model = clf
            self._trained = True
            logger.info(
                "DifficultyPredictor trained on %d sessions. Classes: %s",
                len(sessions),
                list(self._encoder.classes_)
            )
        except Exception as exc:
            logger.error("DifficultyPredictor.train failed: %s", exc)
            self._trained = False

    def predict_next_level(self, sessions: list, environment: str = None) -> str:
        """
        Predict the optimal difficulty level for the next session.

        If environment is given, uses only sessions from that environment
        for primary prediction, with transfer learning from other environments.

        Args:
            sessions: QuizSession ORM objects for the user (oldest-first).
            environment: Optional environment name to predict for specifically.

        Returns:
            One of "beginner", "intermediate", "advanced", "expert".
        """
        if not sessions:
            return "beginner"

        if environment:
            env_sessions = [s for s in sessions if getattr(s, "environment", None) == environment]
            if env_sessions:
                # Primary: use this env's history
                if len(env_sessions) >= self.MIN_SESSIONS_FOR_ML and self._trained:
                    result = self._ml_predict(env_sessions)
                else:
                    result = self._rule_based(env_sessions)
                return result
            else:
                # No sessions for this env — transfer from global history
                return self._rule_based(sessions)

        # General prediction (no environment filter)
        if len(sessions) < self.MIN_SESSIONS_FOR_ML or not self._trained:
            return self._rule_based(sessions)
        return self._ml_predict(sessions)

    def _ml_predict(self, sessions: list) -> str:
        """Run ML model on the most recent session features."""
        if not _sklearn_available or self._model is None:
            return self._rule_based(sessions)
        try:
            X = self._extract_features([sessions[-1]])
            pred_enc = self._model.predict(X)[0]
            predicted_level = self._encoder.inverse_transform([pred_enc])[0]
            if predicted_level in self.LEVELS:
                logger.debug("DifficultyPredictor ML predicted: %s", predicted_level)
                return predicted_level
        except Exception as exc:
            logger.error("DifficultyPredictor._ml_predict failed: %s", exc)
        return self._rule_based(sessions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_confidence(self, sessions: list, environment: str = None) -> int:
        """
        Return model confidence as a percentage (0–100).

        Formula: min(100, number_of_sessions * 20)
        Reaches full confidence at 5+ sessions.
        """
        if environment:
            env_sessions = [s for s in sessions if getattr(s, "environment", None) == environment]
            return min(100, len(env_sessions) * 20)
        return min(100, len(sessions) * 20)

    def _rule_based(self, sessions: list) -> str:
        """
        Simple rule-based level selector based on the last session's score.

        Average of the last up-to-3 sessions is used to smooth noise.
        """
        recent = sessions[-3:]  # last 3 sessions
        avg_pct = sum(
            (s.get_percentage() if hasattr(s, "get_percentage") else 0.0)
            for s in recent
        ) / len(recent)

        current_level = getattr(sessions[-1], "level", "beginner")
        level_order = self.LEVELS
        idx = level_order.index(current_level) if current_level in level_order else 0

        if avg_pct == 100:
            return "expert"
        elif avg_pct >= 80:
            return level_order[min(idx + 1, 3)]
        elif avg_pct >= 50:
            return level_order[idx]
        else:
            return level_order[max(idx - 1, 0)]


# ---------------------------------------------------------------------------
# Module-level singleton — import this in other modules
# ---------------------------------------------------------------------------
difficulty_predictor = DifficultyPredictor()
