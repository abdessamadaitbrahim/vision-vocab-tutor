"""
routes/quiz.py - Quiz blueprint.

Integrates:
  - OpenRouter API (free tier) for question generation
  - Groq API (free tier) as primary fallback
  - YOLOv8 scene analysis via ai_engine.build_scene_context
  - Adaptive difficulty via ai_engine.difficulty_predictor
"""

from flask import Blueprint, render_template, request, jsonify, session, flash, redirect, url_for, abort
from flask_login import login_required, current_user
from models.session import QuizSession
from models.vocabulary import VocabularyWord
from extensions import db
from config import Config
from ai_engine import build_scene_context, difficulty_predictor
import PIL.Image
import io
import os
import random
import json
import re
import logging
import time
import base64
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Ensure console output mirrors ai_engine style
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[quiz] %(levelname)s - %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

quiz = Blueprint("quiz", __name__)

# OpenRouter free models (fallback si Groq échoue)
PRIMARY_MODEL    = "google/gemma-3-27b-it:free"
FALLBACK_MODEL_1 = "google/gemma-3-12b-it:free"
FALLBACK_MODEL_2 = "meta-llama/llama-4-scout-17b-16e-instruct:free"

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ✅ Groq — provider principal (quota généreux, très rapide, vision supportée)
_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # vision gratuit Groq


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_QUESTION_TYPES = ["color", "location", "usage", "counting", "naming"]

_QUESTION_TYPE_GUIDANCE = {
    "naming":   "Ask the name of a specific object. Example: 'What is the item on the table?' or 'Which object is on the left?'",
    "color":    "Ask about the color of a visible object. Example: 'What color is the apple?' or 'What color is the chair?'",
    "counting": "Ask how many of something are visible. Example: 'How many books are on the shelf?' or 'How many chairs do you see?'",
    "usage":    "Ask what an object is used for. Example: 'What is the man holding?' or 'What do you use a knife for?'",
    "location": "Ask where an object is located. Example: 'Where is the clock?' or 'Which object is on the top shelf?'",
}


def get_prompt_by_level(level: str, asked_questions: list, scene_context: dict, question_type: str = "naming") -> str:
    no_repeat = ""
    if asked_questions:
        no_repeat = f"IMPORTANT: Do NOT ask any of these questions again: {asked_questions}"

    context_block = scene_context.get("context_text", "Focus on the objects visible in the image.")
    objects_list = scene_context.get("objects", [])
    objects_hint = (
        f"Detected objects in this image: {', '.join(objects_list)}. " if objects_list else ""
    )

    type_guidance = _QUESTION_TYPE_GUIDANCE.get(question_type, _QUESTION_TYPE_GUIDANCE["naming"])
    type_directive = f"Use question type: {question_type.upper()}. {type_guidance}"

    brevity_rules = """QUESTION RULES (follow strictly):
            - IMPORTANT: Use correct English grammar in questions.
              Always use articles (a, an, the) before object names.
              WRONG: "What is dining table used for?" | RIGHT: "What is the dining table used for?"
              WRONG: "Where is laptop located?" | RIGHT: "Where is the laptop located?"
              Always say 'the [object]' not just '[object]'
            - Ask ONE simple, direct question about a clearly visible object
            - Question must be maximum 10 words
            - NO long descriptions, NO complex sentences
            - NEVER use the question: "What is this object called?" — it is banned
            - ALWAYS vary the question format. Use one of these rotating formats:
              * What color is the [object]?
              * Where is the [object] located?
              * What is [object] used for?
              * How many [objects] can you see?
              * Which object is [position] in the image?
            GOOD examples: 'What color is the apple?' | 'How many books are on the shelf?' | 'What is the man holding?' | 'Where is the clock?'
            BAD examples: 'What is this object called?' | 'Can you identify the object that is placed on the left side of the table?'"""

    prompts = {
        "beginner": f"""You are a vocabulary teacher. Look carefully at this image.
            SCENE CONTEXT: {context_block}
            {objects_hint}
            {no_repeat}
            {brevity_rules}
            {type_directive}
            Generate a NEW simple vocabulary question about a SPECIFIC visible object listed above.
            - Use simple common English words
            - Only ask about objects confirmed detected in the scene context
            - Provide a short hint describing WHERE in the image the object is (e.g. 'Look at the front wall')
            - Provide exactly 3 answer choices (A, B, C)
            - Format response as JSON only, no extra text:
            {{
                "question": "your question here",
                "hint": "Look at the [location in image]...",
                "choices": ["A. word1", "B. word2", "C. word3"],
                "correct": "A. word1",
                "explanation": "A [correct answer] is used for..."
            }}""",

        "intermediate": f"""You are a vocabulary teacher. Look carefully at this image.
            SCENE CONTEXT: {context_block}
            {objects_hint}
            {no_repeat}
            {brevity_rules}
            {type_directive}
            Generate a NEW intermediate vocabulary question about a SPECIFIC object from the detected list above.
            - Only ask about objects confirmed detected in the scene context
            - Provide a short hint describing WHERE in the image the object is
            - Provide exactly 3 answer choices (A, B, C)
            - Format response as JSON only, no extra text:
            {{
                "question": "your question here",
                "hint": "Look at the [location in image]...",
                "choices": ["A. answer1", "B. answer2", "C. answer3"],
                "correct": "A. answer1",
                "explanation": "This is correct because..."
            }}""",

        "advanced": f"""You are a vocabulary teacher. Look carefully at this image.
            SCENE CONTEXT: {context_block}
            {objects_hint}
            {no_repeat}
            {brevity_rules}
            {type_directive}
            Generate a NEW advanced vocabulary question about concepts related to the detected objects above.
            - Ask about nuance, precise usage, or vocabulary meaning
            - Only ask about objects or themes confirmed visible in the scene context
            - Provide a short hint describing WHERE in the image to focus
            - Provide exactly 3 answer choices (A, B, C)
            - Format response as JSON only, no extra text:
            {{
                "question": "your question here",
                "hint": "Look at the [location in image]...",
                "choices": ["A. answer1", "B. answer2", "C. answer3"],
                "correct": "A. answer1",
                "explanation": "This word means..."
            }}""",

        "expert": f"""You are an expert vocabulary teacher. Look carefully at this image.
            SCENE CONTEXT: {context_block}
            {objects_hint}
            {no_repeat}
            Generate a CHALLENGING expert-level vocabulary question about objects visible in this image.
            Focus on ONE of these expert question types:
            - SYNONYMS: "Which word means the same as [word related to the object]?"
            - COMPLEX USAGE: "In which sentence is '[word]' used correctly?"
            - WORD ORIGIN: "The word '[object]' originates from which language?"
            - MULTIPLE MEANINGS: "Which definition of '[word]' applies in this context?"
            - CATEGORY: "Which category does [object] belong to?"
            Only ask about objects confirmed visible in the scene context.
            Provide a short hint describing WHERE in the image to focus.
            Provide exactly 3 answer choices (A, B, C).
            Format response as JSON only, no extra text:
            {{
                "question": "your expert question here",
                "hint": "Focus on the [object] in the image...",
                "choices": ["A. answer1", "B. answer2", "C. answer3"],
                "correct": "A. answer1",
                "explanation": "This is correct because..."
            }}"""
    }
    return prompts.get(level, prompts["beginner"])


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

_EMPTY_SCENE_CTX = {
    "objects": [],
    "relations": [],
    "context_text": "Focus on the objects visible in the image."
}


def safe_parse_json(raw_text: str) -> dict:
    """Robustly parse JSON from AI response, fixing common malformed patterns."""
    text = raw_text.strip()
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    text = re.sub(r'(\[[^\]]*?)(\s*,\s*"correct"\s*:)', r'\1]\2', text)
    text = re.sub(r',\s*\]', ']', text)
    text = re.sub(r',\s*\}', '}', text)
    return json.loads(text)


def _parse_and_shuffle(content: str) -> dict:
    question_data = safe_parse_json(content)
    correct_text = question_data["correct"].split(". ", 1)[-1]
    choices_text = [c.split(". ", 1)[-1] for c in question_data["choices"]]
    random.shuffle(choices_text)
    labeled_choices = [f"{chr(65 + i)}. {c}" for i, c in enumerate(choices_text)]
    correct_labeled = next(
        c for c in labeled_choices if c.split(". ", 1)[-1] == correct_text
    )
    question_data["choices"] = labeled_choices
    question_data["correct"] = correct_labeled
    question_data.setdefault("hint", "Look carefully at the image...")
    question_data.setdefault("explanation", f"The correct answer is: {correct_labeled}")
    return question_data


#  NOUVEAU — appel Groq séparé (provider principal, quota généreux)
def _call_groq(prompt: str, img_b64: str) -> str | None:
    """
    Call Groq vision API. Returns raw content string or None on failure.
    Groq supports llama-4-scout with image_url (base64).
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — skipping Groq")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _GROQ_MODEL,
        "messages": [
            {
                
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ],
        "max_tokens": 350,
    }
    try:
        logger.info("Using model: %s (Groq)", _GROQ_MODEL)
        resp = requests.post(_GROQ_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429:
            logger.warning("Groq rate limited (429) — will try OpenRouter fallback")
            return None
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        logger.debug("Groq responded (%d chars)", len(content))
        return content
    except requests.exceptions.Timeout:
        logger.warning("Groq timeout — will try OpenRouter fallback")
        return None
    except Exception as exc:
        logger.error("Groq call failed: %s", exc)
        return None


def generate_question(image_path: str, level: str, asked_questions: list, question_type: str = "naming", scene_ctx: dict = None) -> dict | None:
    """
    Generate a quiz question using Groq (primary) then OpenRouter (fallback),
    guided by YOLO scene context.
    """
    logger.debug("generate_question called — image=%s  level=%s  type=%s", image_path, level, question_type)

    # Stage 1: scene analysis
    if scene_ctx is None:
        try:
            scene_ctx = build_scene_context(image_path)
            logger.debug("Scene context ready: %d object(s)", len(scene_ctx.get("objects", [])))
        except Exception as exc:
            logger.warning("build_scene_context raised unexpectedly: %s — using empty context", exc)
            scene_ctx = _EMPTY_SCENE_CTX

    # Stage 2: image preparation
    try:
        img = PIL.Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        highlight = scene_ctx.get("highlight")
        if highlight:
            orig_w, orig_h = img.size
            cx = highlight["x"] / 100 * orig_w
            cy = highlight["y"] / 100 * orig_h
            bw = highlight["w"] / 100 * orig_w
            bh = highlight["h"] / 100 * orig_h
            x1 = max(0, int(cx - bw / 2) - 20)
            y1 = max(0, int(cy - bh / 2) - 20)
            x2 = min(orig_w, int(cx + bw / 2) + 20)
            y2 = min(orig_h, int(cy + bh / 2) + 20)
            img = img.crop((x1, y1, x2, y2))
            logger.debug("Cropped to object: [%d,%d,%d,%d] (orig %dx%d)", x1, y1, x2, y2, orig_w, orig_h)
        img.thumbnail((400, 400))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        img_bytes = buf.getvalue()
        logger.debug("Image ready: %dx%d  (%d bytes)", *img.size, len(img_bytes))
    except Exception as exc:
        logger.error("Image preparation failed for %s: %s", image_path, exc)
        return None

    # Stage 3: prompt construction
    # you must as a next developper to add what is the prompt for each level and each question type
    try:
        prompt = get_prompt_by_level(level, asked_questions, scene_ctx, question_type)
    except Exception as exc:
        logger.error("get_prompt_by_level failed: %s", exc)
        return None

    time.sleep(1)
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Stage 4a:  Groq en premier (quota généreux, rapide)
    raw_content = _call_groq(prompt, img_b64)

    # Stage 4b: OpenRouter en fallback si Groq échoue
    if not raw_content:
        logger.info("Groq failed — trying OpenRouter fallback models")
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ]
        for model in (PRIMARY_MODEL, FALLBACK_MODEL_1, FALLBACK_MODEL_2):
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 350,
            }
            try:
                logger.info("Using model: %s (OpenRouter)", model)
                resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=40)
                if resp.status_code == 429:
                    logger.warning("OpenRouter rate limited (429) on %s, trying next...", model)
                    time.sleep(3)
                    continue
                resp.raise_for_status()
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"].strip()
                logger.debug("OpenRouter responded (%d chars) via %s", len(raw_content), model)
                break
            except requests.exceptions.Timeout:
                logger.warning("Timeout on %s, trying next model...", model)
            except Exception as exc:
                logger.error("OpenRouter call failed on %s: %s", model, exc)

    if not raw_content:
        logger.error("All providers failed — returning None")
        return None

    # Stage 5: JSON parsing
    try:
        question_data = _parse_and_shuffle(raw_content)
        logger.debug("Question parsed OK: %s", question_data.get("question", "")[:60])
        return question_data
    except Exception as exc:
        logger.error("Failed to parse AI response: %s | raw: %.200s", exc, raw_content)
        return None


# ---------------------------------------------------------------------------
# Level progression helpers
# ---------------------------------------------------------------------------

_LEVEL_PROGRESSION = ["beginner", "intermediate", "advanced", "expert"]


def _next_level_after_session(current_level: str, score_pct: float):
    """
    Determine the next level and whether the continuous session should continue.

    Returns (next_level: str, should_continue: bool).

    End conditions:
      - User is at 'advanced' and score < 80%
      - User is at 'expert' (after expert session, session truly ends)
    All other cases: continue at the appropriate next level.
    """
    # True end conditions
    if current_level == "advanced" and score_pct < 80:
        return current_level, False
    if current_level == "expert":
        return current_level, False

    # Determine next level
    if score_pct == 100:
        next_level = "expert"
    elif score_pct >= 50:
        idx = _LEVEL_PROGRESSION.index(current_level) if current_level in _LEVEL_PROGRESSION else 0
        next_level = _LEVEL_PROGRESSION[min(idx + 1, len(_LEVEL_PROGRESSION) - 1)]
    else:
        # score < 50%: stay at same level but keep going (only end at advanced+<80%)
        next_level = current_level

    return next_level, True


def update_user_level(user, score_percentage: float) -> None:
    if score_percentage == 100:
        user.level = "expert"
    elif score_percentage >= Config.SCORE_ADVANCED:
        user.level = Config.LEVEL_ADVANCED
    elif score_percentage >= Config.SCORE_INTERMEDIATE:
        user.level = Config.LEVEL_INTERMEDIATE
    else:
        user.level = Config.LEVEL_BEGINNER
    db.session.commit()


def _update_env_score(user, env: str, points: int, score_pct: float) -> None:
    """Update the per-environment points and level on the user model."""
    points_attr = f"{env}_points"
    level_attr = f"{env}_level"
    if hasattr(user, points_attr):
        setattr(user, points_attr, getattr(user, points_attr, 0) + points)
    if hasattr(user, level_attr):
        if score_pct == 100:
            env_level = "expert"
        elif score_pct >= Config.SCORE_ADVANCED:
            env_level = Config.LEVEL_ADVANCED
        elif score_pct >= Config.SCORE_INTERMEDIATE:
            env_level = Config.LEVEL_INTERMEDIATE
        else:
            env_level = Config.LEVEL_BEGINNER
        # Only upgrade env level, never downgrade
        current = getattr(user, level_attr, "beginner") or "beginner"
        if _LEVEL_PROGRESSION.index(env_level) > _LEVEL_PROGRESSION.index(current):
            setattr(user, level_attr, env_level)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_ALLOWED_ENVIRONMENTS = {"classroom", "supermarket", "kitchen", "fruits"}


@quiz.route("/quiz")
@login_required
def quiz_page():
    env = request.args.get("env")
    if not env or env.strip().lower() not in _ALLOWED_ENVIRONMENTS:
        flash("Please choose an environment first!", "warning")
        return redirect(url_for("main.home"))
    env = env.strip().lower()

    images_folder = os.path.join("static", "images", env)

    if not os.path.isdir(images_folder):
        logger.error("Images folder not found: %s", images_folder)
        return render_template(
            "quiz.html",
            error=f"No image folder found for environment '{env}'. "
                  f"Please create static/images/{env}/."
        )

    _ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    images = [
        os.path.basename(f) for f in os.listdir(images_folder)
        if os.path.splitext(f.lower())[1] in _ALLOWED_EXTENSIONS
        and ".." not in f
    ]

    if not images:
        return render_template(
            "quiz.html",
            error=f"No images found for environment: {env}"
        )

    logger.info("Loading images from: static/images/%s/ — %d images found", env, len(images))

    try:
        recent_sessions = (
            QuizSession.query
            .filter_by(user_id=current_user.id)
            .order_by(QuizSession.created_at.asc())
            .all()
        )
        predicted_level = difficulty_predictor.predict_next_level(recent_sessions, environment=env)
    except Exception as exc:
        logger.error("predict_next_level failed: %s", exc)
        # Fall back to per-env level stored on user
        predicted_level = getattr(current_user, f"{env}_level", "beginner") or "beginner"

    logger.info(
        "Starting quiz — user=%s  env=%s  level=%s  images=%d",
        current_user.username, env, predicted_level, len(images)
    )

    random.shuffle(images)
    session["images_list"] = images
    session["image_index"] = 0
    session["score"] = 0
    session["question_count"] = 0
    session["level"] = predicted_level
    session["asked_questions"] = []
    session["question_types"] = []
    session["environment"] = env
    session["environment_level"] = {env: predicted_level}
    session["vocab_answers"] = []

    first_image = images[0]
    image_url = f"/static/images/{env}/{first_image}"

    return render_template(
        "quiz.html",
        image_url=image_url,
        level=predicted_level,
        environment=env,
        username=current_user.username
    )


@quiz.route("/get_question", methods=["GET"])
@login_required
def get_question():
    level = session.get("level", "beginner")
    question_count = session.get("question_count", 0)
    asked_questions = session.get("asked_questions", [])
    images_list = session.get("images_list", [])
    image_index = session.get("image_index", 0)

    if question_count >= Config.MAX_QUESTIONS_PER_SESSION:
        return jsonify({"session_complete": True})

    env = session.get("environment", "classroom")
    image_folder = os.path.join("static", "images", env)

    if not os.path.isdir(image_folder) or not any(
        f.lower().endswith((".jpg", ".jpeg", ".png"))
        for f in (os.listdir(image_folder) if os.path.isdir(image_folder) else [])
    ):
        logger.error("get_question: folder missing or empty — %s", image_folder)
        return jsonify({"question_error": True, "error": f"No images found for environment: {env}"}), 200

    environment = env
    logger.info("Loading images from: static/images/%s/ — %d images found", env, len(images_list))

    if not images_list:
        logger.error("get_question: images_list is empty — session not initialised?")
        return jsonify({"question_error": True, "error": "No images available"}), 200

    current_image_name = os.path.basename(images_list[image_index % len(images_list)])
    image_path = os.path.join("static", "images", environment, current_image_name)
    image_url = f"/static/images/{environment}/{current_image_name}"
    session["image_index"] = image_index + 1

    question_types = session.get("question_types", [])
    next_type = _QUESTION_TYPES[question_count % 5]
    logger.info("Question type: %s (question %d)", next_type, question_count + 1)

    try:
        scene_ctx = build_scene_context(image_path)
    except Exception as exc:
        logger.warning("build_scene_context failed in get_question: %s", exc)
        scene_ctx = _EMPTY_SCENE_CTX
    highlight = scene_ctx.get("highlight", None)

    logger.info("Attempt 1 — generating question for %s", current_image_name)
    question_data = generate_question(image_path, level, asked_questions, next_type, scene_ctx=scene_ctx)

    if question_data is None:
        next_index = (image_index + 1) % len(images_list)
        fallback_image_name = os.path.basename(images_list[next_index])
        fallback_image_path = os.path.join("static", "images", environment, fallback_image_name)
        fallback_image_url = f"/static/images/{environment}/{fallback_image_name}"

        logger.warning("Attempt 1 failed — retrying with fallback image: %s", fallback_image_name)
        question_data = generate_question(fallback_image_path, level, asked_questions, next_type)

        if question_data is not None:
            image_url = fallback_image_url
            highlight = None
            logger.info("Retry succeeded with %s", fallback_image_name)
        else:
            logger.error(
                "Both attempts failed for %s and %s — returning question_error",
                current_image_name, fallback_image_name
            )
            return jsonify({
                "question_error": True,
                "error": "Could not generate a question right now. Please try again.",
                "image_url": image_url
            }), 200

    asked_questions.append(question_data["question"])
    question_types.append(next_type)
    session["asked_questions"] = asked_questions
    session["question_types"] = question_types
    session["current_question"] = question_data
    session["question_count"] = question_count + 1

    return jsonify({
        "question": question_data["question"],
        "hint": question_data.get("hint", "Look carefully at the image..."),
        "choices": question_data["choices"],
        "explanation": question_data.get("explanation", ""),
        "question_number": session["question_count"],
        "total_questions": Config.MAX_QUESTIONS_PER_SESSION,
        "level": level,
        "environment": environment,
        "image_url": image_url,
        "highlight": highlight,
    })


ALLOWED_ENVS = {"classroom", "supermarket", "kitchen", "fruits"}


@quiz.route("/submit_answer", methods=["POST"])
@login_required
def submit_answer():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    user_answer = data.get("answer")
    if not isinstance(user_answer, str):
        return jsonify({"error": "Invalid answer"}), 400
    user_answer = user_answer.strip()[:200]

    current_question = session.get("current_question")

    if not current_question:
        return jsonify({"error": "No active question"}), 400

    correct_answer = current_question["correct"]
    is_correct = user_answer == correct_answer

    points = 0
    if is_correct:
        session["score"] = session.get("score", 0) + 1
        points = 10

    # Collect for SRS vocabulary bank
    word = correct_answer.split(". ", 1)[-1]
    # question_type: last type stored by get_question (matches current question)
    q_types = session.get("question_types", [])
    q_type = q_types[-1] if q_types else "naming"
    vocab_answers = session.get("vocab_answers", [])
    vocab_answers.append({
        "word": word,
        "question": current_question.get("question", ""),
        "was_correct": is_correct,
        "question_type": q_type,
    })
    session["vocab_answers"] = vocab_answers

    return jsonify({
        "is_correct": is_correct,
        "correct_answer": correct_answer,
        "explanation": current_question.get("explanation", ""),
        "score": session["score"],
        "points": points
    })


@quiz.route("/end_session", methods=["POST"])
@login_required
def end_session():
    score = session.get("score", 0)
    total = Config.MAX_QUESTIONS_PER_SESSION
    score_percentage = (score / total) * 100
    current_level = session.get("level", "beginner")
    env = session.get("environment", "classroom")

    # Check if this is a manual stop (frontend sends {"stop": true})
    request_data = request.get_json(silent=True) or {}
    manual_stop = request_data.get("stop", False)

    points_earned = score * 10
    if score_percentage == 100:
        points_earned += 50
    elif score_percentage >= 80:
        points_earned += 20

    quiz_session = QuizSession(
        user_id=current_user.id,
        score=score,
        total_questions=total,
        level=current_level,
        environment=env
    )
    db.session.add(quiz_session)

    current_user.total_points += points_earned
    current_user.total_correct += score
    current_user.sessions_count += 1

    # Update per-environment score
    _update_env_score(current_user, env, points_earned, score_percentage)

    db.session.commit()

    # ── SRS vocabulary bank update ────────────────────────────────────────────
    try:
        vocab_answers = session.get("vocab_answers", [])
        now = datetime.utcnow()
        _POS_PREFIXES = ("on the", "in the", "at the", "to the", "near the",
                         "behind", "above", "below", "in front")
        _POS_WORDS    = {"left", "right", "center", "side", "bottom", "top",
                         "middle", "front", "back"}
        for entry in vocab_answers:
            word        = (entry.get("word") or "").strip()[:100]
            question    = (entry.get("question") or "").strip()
            was_correct = bool(entry.get("was_correct"))
            q_type      = (entry.get("question_type") or "naming").strip()
            # Only track naming-type answers that are real object names
            if q_type != "naming" or not word:
                continue
            wl = word.lower()
            if wl.startswith(_POS_PREFIXES) or _POS_WORDS.intersection(wl.split()):
                continue
            vw = VocabularyWord.query.filter_by(
                user_id=current_user.id, word=word, environment=env
            ).first()
            if vw:
                vw.update_sm2(was_correct)
                vw.question = question
                vw.question_type = q_type
            else:
                initial_interval = 1
                vw = VocabularyWord(
                    user_id=current_user.id,
                    word=word,
                    question=question,
                    question_type=q_type,
                    environment=env,
                    correct_count=1 if was_correct else 0,
                    wrong_count=0 if was_correct else 1,
                    ease_factor=2.5,
                    interval_days=initial_interval,
                    next_review=now + timedelta(days=initial_interval) if was_correct else now,
                    last_seen=now,
                    created_at=now,
                    is_mastered=False,
                )
                db.session.add(vw)
        db.session.commit()
        session["vocab_answers"] = []
    except Exception as exc:
        logger.error("SRS vocab update failed: %s", exc)
        db.session.rollback()

    try:
        all_sessions = (
            QuizSession.query
            .filter_by(user_id=current_user.id)
            .order_by(QuizSession.created_at.asc())
            .all()
        )
        difficulty_predictor.train(all_sessions)
    except Exception as exc:
        logger.error("difficulty_predictor.train failed: %s", exc)

    new_badges = current_user.get_badges()
    update_user_level(current_user, score_percentage)

    # Determine if the continuous session should continue
    next_level, should_continue = _next_level_after_session(current_level, score_percentage)
    if manual_stop:
        should_continue = False

    if should_continue:
        # Reset session state for next round (same environment, new level)
        images_list = session.get("images_list", [])
        random.shuffle(images_list)
        session["images_list"] = images_list
        session["image_index"] = 0
        session["score"] = 0
        session["question_count"] = 0
        session["level"] = next_level
        session["asked_questions"] = []
        session["question_types"] = []
        session["vocab_answers"] = []
        env_levels = session.get("environment_level", {})
        env_levels[env] = next_level
        session["environment_level"] = env_levels
        logger.info(
            "Continuous session: %s → %s (score=%.1f%%) env=%s",
            current_level, next_level, score_percentage, env
        )

    return jsonify({
        "score": score,
        "total": total,
        "percentage": round(score_percentage, 1),
        "new_level": current_user.level,
        "next_level": next_level,
        "continue_session": should_continue,
        "points_earned": points_earned,
        "total_points": current_user.total_points,
        "badges": new_badges
    })