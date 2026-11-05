from flask import Blueprint, render_template, session, redirect, url_for, jsonify, request, flash
from flask_login import login_required, current_user
from models.session import QuizSession
from models.vocabulary import VocabularyWord
from extensions import db
from ai_engine import difficulty_predictor
from datetime import datetime
import json
import logging
import random
import os
import re
import requests

logger = logging.getLogger(__name__)

main = Blueprint("main", __name__)


@main.route("/")
@login_required
def home():
    recent_sessions = QuizSession.query.filter_by(
        user_id=current_user.id
    ).order_by(QuizSession.created_at.desc()).limit(5).all()

    return render_template("home.html", user=current_user, recent_sessions=recent_sessions)


@main.route("/profile")
@login_required
def profile():
    all_sessions = QuizSession.query.filter_by(
        user_id=current_user.id
    ).order_by(QuizSession.created_at.desc()).all()

    total_sessions = len(all_sessions)
    total_correct = sum(s.score for s in all_sessions)
    total_questions = sum(s.total_questions for s in all_sessions)
    average_score = round((total_correct / total_questions) * 100, 1) if total_questions > 0 else 0

    return render_template("profile.html", user=current_user, all_sessions=all_sessions,
                           total_sessions=total_sessions, average_score=average_score)


def _score_to_level(pct: float) -> str:
    """Mirror the update_user_level() logic so distributions are consistent."""
    if pct >= 80:
        return "advanced"
    elif pct >= 50:
        return "intermediate"
    return "beginner"


@main.route("/dashboard")
@login_required
def dashboard():
    try:
        # Oldest-first for chart continuity; newest-first slice for table
        all_sessions_asc = QuizSession.query.filter_by(
            user_id=current_user.id
        ).order_by(QuizSession.created_at.asc()).all()

        recent_sessions = list(reversed(all_sessions_asc))[:10]

        total_sessions = len(all_sessions_asc)

        # ── Aggregate stats ──────────────────────────────────────────────────
        # Use user model fields for total_points / total_correct — these are
        # the authoritative values shown in the navbar, guaranteeing consistency.
        total_points  = current_user.total_points
        total_correct = current_user.total_correct

        if all_sessions_asc:
            total_q   = sum(s.total_questions for s in all_sessions_asc)
            total_s   = sum(s.score for s in all_sessions_asc)
            avg_score = round((total_s / total_q) * 100, 1) if total_q > 0 else 0
            best_score = round(max(s.get_percentage() for s in all_sessions_asc), 1)
        else:
            avg_score  = 0
            best_score = 0

        # ── Level distribution — derived from SCORE, not session.level ───────
        # session.level stores the *starting* difficulty; the *resulting* level
        # is what update_user_level() computes from the score percentage.
        level_counts = {"beginner": 0, "intermediate": 0, "advanced": 0}
        for s in all_sessions_asc:
            resulting_level = _score_to_level(s.get_percentage())
            level_counts[resulting_level] += 1

        # ── Chart.js data ────────────────────────────────────────────────────
        chart_labels  = [str(i + 1) for i in range(total_sessions)]
        chart_scores  = [round(s.get_percentage(), 1) for s in all_sessions_asc]
        # Derive chart level label from score too, for tooltip consistency
        chart_levels  = [_score_to_level(s.get_percentage()) for s in all_sessions_asc]

        point_colors = []
        for lvl in chart_levels:
            if lvl == "advanced":
                point_colors.append("rgba(25, 135, 84, 1)")    # green
            elif lvl == "intermediate":
                point_colors.append("rgba(255, 193, 7, 1)")    # yellow
            else:
                point_colors.append("rgba(220, 53, 69, 1)")    # red

        # ── ML prediction ────────────────────────────────────────────────────
        difficulty_predictor.train(all_sessions_asc)
        predicted_level = difficulty_predictor.predict_next_level(all_sessions_asc)
        ml_confidence   = difficulty_predictor.get_confidence(all_sessions_asc)

        # ── Per-environment stats ─────────────────────────────────────────────
        _envs = ["classroom", "kitchen", "supermarket", "fruits"]
        env_stats = {}
        for env_name in _envs:
            env_sessions = [s for s in all_sessions_asc
                            if getattr(s, "environment", None) == env_name]
            if env_sessions:
                env_avg = round(sum(s.get_percentage() for s in env_sessions) / len(env_sessions), 1)
                env_best = round(max(s.get_percentage() for s in env_sessions), 1)
            else:
                env_avg = 0
                env_best = 0
            env_pred = difficulty_predictor.predict_next_level(all_sessions_asc, environment=env_name)
            env_conf = difficulty_predictor.get_confidence(all_sessions_asc, environment=env_name)
            env_stats[env_name] = {
                "sessions": len(env_sessions),
                "avg_score": env_avg,
                "best_score": env_best,
                "points": getattr(current_user, f"{env_name}_points", 0),
                "level": getattr(current_user, f"{env_name}_level", "beginner") or "beginner",
                "predicted_level": env_pred,
                "confidence": env_conf,
            }

        # ── Vocabulary stats for dashboard mini-section ───────────────────────
        now = datetime.utcnow()
        vocab_total    = VocabularyWord.query.filter_by(user_id=current_user.id).count()
        vocab_mastered = VocabularyWord.query.filter_by(user_id=current_user.id, is_mastered=True).count()
        vocab_due      = VocabularyWord.query.filter(
            VocabularyWord.user_id == current_user.id,
            VocabularyWord.next_review <= now,
        ).count()

        return render_template(
            "dashboard.html",
            user=current_user,
            total_sessions=total_sessions,
            avg_score=avg_score,
            best_score=best_score,
            total_points=total_points,
            total_correct=total_correct,
            level_counts=level_counts,
            recent_sessions=recent_sessions,
            chart_labels=json.dumps(chart_labels),
            chart_scores=json.dumps(chart_scores),
            chart_levels=json.dumps(chart_levels),
            point_colors=json.dumps(point_colors),
            has_data=(total_sessions > 0),
            predicted_level=predicted_level,
            ml_confidence=ml_confidence,
            env_stats=env_stats,
            vocab_total=vocab_total,
            vocab_mastered=vocab_mastered,
            vocab_due=vocab_due,
        )
    except Exception as exc:
        logger.error("dashboard route error: %s", exc)
        _default_env_stats = {env_name: {"sessions": 0, "avg_score": 0, "best_score": 0,
                                         "points": 0, "level": "beginner",
                                         "predicted_level": "beginner", "confidence": 0}
                              for env_name in ["classroom", "kitchen", "supermarket", "fruits"]}
        return render_template(
            "dashboard.html", user=current_user, has_data=False,
            total_sessions=0, avg_score=0, best_score=0,
            total_points=current_user.total_points,
            total_correct=current_user.total_correct,
            level_counts={"beginner": 0, "intermediate": 0, "advanced": 0},
            recent_sessions=[], chart_labels="[]", chart_scores="[]",
            chart_levels="[]", point_colors="[]",
            predicted_level="beginner", ml_confidence=0,
            env_stats=_default_env_stats,
            vocab_total=0, vocab_mastered=0, vocab_due=0,
        )


_ENVS = ["classroom", "kitchen", "supermarket", "fruits"]


@main.route("/vocabulary")
@login_required
def vocabulary():
    now = datetime.utcnow()
    uid = current_user.id

    words_due = (
        VocabularyWord.query
        .filter(VocabularyWord.user_id == uid, VocabularyWord.next_review <= now)
        .order_by(VocabularyWord.next_review.asc())
        .limit(20)
        .all()
    )

    mastered_count   = VocabularyWord.query.filter_by(user_id=uid, is_mastered=True).count()
    total_words      = VocabularyWord.query.filter_by(user_id=uid).count()
    to_review_count  = len(words_due)

    words_by_env = {}
    for env_name in _ENVS:
        total   = VocabularyWord.query.filter_by(user_id=uid, environment=env_name).count()
        mastered = VocabularyWord.query.filter_by(user_id=uid, environment=env_name, is_mastered=True).count()
        due     = VocabularyWord.query.filter(
            VocabularyWord.user_id == uid,
            VocabularyWord.environment == env_name,
            VocabularyWord.next_review <= now,
        ).count()
        words_by_env[env_name] = {"total": total, "mastered": mastered, "due": due}

    all_words = (
        VocabularyWord.query
        .filter_by(user_id=uid)
        .order_by(VocabularyWord.next_review.asc())
        .limit(50)
        .all()
    )

    return render_template(
        "vocabulary.html",
        words_due=words_due,
        mastered_count=mastered_count,
        total_words=total_words,
        to_review_count=to_review_count,
        words_by_env=words_by_env,
        all_words=all_words,
        now=now,
    )


# ── SRS Flashcard Review ─────────────────────────────────────────────────────

@main.route("/vocabulary/review")
@login_required
def vocabulary_review():
    now = datetime.utcnow()
    uid = current_user.id

    due_words = (
        VocabularyWord.query
        .filter(VocabularyWord.user_id == uid, VocabularyWord.next_review <= now)
        .order_by(VocabularyWord.next_review.asc())
        .limit(20)
        .all()
    )

    if not due_words:
        flash("No words due for review right now — great job!", "success")
        return redirect(url_for("main.vocabulary"))

    # Store queue of IDs in session
    session["review_queue"] = [w.id for w in due_words]
    session["review_index"] = 0
    session["review_correct"] = 0

    return render_template(
        "vocabulary_review.html",
        total=len(due_words),
    )


_ENV_EMOJIS = {"classroom": "🏫", "kitchen": "🍳", "supermarket": "🛒", "fruits": "🍎"}

# In-memory cache: (word_lower, environment) → {question, correct, wrong: [w1, w2, w3]}
_review_question_cache: dict = {}

_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_OR_URL     = "https://openrouter.ai/api/v1/chat/completions"
_OR_MODEL   = "google/gemma-3-27b-it:free"


def _ai_generate_review_question(word: str, environment: str) -> dict | None:
    """
    Ask the AI to generate a comprehension question for a vocabulary word.
    Returns {question, correct, wrong: [w1, w2, w3]} or None on failure.
    Results are cached in _review_question_cache to avoid redundant API calls.
    """
    cache_key = (word.lower(), environment)
    if cache_key in _review_question_cache:
        return _review_question_cache[cache_key]

    _env_context = {
        "classroom":   "a classroom object",
        "kitchen":     "a kitchen object or food item",
        "supermarket": "a supermarket item",
        "fruits":      "a type of fruit",
    }
    env_desc = _env_context.get(environment, "an everyday object")

    prompt = f"""You are an English vocabulary teacher creating a review flashcard.

Word: "{word}" ({env_desc})

Write ONE comprehension question that tests understanding of this word.
Rules:
- Test FUNCTION, CATEGORY, or CHARACTERISTICS — never ask where to find it
- Good question formats: "{word} is used for...?", "A {word} is...?", "What do you do with a {word}?"
- Provide exactly 1 short correct answer (3-6 words)
- Provide exactly 3 wrong but plausible answers (3-6 words each)
- All 4 answers must be the same type (all uses, all descriptions, etc.)
- Use simple English

Respond with JSON only, no extra text:
{{"question": "...", "correct": "...", "wrong": ["...", "...", "..."]}}"""

    def _call(url, headers, payload):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            if r.status_code == 429:
                return None
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Review AI call failed (%s): %s", url, exc)
            return None

    # Try Groq first (text-only, no image)
    groq_key = os.getenv("GROQ_API_KEY", "")
    content = None
    if groq_key:
        payload = {
            "model": _GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
        }
        content = _call(_GROQ_URL, {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}, payload)

    # Fallback: OpenRouter
    if not content:
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_key:
            payload = {
                "model": _OR_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
            }
            content = _call(_OR_URL, {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}, payload)

    if not content:
        return None

    try:
        text = re.sub(r"```json\s*|```\s*", "", content).strip()
        data = json.loads(text)
        result = {
            "question": data["question"],
            "correct":  data["correct"],
            "wrong":    data["wrong"][:3],
        }
        _review_question_cache[cache_key] = result
        logger.debug("AI review question generated for '%s': %s", word, result["question"])
        return result
    except Exception as exc:
        logger.error("Failed to parse AI review question for '%s': %s | raw: %s", word, exc, content[:200])
        return None


def _fallback_review_question(word: str, environment: str) -> dict:
    """Static fallback when AI is unavailable."""
    _env_labels = {
        "classroom":   "a classroom object",
        "kitchen":     "a kitchen object",
        "supermarket": "a supermarket item",
        "fruits":      "a type of fruit",
    }
    _env_wrong = {
        "classroom":   ["a type of fruit", "a kitchen object", "a supermarket item"],
        "kitchen":     ["a classroom object", "a type of fruit", "a supermarket item"],
        "supermarket": ["a classroom object", "a kitchen object", "a type of fruit"],
        "fruits":      ["a kitchen object", "a classroom object", "a supermarket item"],
    }
    return {
        "question": f'"{word}" is...?',
        "correct":  _env_labels.get(environment, environment),
        "wrong":    _env_wrong.get(environment, ["an animal", "a vehicle", "a sport"]),
    }


@main.route("/vocabulary/review/card")
@login_required
def vocabulary_review_card():
    queue = session.get("review_queue", [])
    index = session.get("review_index", 0)

    if index >= len(queue):
        return jsonify({"session_complete": True,
                        "correct": session.get("review_correct", 0),
                        "total": len(queue)})

    word_id = queue[index]
    vw = db.session.get(VocabularyWord, word_id)
    if not vw or vw.user_id != current_user.id:
        return jsonify({"error": "Word not found"}), 404

    env_emoji = _ENV_EMOJIS.get(vw.environment, "📍")

    qdata = _ai_generate_review_question(vw.word, vw.environment) or \
            _fallback_review_question(vw.word, vw.environment)

    wrong_sample = random.sample(qdata["wrong"], min(2, len(qdata["wrong"])))
    choices_raw = [qdata["correct"]] + wrong_sample
    random.shuffle(choices_raw)
    labeled = [f"{chr(65 + i)}. {c}" for i, c in enumerate(choices_raw)]
    correct_labeled = next(c for c in labeled if c.split(". ", 1)[-1] == qdata["correct"])

    return jsonify({
        "word_id": vw.id,
        "word_display": vw.word,
        "review_question": qdata["question"],
        "choices": labeled,
        "correct": correct_labeled,
        "environment": vw.environment,
        "env_emoji": env_emoji,
        "index": index,
        "total": len(queue),
        "correct_count": vw.correct_count,
        "wrong_count": vw.wrong_count,
        "ease_factor": round(vw.ease_factor, 2),
        "interval_days": vw.interval_days,
        "is_mastered": vw.is_mastered,
    })


@main.route("/vocabulary/review/answer", methods=["POST"])
@login_required
def vocabulary_review_answer():
    data = request.get_json(silent=True) or {}
    word_id    = data.get("word_id")
    was_correct = bool(data.get("was_correct"))

    if not word_id:
        return jsonify({"error": "Missing word_id"}), 400

    vw = db.session.get(VocabularyWord, int(word_id))
    if not vw or vw.user_id != current_user.id:
        return jsonify({"error": "Word not found"}), 404

    vw.update_sm2(was_correct)
    db.session.commit()

    if was_correct:
        session["review_correct"] = session.get("review_correct", 0) + 1

    # Advance queue
    new_index = session.get("review_index", 0) + 1
    session["review_index"] = new_index
    queue_len = len(session.get("review_queue", []))

    return jsonify({
        "was_correct": was_correct,
        "new_interval": vw.interval_days,
        "new_ease": round(vw.ease_factor, 2),
        "is_mastered": vw.is_mastered,
        "session_complete": new_index >= queue_len,
        "correct": session.get("review_correct", 0),
        "total": queue_len,
    })