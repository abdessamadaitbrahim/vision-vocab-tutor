import os
import logging
from flask import Flask
from extensions import db, bcrypt, login_manager
from config import Config
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _apply_schema_migrations(db_instance):
    """Add new columns to existing tables without dropping data (SQLite safe)."""
    from sqlalchemy import text, inspect
    engine = db_instance.engine
    inspector = inspect(engine)

    # New columns for users table
    user_columns = {col["name"] for col in inspector.get_columns("users")}
    new_user_cols = {
        "classroom_points": "INTEGER DEFAULT 0",
        "classroom_level":  "VARCHAR(20) DEFAULT 'beginner'",
        "kitchen_points":   "INTEGER DEFAULT 0",
        "kitchen_level":    "VARCHAR(20) DEFAULT 'beginner'",
        "supermarket_points": "INTEGER DEFAULT 0",
        "supermarket_level":  "VARCHAR(20) DEFAULT 'beginner'",
        "fruits_points":    "INTEGER DEFAULT 0",
        "fruits_level":     "VARCHAR(20) DEFAULT 'beginner'",
    }
    with engine.connect() as conn:
        for col_name, col_def in new_user_cols.items():
            if col_name not in user_columns:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_def}"))
                logger.info("Migration: added users.%s", col_name)

        # New column for quiz_sessions table
        session_columns = {col["name"] for col in inspector.get_columns("quiz_sessions")}
        if "environment" not in session_columns:
            conn.execute(text("ALTER TABLE quiz_sessions ADD COLUMN environment VARCHAR(30) DEFAULT 'classroom'"))
            logger.info("Migration: added quiz_sessions.environment")

        # New columns for vocabulary_words table
        try:
            vocab_columns = {col["name"] for col in inspector.get_columns("vocabulary_words")}
            if "question_type" not in vocab_columns:
                conn.execute(text("ALTER TABLE vocabulary_words ADD COLUMN question_type VARCHAR(20) DEFAULT 'naming'"))
                logger.info("Migration: added vocabulary_words.question_type")

            # Purge non-naming words and positional phrases stored as legacy data.
            deleted = conn.execute(text(
                "DELETE FROM vocabulary_words WHERE question_type != 'naming'"
                " OR LOWER(word) LIKE 'on the%'"
                " OR LOWER(word) LIKE 'in the%'"
                " OR LOWER(word) LIKE 'at the%'"
                " OR LOWER(word) LIKE 'to the%'"
                " OR LOWER(word) LIKE 'near the%'"
                " OR LOWER(word) LIKE 'behind%'"
                " OR LOWER(word) LIKE 'above%'"
                " OR LOWER(word) LIKE 'below%'"
                " OR LOWER(word) LIKE '% left%'"
                " OR LOWER(word) LIKE '% right%'"
                " OR LOWER(word) LIKE '% center%'"
                " OR LOWER(word) LIKE '% side%'"
                " OR LOWER(word) LIKE '% bottom%'"
            ))
            if deleted.rowcount:
                logger.info("Migration: purged %d positional/non-naming vocabulary_words", deleted.rowcount)
        except Exception:
            pass  # table may not exist yet; db.create_all() will handle it

        conn.commit()

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    with app.app_context():
        from models.user import User
        from models.session import QuizSession, Answer
        from models.vocabulary import VocabularyWord  # noqa: F401 — needed for db.create_all()

        @login_manager.user_loader
        def load_user(user_id):
            return db.session.get(User, int(user_id))

        from routes.auth import auth
        from routes.quiz import quiz
        from routes.main import main

        app.register_blueprint(auth)
        app.register_blueprint(quiz)
        app.register_blueprint(main)

        db.create_all()
        _apply_schema_migrations(db)

        # ── Context processor — vocab_due_count available in all templates ────
        from flask_login import current_user as _cu
        from datetime import datetime as _dt

        @app.context_processor
        def inject_vocab_due():
            try:
                if _cu.is_authenticated:
                    count = VocabularyWord.query.filter(
                        VocabularyWord.user_id == _cu.id,
                        VocabularyWord.next_review <= _dt.utcnow(),
                    ).count()
                    return {"vocab_due_count": count}
            except Exception:
                pass
            return {"vocab_due_count": 0}

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=os.getenv("FLASK_DEBUG", "False") == "True")
