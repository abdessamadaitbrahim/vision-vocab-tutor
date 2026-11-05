import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Security — must be set in .env, no hardcoded fallback
    SECRET_KEY = os.getenv("SECRET_KEY")
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY missing from .env!")

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///vocab_tutor.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Debug — read from .env, default False for production safety
    DEBUG = os.getenv("FLASK_DEBUG", "False") == "True"

    # Secure session cookies
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = 3600

    # Groq API (primary LLM provider)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # App settings
    MAX_QUESTIONS_PER_SESSION = 10

    # Difficulty levels scores
    SCORE_BEGINNER = 0        # 0% - 49%
    SCORE_INTERMEDIATE = 50   # 50% - 79%
    SCORE_ADVANCED = 80       # 80% et plus

    # Difficulty labels
    LEVEL_BEGINNER = "beginner"
    LEVEL_INTERMEDIATE = "intermediate"
    LEVEL_ADVANCED = "advanced"