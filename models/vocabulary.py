"""
models/vocabulary.py — Spaced Repetition vocabulary bank (SM-2 simplified).
"""

from datetime import datetime, timedelta
from extensions import db


class VocabularyWord(db.Model):
    __tablename__ = "vocabulary_words"

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    word           = db.Column(db.String(100), nullable=False)
    question       = db.Column(db.Text, nullable=False)
    environment    = db.Column(db.String(30), nullable=False)
    question_type  = db.Column(db.String(20), default="naming", nullable=False)
    correct_count  = db.Column(db.Integer, default=0, nullable=False)
    wrong_count    = db.Column(db.Integer, default=0, nullable=False)
    ease_factor    = db.Column(db.Float, default=2.5, nullable=False)
    interval_days  = db.Column(db.Integer, default=1, nullable=False)
    next_review    = db.Column(db.DateTime, nullable=True)
    last_seen      = db.Column(db.DateTime, nullable=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    is_mastered    = db.Column(db.Boolean, default=False, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("user_id", "word", "environment", name="uq_user_word_env"),
    )

    def update_sm2(self, was_correct: bool) -> None:
        """Update SM-2 fields after an answer."""
        now = datetime.utcnow()
        if was_correct:
            self.correct_count += 1
            self.ease_factor = min(self.ease_factor + 0.1, 4.0)
            new_interval = max(1, min(30, round(self.interval_days * self.ease_factor)))
            self.interval_days = new_interval
        else:
            self.wrong_count += 1
            self.ease_factor = max(1.3, self.ease_factor - 0.3)
            self.interval_days = 1

        self.next_review = now + timedelta(days=self.interval_days)
        self.last_seen = now
        self.is_mastered = (self.correct_count >= 3 and self.ease_factor >= 2.0)

    def is_due(self) -> bool:
        """Return True if this word is due for review."""
        if self.next_review is None:
            return True
        return self.next_review <= datetime.utcnow()

    def __repr__(self):
        return f"<VocabularyWord {self.word!r} env={self.environment} user={self.user_id}>"
