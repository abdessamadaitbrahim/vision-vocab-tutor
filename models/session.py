from datetime import datetime
from extensions import db

class QuizSession(db.Model):
    __tablename__ = "quiz_sessions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    score = db.Column(db.Integer, default=0)
    total_questions = db.Column(db.Integer, default=10)
    level = db.Column(db.String(20), nullable=False)
    environment = db.Column(db.String(30), default="classroom")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    answers = db.relationship("Answer", backref="session", lazy=True)

    def get_percentage(self):
        if self.total_questions == 0:
            return 0
        return (self.score / self.total_questions) * 100

class Answer(db.Model):
    __tablename__ = "answers"
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("quiz_sessions.id"), nullable=False)
    question = db.Column(db.Text, nullable=False)
    correct_answer = db.Column(db.String(255), nullable=False)
    user_answer = db.Column(db.String(255), nullable=False)
    is_correct = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)