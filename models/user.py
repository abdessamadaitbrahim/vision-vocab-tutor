from flask_login import UserMixin
from datetime import datetime
from extensions import db, bcrypt

class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    level = db.Column(db.String(20), default="beginner")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    total_points = db.Column(db.Integer, default=0)
    total_correct = db.Column(db.Integer, default=0)
    sessions_count = db.Column(db.Integer, default=0)
    # Per-environment progress
    classroom_points = db.Column(db.Integer, default=0)
    classroom_level = db.Column(db.String(20), default="beginner")
    kitchen_points = db.Column(db.Integer, default=0)
    kitchen_level = db.Column(db.String(20), default="beginner")
    supermarket_points = db.Column(db.Integer, default=0)
    supermarket_level = db.Column(db.String(20), default="beginner")
    fruits_points = db.Column(db.Integer, default=0)
    fruits_level = db.Column(db.String(20), default="beginner")
    sessions = db.relationship("QuizSession", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def get_badges(self):
        badges = []
        if self.total_correct >= 1:
            badges.append({"icon": "⭐", "name": "First Star", "desc": "First correct answer!"})
        if self.total_correct >= 10:
            badges.append({"icon": "🌟", "name": "Vocabulary Star", "desc": "10 correct answers!"})
        if self.total_correct >= 50:
            badges.append({"icon": "🔥", "name": "On Fire", "desc": "50 correct answers!"})
        if self.total_correct >= 100:
            badges.append({"icon": "💎", "name": "Diamond", "desc": "100 correct answers!"})
        if self.sessions_count >= 1:
            badges.append({"icon": "🎮", "name": "First Session", "desc": "Completed first session!"})
        if self.sessions_count >= 5:
            badges.append({"icon": "🏅", "name": "Regular Player", "desc": "5 sessions completed!"})
        if self.sessions_count >= 10:
            badges.append({"icon": "🏆", "name": "Champion", "desc": "10 sessions completed!"})
        if self.level == "advanced":
            badges.append({"icon": "🎓", "name": "Advanced", "desc": "Reached advanced level!"})
        return badges

    def __repr__(self):
        return f"<User {self.username} - Level: {self.level}>"