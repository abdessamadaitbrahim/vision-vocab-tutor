import time
from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db, bcrypt
from models.user import User

auth = Blueprint("auth", __name__)

_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 300  # 5 minutes

@auth.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    if request.method == "POST":
        username = request.form.get("username").strip()
        email = request.form.get("email").strip().lower()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("auth.register"))

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("auth.register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("auth.register"))

        if User.query.filter_by(email=email).first():
            flash("Email already exists.", "danger")
            return redirect(url_for("auth.register"))

        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
            return redirect(url_for("auth.register"))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")

@auth.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    if request.method == "POST":
        # Brute force protection
        attempts = session.get("login_attempts", 0)
        lockout_until = session.get("login_lockout_until", 0)
        now = time.time()

        if now < lockout_until:
            remaining = int(lockout_until - now)
            flash(f"Too many attempts. Try again in {remaining // 60 + 1} minute(s).", "danger")
            return redirect(url_for("auth.login"))

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session.pop("login_attempts", None)
            session.pop("login_lockout_until", None)
            login_user(user)
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for("main.home"))
        else:
            attempts += 1
            session["login_attempts"] = attempts
            if attempts >= _MAX_ATTEMPTS:
                session["login_lockout_until"] = now + _LOCKOUT_SECONDS
                session["login_attempts"] = 0
                flash("Too many attempts. Try again in 5 minutes.", "danger")
            else:
                remaining_tries = _MAX_ATTEMPTS - attempts
                flash(f"Invalid email or password. {remaining_tries} attempt(s) left.", "danger")
            return redirect(url_for("auth.login"))

    return render_template("login.html")

@auth.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))