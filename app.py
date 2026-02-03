from flask import Flask, render_template, request, redirect, session, url_for, flash
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
import os
import torch

from model.model_def import VideoClassifier
from utils.video_utils import predict_video_file

# ==================== APP CONFIG ====================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key")

# Limit upload size (IMPORTANT for Render)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================== DATABASE ====================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ==================== USER MODEL ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# ==================== LOAD ML MODEL (CPU ONLY) ====================
device = "cpu"

def load_model():
    model = VideoClassifier(pretrained=False)
    model_path = os.path.join(BASE_DIR, "model", "video_detector1.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ==================== AUTO CREATE ADMIN ====================
def create_admin_account():
    admin = User.query.filter_by(username="admin").first()
    if not admin:
        hashed = bcrypt.generate_password_hash("admin1223").decode("utf-8")
        db.session.add(User(username="admin", password=hashed))
        db.session.commit()

# ==================== ROUTES ====================
@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("admin" if session["username"] == "admin" else "dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        if User.query.filter_by(username=request.form["username"]).first():
            flash("User already exists", "danger")
            return redirect(url_for("register"))

        hashed = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")
        db.session.add(User(username=request.form["username"], password=hashed))
        db.session.commit()
        flash("Registration successful", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and bcrypt.check_password_hash(user.password, request.form["password"]):
            session["username"] = user.username
            return redirect(url_for("admin" if user.username == "admin" else "dashboard"))

        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            video = request.files.get("video")
            if not video or video.filename == "":
                flash("No video selected", "danger")
                return redirect(url_for("predict_page"))

            save_path = os.path.join(UPLOAD_DIR, video.filename)
            video.save(save_path)

            label, confidence = predict_video_file(save_path, model, device=device)
            if label is None:
                raise RuntimeError("Prediction failed")

            session["label"] = label
            session["confidence"] = confidence
            return redirect(url_for("result_page", filename=video.filename))

        except Exception as e:
            print("Prediction error:", e)
            flash("Prediction failed. Try a smaller video.", "danger")
            return redirect(url_for("predict_page"))

    return render_template("predict.html")

@app.route("/result/<filename>")
def result_page(filename):
    if "label" not in session:
        return redirect(url_for("predict_page"))

    return render_template(
        "result.html",
        filename=filename,
        label=session["label"],
        confidence=session["confidence"]
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/admin")
def admin():
    if session.get("username") != "admin":
        return redirect(url_for("login"))
    return render_template("admin.html", users=User.query.all())

@app.route("/admin/delete/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    if session.get("username") != "admin":
        return redirect(url_for("login"))

    user = User.query.get(user_id)
    if user and user.username != "admin":
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for("admin"))

# ==================== START ====================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        create_admin_account()
    app.run()
