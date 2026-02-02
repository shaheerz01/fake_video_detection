from flask import Flask, render_template, request, redirect, session, url_for, flash
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
import os
import torch

from model.model_def import VideoClassifier
from utils.video_utils import predict_video_file

app = Flask(__name__)
app.secret_key = "your_secret_key_123"

# ==================== DATABASE CONFIG ====================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ==================== USER MODEL ====================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())


bcrypt = Bcrypt(app)

# ==================== LOAD ML MODEL ====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VideoClassifier(pretrained=False).to(device)
model.load_state_dict(torch.load("model/video_detector1.pth", map_location=device))
model.eval()

# ==================== AUTO CREATE ADMIN ====================
def create_admin_account():
    admin = User.query.filter_by(username="admin").first()
    if not admin:
        hashed = bcrypt.generate_password_hash("admin1223").decode('utf-8')
        new_admin = User(username="admin", password=hashed)
        db.session.add(new_admin)
        db.session.commit()
        print("👑 Auto Admin Created: username=admin, password=admin1223")
    else:
        print("✔ Admin already exists")


# ==================== ROUTES ====================
@app.route('/')
def index():
    if 'username' in session:
        if session['username'] == "admin":
            return redirect(url_for('admin'))
        return redirect(url_for('dashboard'))
    return render_template('index.html')


# ---------- REGISTER ----------
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        existing = User.query.filter_by(username=username).first()
        if existing:
            flash("❌ User already exists!", "danger")
            return redirect(url_for('register'))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("✔ Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


# ---------- LOGIN ----------
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):

            session['username'] = username
            flash("✔ Login Successful!", "success")

            if username == "admin":
                return redirect(url_for('admin'))
            else:
                return redirect(url_for('dashboard'))

        flash("❌ Invalid username or password!", "danger")
        return redirect(url_for('login'))

    return render_template('login.html')


# ---------- DASHBOARD ----------
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])


# ---------- PREDICT ----------
@app.route('/predict', methods=['GET','POST'])
def predict_page():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        video = request.files.get('video')
        if not video:
            flash("❌ No video selected!", "danger")
            return redirect(url_for('predict_page'))

        allowed = ('.mp4', '.avi', '.mkv', '.mov')
        if not video.filename.lower().endswith(allowed):
            flash("❌ Unsupported video format!", "danger")
            return redirect(url_for('predict_page'))

        os.makedirs("uploads", exist_ok=True)
        save_path = os.path.join("uploads", video.filename)
        video.save(save_path)

        result = predict_video_file(save_path, model, device=device)
        if not result:
            flash("❌ Prediction failed!", "danger")
            return redirect(url_for('predict_page'))

        label, confidence = result
        session['label'] = label
        session['confidence'] = float(confidence)

        return redirect(url_for('result_page', filename=video.filename))

    return render_template('predict.html')


# ---------- RESULT ----------
@app.route('/result/<filename>')
def result_page(filename):
    if 'label' not in session:
        return redirect(url_for('predict_page'))

    return render_template('result.html',
                           filename=filename,
                           label=session['label'],
                           confidence=session['confidence'])


# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.clear()
    flash("✔ Logged out successfully!", "info")
    return redirect(url_for('index'))


# ---------- ADMIN PANEL (Protected) ----------
@app.route('/admin')
def admin():
    if 'username' not in session or session['username'] != "admin":
        flash("⛔ Access Denied! Admin Only!", "danger")
        return redirect(url_for('login'))

    users = User.query.all()
    return render_template('admin.html', users=users)


# ---------- DELETE USER ----------
@app.route('/admin/delete/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'username' not in session or session['username'] != "admin":
        flash("⛔ Access Denied!", "danger")
        return redirect(url_for('login'))

    user = User.query.get(user_id)
    if not user:
        flash("❌ User not found!", "danger")
        return redirect(url_for('admin'))

    if user.username == "admin":
        flash("⚠ Admin account cannot be deleted!", "warning")
        return redirect(url_for('admin'))

    db.session.delete(user)
    db.session.commit()
    flash(f"✔ User '{user.username}' deleted successfully!", "success")
    return redirect(url_for('admin'))


# ==================== RUN SERVER ====================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        create_admin_account()
    app.run(debug=True)
