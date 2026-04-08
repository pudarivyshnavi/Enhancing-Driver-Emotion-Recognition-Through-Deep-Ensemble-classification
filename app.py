from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import json
import uuid
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from driver_monitoring import analyze_driver_video
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'generated_results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='user', lazy=True)
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_filename = db.Column(db.String(200), nullable=False)
    original_video = db.Column(db.String(200), nullable=False)
    concentration = db.Column(db.Float)
    driving_accuracy = db.Column(db.Float)
    fit_for_driving = db.Column(db.String(20))
    primary_emotion = db.Column(db.String(50))
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)
    concentration_graph = db.Column(db.String(200))
    emotion_pie = db.Column(db.String(200))
    emotion_bar = db.Column(db.String(200))
    dashboard = db.Column(db.String(200))
    json_report = db.Column(db.String(200))
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
with app.app_context():
    db.create_all()
def save_analysis_results(user_id, video_path, results):
    """Save analysis results to database"""
    unique_id = str(uuid.uuid4())[:8]
    import shutil
    files_map = {}
    for filename in ['concentration_timeline.png', 'emotion_distribution_pie.png', 
                     'emotion_bar_chart.png', 'driver_dashboard.png', 'driver_report.json']:
        if os.path.exists(filename):
            new_name = f"{unique_id}_{filename}"
            new_path = os.path.join(app.config['RESULTS_FOLDER'], new_name)
            shutil.move(filename, new_path)
            files_map[filename] = new_name
    video_filename = secure_filename(os.path.basename(video_path))
    unique_video = f"{unique_id}_{video_filename}"
    video_save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_video)
    if os.path.exists(video_path):
        shutil.copy(video_path, video_save_path)
    analysis = Analysis(
        user_id=user_id,
        video_filename=unique_video,
        original_video=video_filename,
        concentration=results.get('concentration_level_percent'),
        driving_accuracy=results.get('driving_accuracy_percent'),
        fit_for_driving=results.get('fit_for_driving'),
        primary_emotion=results.get('primary_emotion'),
        concentration_graph=files_map.get('concentration_timeline.png'),
        emotion_pie=files_map.get('emotion_distribution_pie.png'),
        emotion_bar=files_map.get('emotion_bar_chart.png'),
        dashboard=files_map.get('driver_dashboard.png'),
        json_report=files_map.get('driver_report.json')
    )
    db.session.add(analysis)
    db.session.commit()
    return analysis
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already registered')
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))
@app.route('/dashboard')
@login_required
def dashboard():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.analyzed_at.desc()).all()
    return render_template('dashboard.html', analyses=analyses)
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('upload.html', error='No file selected')
        file = request.files['video']
        if file.filename == '':
            return render_template('upload.html', error='No file selected')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            session['video_to_analyze'] = filepath
            return redirect(url_for('analyze'))
        else:
            return render_template('upload.html', error='Invalid file format. Please upload MP4 file.')
    return render_template('upload.html')
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}
@app.route('/analyze')
@login_required
def analyze():
    video_path = session.get('video_to_analyze')
    if not video_path or not os.path.exists(video_path):
        return redirect(url_for('upload'))
    try:
        results = analyze_driver_video(video_path)
        if results:
            analysis = save_analysis_results(current_user.id, video_path, results)
            
            # Clear session
            session.pop('video_to_analyze', None)
            
            return redirect(url_for('results', analysis_id=analysis.id))
        else:
            return render_template('upload.html', error='Analysis failed. Please try again.')
    
    except Exception as e:
        return render_template('upload.html', error=f'Error during analysis: {str(e)}')

@app.route('/results/<int:analysis_id>')
@login_required
def results(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Verify ownership
    if analysis.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    
    return render_template('results.html', analysis=analysis)

@app.route('/download/<int:analysis_id>/<file_type>')
@login_required
def download_file(analysis_id, file_type):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    if analysis.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    
    file_map = {
        'concentration': analysis.concentration_graph,
        'pie': analysis.emotion_pie,
        'bar': analysis.emotion_bar,
        'dashboard': analysis.dashboard,
        'json': analysis.json_report,
        'video': analysis.video_filename
    }
    
    filename = file_map.get(file_type)
    if filename:
        if file_type == 'video':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        else:
            filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        if os.path.exists(filepath):
            from flask import send_file
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return f"File not found: {filename}", 404
    
    return redirect(url_for('results', analysis_id=analysis_id))

@app.route('/delete_analysis/<int:analysis_id>')
@login_required
def delete_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    if analysis.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    
    # Delete files
    files_to_delete = [
        analysis.concentration_graph,
        analysis.emotion_pie,
        analysis.emotion_bar,
        analysis.dashboard,
        analysis.json_report,
        analysis.video_filename
    ]
    
    for file in files_to_delete:
        if file:
            filepath = os.path.join(app.config['RESULTS_FOLDER'], file) if 'graph' in str(file) or 'dashboard' in str(file) or 'json' in str(file) else os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.exists(filepath):
                os.remove(filepath)
    
    db.session.delete(analysis)
    db.session.commit()
    
    return redirect(url_for('dashboard'))


@app.route('/generated_results/<filename>')
def serve_generated_result(filename):
    """Serve generated result files"""
    from flask import send_from_directory
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

# Add route for uploads folder
@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded video files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/debug/files')
@login_required
def debug_files():
    """Debug endpoint to check what files exist"""
    import os
    files = {
        'generated_results': os.listdir(app.config['RESULTS_FOLDER']) if os.path.exists(app.config['RESULTS_FOLDER']) else [],
        'uploads': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    }
    return jsonify(files)


    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)