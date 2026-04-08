"""
Driver Monitoring System - Complete Version with Graphs and Visualizations
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import json
from collections import deque
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Set style for better looking graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Check if dlib is available
try:
    import dlib
    from scipy.spatial import distance as dist
    DLIB_AVAILABLE = True
    print("✅ dlib loaded successfully")
except ImportError:
    DLIB_AVAILABLE = False
    print("⚠️ dlib not available - using simplified detection")

# -------------------------------
# Emotion Predictor
# -------------------------------
class EmotionPredictor:
    def __init__(self, model_path='models/emotion_model.h5', 
                 labels_path='models/emotion_labels.json'):
        self.model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                self.model = None
        else:
            print(f"⚠️ Model not found at {model_path}")
            print("   Using fallback emotion detection")
        
        if os.path.exists(labels_path):
            try:
                with open(labels_path, 'r') as f:
                    self.emotion_labels = json.load(f)
            except:
                pass
        
        self.img_size = (224, 224)
    
    def predict(self, face_img):
        if self.model is None or face_img.size == 0:
            return self._simple_emotion_guess(face_img), 0.5, None
        
        try:
            face_img = cv2.resize(face_img, self.img_size)
            face_img = np.expand_dims(face_img, axis=0)
            face_img = face_img.astype(np.float32) / 255.0
            
            pred = self.model.predict(face_img, verbose=0)[0]
            emotion = self.emotion_labels[np.argmax(pred)]
            confidence = np.max(pred)
            return emotion, confidence, pred
        except:
            return "neutral", 0.5, None
    
    def _simple_emotion_guess(self, face_img):
        if face_img.size == 0:
            return "neutral"
        
        try:
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            avg_saturation = np.mean(hsv[:, :, 1])
            
            if avg_saturation > 100:
                return "happy"
            elif avg_saturation < 50:
                return "sad"
            else:
                return "neutral"
        except:
            return "neutral"

# -------------------------------
# Driver Monitor with History Tracking
# -------------------------------
class DriverMonitorWithHistory:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # History tracking for graphs
        self.emotion_history = []  # Store emotions over time
        self.concentration_history = []  # Store concentration scores
        self.timestamp_history = []  # Store timestamps
        self.emotion_buffer = deque(maxlen=150)
        self.face_size_buffer = deque(maxlen=150)
        
        self.total_frames = 0
        self.frames_with_face = 0
        self.frame_timestamp = 0
        
    def process_frame(self, frame, emotion_predictor, fps):
        self.total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return frame, None
        
        self.frames_with_face += 1
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces[0]
        
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size > 0 and face_roi.shape[0] > 50 and face_roi.shape[1] > 50:
            emotion, confidence, _ = emotion_predictor.predict(face_roi)
        else:
            emotion, confidence = "neutral", 0.5
        
        # Store history for graphs
        current_time = self.frame_timestamp / fps
        self.emotion_history.append((current_time, emotion, confidence))
        self.emotion_buffer.append(emotion)
        self.face_size_buffer.append(w * h)
        
        # Calculate current concentration
        concentration = self._calculate_current_concentration()
        self.concentration_history.append((current_time, concentration))
        self.timestamp_history.append(current_time)
        
        self.frame_timestamp += 1
        
        # Draw annotations
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Emotion label with background
        label = f"{emotion} ({confidence:.2f})"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y-label_h-10), (x+label_w+10, y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x+5, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Concentration meter
        cv2.putText(frame, f"Concentration: {concentration:.0f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, {
            'emotion': emotion,
            'confidence': confidence,
            'concentration': concentration,
            'face_size': w * h,
            'timestamp': current_time
        }
    
    def _calculate_current_concentration(self):
        """Calculate current concentration from recent buffer"""
        if len(self.emotion_buffer) == 0:
            return 50.0
        
        emotion_scores = {
            'angry': 20, 'disgust': 25, 'fear': 25, 
            'happy': 85, 'neutral': 70, 'sad': 30, 'surprise': 60, 
            'unknown': 50
        }
        
        avg_emotion_score = np.mean([emotion_scores.get(e, 50) for e in self.emotion_buffer])
        
        if len(self.face_size_buffer) > 10:
            face_variation = np.std(self.face_size_buffer) / (np.mean(self.face_size_buffer) + 1e-6)
            stability_score = max(0, 100 - min(100, face_variation * 100))
        else:
            stability_score = 70
        
        concentration = 0.7 * avg_emotion_score + 0.3 * stability_score
        return min(100, max(0, concentration))
    
    def get_final_metrics(self):
        """Calculate final metrics from all history"""
        if len(self.emotion_buffer) == 0:
            return 50.0, 50.0, "unknown", {}
        
        emotion_scores = {
            'angry': 20, 'disgust': 25, 'fear': 25, 
            'happy': 85, 'neutral': 70, 'sad': 30, 'surprise': 60
        }
        
        avg_emotion_score = np.mean([emotion_scores.get(e, 50) for e in self.emotion_buffer])
        
        if len(self.face_size_buffer) > 10:
            face_variation = np.std(self.face_size_buffer) / (np.mean(self.face_size_buffer) + 1e-6)
            stability_score = max(0, 100 - min(100, face_variation * 100))
        else:
            stability_score = 70
        
        concentration = 0.7 * avg_emotion_score + 0.3 * stability_score
        concentration = min(100, max(0, concentration))
        
        negative_emotions = ['angry', 'fear', 'sad', 'disgust']
        neg_count = sum(1 for e in self.emotion_buffer if e in negative_emotions)
        neg_ratio = neg_count / len(self.emotion_buffer) if self.emotion_buffer else 0
        
        accuracy = concentration - (neg_ratio * 30)
        accuracy = min(100, max(0, accuracy))
        
        emotion_counts = {}
        for e in self.emotion_buffer:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        primary_emotion = max(emotion_counts, key=emotion_counts.get)
        
        return concentration, accuracy, primary_emotion, emotion_counts

# -------------------------------
# Graph Generation Functions
# -------------------------------

def create_concentration_timeline(concentration_history, duration):
    """Create concentration over time graph"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    times = [h[0] for h in concentration_history]
    concentrations = [h[1] for h in concentration_history]
    
    # Plot line with gradient
    ax.plot(times, concentrations, linewidth=2.5, color='#2E86AB', alpha=0.8)
    ax.fill_between(times, concentrations, alpha=0.3, color='#2E86AB')
    
    # Add threshold lines
    ax.axhline(y=60, color='green', linestyle='--', linewidth=2, label='Safe Threshold (60%)', alpha=0.7)
    ax.axhline(y=40, color='orange', linestyle='--', linewidth=2, label='Warning Threshold (40%)', alpha=0.7)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Critical Threshold (20%)', alpha=0.7)
    
    # Color zones
    ax.axhspan(60, 100, alpha=0.1, color='green', label='Safe Zone')
    ax.axhspan(40, 60, alpha=0.1, color='yellow', label='Caution Zone')
    ax.axhspan(0, 40, alpha=0.1, color='red', label='Danger Zone')
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Concentration Level (%)', fontsize=12, fontweight='bold')
    ax.set_title('Driver Concentration Over Time', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add average line
    avg_concentration = np.mean(concentrations)
    ax.axhline(y=avg_concentration, color='blue', linestyle='-', linewidth=2, 
               label=f'Average: {avg_concentration:.1f}%', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('concentration_timeline.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: concentration_timeline.png")

def create_emotion_distribution_pie(emotion_counts):
    """Create emotion distribution pie chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for each emotion
    colors = {
        'angry': '#E74C3C', 'disgust': '#8E44AD', 'fear': '#F39C12',
        'happy': '#2ECC71', 'neutral': '#95A5A6', 'sad': '#3498DB', 
        'surprise': '#E67E22', 'unknown': '#7F8C8D'
    }
    
    labels = []
    sizes = []
    pie_colors = []
    
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        labels.append(f"{emotion.capitalize()} ({count} frames)")
        sizes.append(count)
        pie_colors.append(colors.get(emotion, '#95A5A6'))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 11})
    
    # Style the text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax.set_title('Driver Emotion Distribution', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('emotion_distribution_pie.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: emotion_distribution_pie.png")

def create_emotion_bar_chart(emotion_counts):
    """Create horizontal bar chart for emotions"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    
    # Color mapping
    colors = ['#E74C3C' if e == 'angry' else
              '#8E44AD' if e == 'disgust' else
              '#F39C12' if e == 'fear' else
              '#2ECC71' if e == 'happy' else
              '#95A5A6' if e == 'neutral' else
              '#3498DB' if e == 'sad' else
              '#E67E22' for e in emotions]
    
    bars = ax.bar(emotions, counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Frames', fontsize=12, fontweight='bold')
    ax.set_title('Emotion Frequency Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(counts) * 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('emotion_bar_chart.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: emotion_bar_chart.png")

def create_driver_dashboard(emotion_counts, concentration, accuracy, fit_for_driving, duration):
    """Create comprehensive driver dashboard"""
    fig = plt.figure(figsize=(16, 10))
    
    # Color scheme
    colors = {
        'angry': '#E74C3C', 'disgust': '#8E44AD', 'fear': '#F39C12',
        'happy': '#2ECC71', 'neutral': '#95A5A6', 'sad': '#3498DB', 
        'surprise': '#E67E22'
    }
    
    # 1. Gauge meter for concentration
    ax1 = plt.subplot(2, 3, 1)
    gauge_colors = ['#FF4444', '#FF8800', '#FFFF00', '#88FF00', '#00CC00']
    theta = np.linspace(0, np.pi, 100)
    r = 1.0
    
    # Create gauge
    ax1.pie([concentration, 100-concentration], 
            colors=['#00CC00' if concentration > 60 else '#FF8800' if concentration > 40 else '#FF4444', '#EEEEEE'],
            startangle=90, counterclock=False,
            wedgeprops={'width': 0.3, 'edgecolor': 'white', 'linewidth': 2})
    
    circle = plt.Circle((0, 0), 0.7, color='white', linewidth=2, fill=True, zorder=2)
    ax1.add_artist(circle)
    ax1.text(0, 0, f'{concentration:.0f}%', ha='center', va='center', fontsize=24, fontweight='bold')
    ax1.set_title('Concentration Level', fontsize=12, fontweight='bold', pad=10)
    ax1.set_aspect('equal')
    
    # 2. Driving accuracy meter
    ax2 = plt.subplot(2, 3, 2)
    ax2.pie([accuracy, 100-accuracy],
            colors=['#3498DB' if accuracy > 50 else '#E74C3C', '#EEEEEE'],
            startangle=90, counterclock=False,
            wedgeprops={'width': 0.3, 'edgecolor': 'white', 'linewidth': 2})
    circle2 = plt.Circle((0, 0), 0.7, color='white', linewidth=2, fill=True, zorder=2)
    ax2.add_artist(circle2)
    ax2.text(0, 0, f'{accuracy:.0f}%', ha='center', va='center', fontsize=24, fontweight='bold')
    ax2.set_title('Driving Accuracy', fontsize=12, fontweight='bold', pad=10)
    ax2.set_aspect('equal')
    
    # 3. Fitness status
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    status_color = '#2ECC71' if fit_for_driving == "Yes" else '#E74C3C' if fit_for_driving == "No" else '#F39C12'
    ax3.text(0.5, 0.6, 'FITNESS STATUS', ha='center', va='center', 
             fontsize=14, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.4, fit_for_driving.upper(), ha='center', va='center', 
             fontsize=28, fontweight='bold', color=status_color, transform=ax3.transAxes)
    ax3.text(0.5, 0.2, f'Video Duration: {duration:.1f}s', ha='center', va='center',
             fontsize=10, transform=ax3.transAxes)
    
    # 4. Emotion pie chart
    ax4 = plt.subplot(2, 3, 4)
    sizes = list(emotion_counts.values())
    labels = list(emotion_counts.keys())
    pie_colors = [colors.get(e, '#95A5A6') for e in labels]
    ax4.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Emotion Distribution', fontsize=12, fontweight='bold', pad=10)
    
    # 5. Emotion percentages bar
    ax5 = plt.subplot(2, 3, 5)
    emotions = list(emotion_counts.keys())
    percentages = [count/sum(emotion_counts.values())*100 for count in emotion_counts.values()]
    bar_colors = [colors.get(e, '#95A5A6') for e in emotions]
    bars = ax5.barh(emotions, percentages, color=bar_colors, edgecolor='black')
    
    for bar, pct in zip(bars, percentages):
        ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontweight='bold')
    
    ax5.set_xlabel('Percentage (%)', fontweight='bold')
    ax5.set_title('Emotion Percentages', fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlim(0, 100)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    📊 SUMMARY STATISTICS
    ────────────────────────────
    Total Frames Analyzed: {sum(emotion_counts.values())}
    Primary Emotion: {max(emotion_counts, key=emotion_counts.get).capitalize()}
    Positive Emotion Ratio: {(emotion_counts.get('happy', 0) / sum(emotion_counts.values()) * 100):.1f}%
    Negative Emotion Ratio: {((emotion_counts.get('angry', 0) + emotion_counts.get('fear', 0) + 
                             emotion_counts.get('sad', 0) + emotion_counts.get('disgust', 0)) / 
                            sum(emotion_counts.values()) * 100):.1f}%
    
    💡 RECOMMENDATION:
    {"Driver is fit for driving. Maintain current state." if fit_for_driving == "Yes" else
     "Driver needs rest. Not recommended to drive." if fit_for_driving == "No" else
     "Caution advised. Monitor driver closely."}
    """
    
    ax6.text(0.1, 0.95, stats_text, ha='left', va='top', 
             fontsize=10, fontfamily='monospace', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('🚗 DRIVER MONITORING DASHBOARD', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('driver_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: driver_dashboard.png")

# -------------------------------
# Video Processing Function
# -------------------------------
def analyze_driver_video(video_path, emotion_model_path='models/emotion_model.h5',
                         emotion_labels_path='models/emotion_labels.json'):
    
    video_path = video_path.strip().strip('"').strip("'")
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video not found at {video_path}")
        return None
    
    print(f"\n🎥 Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video file")
        return None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video Info: {duration:.1f} seconds, {fps} FPS, {total_frames} frames")
    
    emotion_predictor = EmotionPredictor(emotion_model_path, emotion_labels_path)
    monitor = DriverMonitorWithHistory()
    
    frame_skip = max(1, fps // 10)
    processed = 0
    display_frame = None
    
    print("\n🔄 Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if processed % frame_skip == 0:
            annotated_frame, metrics = monitor.process_frame(frame, emotion_predictor, fps)
            display_frame = annotated_frame
            
            if processed % (fps * 5) == 0 and processed > 0:
                progress = (processed / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({processed}/{total_frames} frames)")
        
        processed += 1
        
        # Limit for long videos
        if processed > 20000:
            print(f"   Reached frame limit ({processed} frames)")
            break
    
    cap.release()
    
    if display_frame is not None:
        cv2.imshow('Driver Monitoring - Result', display_frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    
    print(f"\n✅ Processing complete! Analyzed {monitor.frames_with_face} faces in {processed} frames")
    
    # Get final metrics
    concentration, accuracy, primary_emotion, emotion_counts = monitor.get_final_metrics()
    face_detection_rate = (monitor.frames_with_face / monitor.total_frames * 100) if monitor.total_frames > 0 else 0
    
    # Determine fit status
    if concentration >= 60 and accuracy >= 50:
        fit = "Yes"
        reason = "Driver appears attentive and emotionally stable."
    elif concentration < 40 or accuracy < 30:
        fit = "No"
        reason = "Driver shows signs of distraction or emotional distress."
    else:
        fit = "Borderline"
        reason = "Caution advised. Driver may need a break."
    
    # Generate all graphs
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Concentration timeline
    if len(monitor.concentration_history) > 0:
        create_concentration_timeline(monitor.concentration_history, duration)
    
    # 2. Emotion distribution pie chart
    if len(emotion_counts) > 0:
        create_emotion_distribution_pie(emotion_counts)
        create_emotion_bar_chart(emotion_counts)
    
    # 3. Complete dashboard
    create_driver_dashboard(emotion_counts, concentration, accuracy, fit, duration)
    
    # Print report
    print("\n" + "="*60)
    print("🚗 DRIVER MONITORING REPORT")
    print("="*60)
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"⏱️ Duration: {duration:.1f} seconds")
    print(f"🎯 Face Detection Rate: {face_detection_rate:.1f}%")
    print("-"*60)
    print(f"🧠 Concentration Level: {concentration:.1f}%")
    print(f"🎯 Driving Accuracy: {accuracy:.1f}%")
    print(f"✅ Fit for Driving: {fit}")
    print(f"📝 Reason: {reason}")
    print("-"*60)
    print(f"😊 Primary Emotion: {primary_emotion}")
    print("\nEmotion Distribution:")
    for emo, cnt in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (cnt / sum(emotion_counts.values())) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {emo:10s}: {bar} {percentage:.1f}% ({cnt} frames)")
    print("="*60)
    
    # Save report
    result = {
        "video_file": os.path.basename(video_path),
        "video_duration_seconds": round(duration, 1),
        "total_frames_processed": processed,
        "faces_detected": monitor.frames_with_face,
        "face_detection_rate_percent": round(face_detection_rate, 1),
        "concentration_level_percent": round(concentration, 1),
        "driving_accuracy_percent": round(accuracy, 1),
        "fit_for_driving": fit,
        "reason": reason,
        "primary_emotion": primary_emotion,
        "emotion_distribution": emotion_counts,
        "total_emotion_frames": sum(emotion_counts.values()),
        "graphs_generated": [
            "concentration_timeline.png",
            "emotion_distribution_pie.png", 
            "emotion_bar_chart.png",
            "driver_dashboard.png"
        ]
    }
    
    output_file = "driver_report.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Report saved to: {output_file}")
    
    return result

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    print("="*60)
    print("🚗 DRIVER MONITORING SYSTEM WITH GRAPHS")
    print("="*60)
    
    tf.get_logger().setLevel('ERROR')
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Create default labels if not exists
    labels_path = "models/emotion_labels.json"
    if not os.path.exists(labels_path):
        with open(labels_path, 'w') as f:
            json.dump(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], f)
    
    # Find available videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    available_videos = [f for f in os.listdir('.') if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if available_videos:
        print("\n📹 Available videos:")
        for i, video in enumerate(available_videos, 1):
            size_mb = os.path.getsize(video) / (1024 * 1024)
            print(f"   {i}. {video} ({size_mb:.1f} MB)")
        
        print(f"\n   {len(available_videos)+1}. Enter custom path")
        
        choice = input(f"\nSelect video (1-{len(available_videos)+1}): ").strip()
        
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_videos):
                video_path = available_videos[idx]
            else:
                video_path = input("Enter video file path: ").strip().strip('"').strip("'")
        else:
            video_path = choice.strip().strip('"').strip("'")
    else:
        video_path = input("Enter video file path: ").strip().strip('"').strip("'")
    
    if video_path and os.path.exists(video_path):
        result = analyze_driver_video(video_path)
        
        if result:
            print("\n" + "="*60)
            print("🎉 ANALYSIS COMPLETE!")
            print("="*60)
            print("\n📁 Generated Files:")
            print("   1. driver_report.json - Complete data report")
            print("   2. concentration_timeline.png - Concentration over time graph")
            print("   3. emotion_distribution_pie.png - Emotion distribution pie chart")
            print("   4. emotion_bar_chart.png - Emotion frequency bar chart")
            print("   5. driver_dashboard.png - Complete driver dashboard")
    else:
        print(f"\n❌ File not found: {video_path}")