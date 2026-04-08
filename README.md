# 🚗 Driver Monitoring System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-red.svg)](https://opencv.org/)

## Overview

The Driver Monitoring System is a Flask-based web application that evaluates driver behavior using computer vision and deep learning. It analyzes uploaded driving videos to detect facial expressions, measure concentration, and generate a fit-for-driving assessment backed by visual dashboards and reports.

## Key Features

- **Facial Emotion Recognition** — Detects 7 core emotions: Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral.
- **Concentration Scoring** — Estimates driver attention and focus over the video.
- **Driving Accuracy Evaluation** — Computes a safety score based on driver behavior.
- **Fit-for-Driving Decision** — Provides a safety recommendation from the analysis.
- **Video Upload & Analysis** — Support for uploaded driving video files.
- **User Authentication** — Login and registration flow with SQLite-backed user data.
- **Results Dashboard** — Displays analysis history and report downloads.
- **Report Generation** — Saves JSON reports and visual graphs automatically.

## Demo

1. Run the application locally.
2. Open the browser at `http://127.0.0.1:5000`.
3. Register or login.
4. Upload a driving video and view the generated safety report.

## Requirements

- Python 3.8+
- Flask
- Flask-Login
- Flask-SQLAlchemy
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Pillow

## Installation

```bash
cd Driver-monitoring-master/Driver-monitoring-master
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Application

```bash
cd Driver-monitoring-master/Driver-monitoring-master
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Project Structure

- `app.py` — Flask application entry point
- `driver_monitoring.py` — video and driver analysis logic
- `requirements.txt` — project dependencies
- `.gitignore` — files and directories excluded from Git
- `templates/` — HTML templates for UI pages
- `static/` — static assets and uploaded files
- `generated_results/` — saved analysis reports
- `models/` — model and label files

## Notes

- The application automatically creates `generated_results/`, `static/uploads/`, and `users.db` when it first runs.
- If `dlib` is unavailable, the app uses a simplified detection mode.
- Use Python 3.8–3.11 for best compatibility with the listed dependencies.

## GitHub Setup

To publish this project on GitHub:

```bash
git init
git add .
git commit -m "chore: initial project commit"
```

Then create a repository on GitHub and connect it:

```bash
git remote add origin https://github.com/<your-username>/<repository-name>.git
git branch -M main
git push -u origin main
```

## License

Add a license file such as `LICENSE` before publishing the repository to GitHub.
