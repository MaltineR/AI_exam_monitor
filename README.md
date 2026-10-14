# AI Exam Monitor

An AI-assisted exam monitoring application built with **Python, YOLOv8, MediaPipe, OpenCV, and Streamlit**. The application analyzes webcam video to identify visual events that may require review during an online exam and records monitoring results locally in CSV files.

> **Note:** This is a prototype and decision-support tool. Alerts indicate events for human review; they are not proof of cheating.

## 1. Project Purpose

Online exams can be challenging to supervise consistently. This project explores how computer vision can assist with online exam monitoring by detecting selected objects, estimating head orientation, and flagging certain events.

The application combines real-time video processing with a web dashboard, allowing users to monitor a session and review recorded events afterward.

The main goals were to:

- Apply computer vision to a practical real-world problem.
- Detect selected objects using an AI object-detection model.
- Estimate head orientation using facial landmarks.
- Identify potentially unusual events according to configured rules.
- Display monitoring information in a web interface.
- Store event logs and session summaries for later analysis.

## 2. How It Works

The application follows this workflow:

1. **Webcam input:** The browser requests permission to access the camera.
2. **Video processing:** Webcam frames are transferred to the Python processing engine.
3. **Object detection:** YOLOv8 detects supported objects, including people, cell phones, and books.
4. **Face analysis:** MediaPipe Face Mesh identifies facial landmarks.
5. **Head-pose estimation:** OpenCV uses facial landmarks and `solvePnP` to estimate approximate head orientation.
6. **Event detection:** Configured rules identify events such as sustained head turns or multiple detected people.
7. **Dashboard display:** Streamlit presents monitoring controls, status information, and recorded results.
8. **Data storage:** Events and session summaries are saved locally in CSV files.

A detected object or head movement does not establish a person's intent or prove that an exam rule was violated.

## 3. Main Features

- Browser-based Streamlit dashboard.
- Live webcam feed using `streamlit-webrtc`.
- YOLOv8 object detection.
- MediaPipe facial landmark detection.
- OpenCV-based head-pose estimation.
- Configured event detection for sustained head turns and multiple people.
- Timestamped event logging.
- Session identifiers and event-duration tracking.
- CSV-based session summaries.
- Dashboard views for session history, event durations, and alerts.

Detection quality depends on camera position, lighting, video quality, and the configured thresholds.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Web interface and dashboard |
| streamlit-webrtc | Webcam video processing in the browser |
| YOLOv8 / Ultralytics | Object detection |
| MediaPipe Face Mesh | Facial landmark detection |
| OpenCV | Video processing and head-pose estimation |
| NumPy | Numerical operations |
| Pandas | Reading and displaying CSV results |
| CSV | Local event and session storage |

## 5. Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/MaltineR/AI_exam_monitor.git
cd YOUR-REPOSITORY
```

Alternatively, download and extract the repository ZIP from GitHub.

### Step 2: Create a virtual environment

**Windows — Command Prompt:**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**Windows — PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS or Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If you already use Conda, you can activate your existing environment instead.

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 8. Running the Application

From the project directory, with your Python environment activated, run:

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser and allow camera access when prompted.

To stop the application, return to the terminal and press `Ctrl + C`.

## 9. Using the Dashboard

1. Start the application.
2. Open the local Streamlit URL.
3. Allow access to your webcam.
4. Start the monitoring session using the dashboard controls.
5. Position the camera so the face and relevant surroundings are visible.
6. Review the displayed status and any generated alerts.
7. Stop monitoring when the session is complete.
8. Review the available session history, event durations, and alert log.

Camera access and real-time video behavior may vary by browser and operating system.

## 6. Limitations and Responsible Use

This project is an experimental prototype, not a certified exam-proctoring system.

- **False positives:** Ordinary movements may trigger alerts.
- **False negatives:** Events may be missed because of poor lighting, occlusion, camera angles, or model limitations.
- **Approximate head pose:** Head orientation does not reliably establish attention or intent.
- **Object detection limits:** Detecting an object does not establish how it is being used.
- **Privacy:** Camera use and saved records should be handled with appropriate consent, access controls, retention policies, and compliance with applicable law.
- **Human review:** Alerts should be evaluated in context and must not be treated as definitive proof of misconduct.

## 7. Future Improvements

Potential future enhancements include:

- Configurable detection thresholds.
- Improved event timelines and report exports.
- Automated tests for event detection and session logging.
- Evaluation using representative, consented test data.
- Better error handling and CSV validation.
- Privacy and data-retention controls.
- Improved robustness across different cameras and lighting conditions.

