"""
AI Exam Monitor engine.

This module processes webcam frames supplied by the Streamlit frontend.
It does not open its own OpenCV webcam window.
"""

import csv
import os
import threading
import time
import uuid
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(BASE_DIR, "exam_events.csv")
SESSION_EVENTS_FILE = os.path.join(BASE_DIR, "exam_session_events.csv")
SESSION_SUMMARY_FILE = os.path.join(BASE_DIR, "exam_session_summary.csv")

PHONE_CLASS = 67
PERSON_CLASS = 0
BOOK_CLASS = 73

YOLO_CONFIDENCE = 0.40
PERSON_CONFIDENCE = 0.60
MIN_PERSON_AREA = 0.02

CALIBRATION_SECONDS = 5
TURN_THRESHOLD = 30
SUSTAIN_SECONDS = 2.0
MULTIPLE_PEOPLE_SECONDS = 1.0
EVENT_COOLDOWN = 3.0
SMOOTHING_FRAMES = 5

EVENT_TYPES = [
    "PHONE DETECTED",
    "BOOK DETECTED",
    "SUSTAINED HEAD TURN - REVIEW",
    "MULTIPLE PEOPLE DETECTED",
]

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

FACE_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)


def ensure_csv_headers():
    """Create CSV files and headers if they do not already exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "event", "details"])

    if not os.path.exists(SESSION_EVENTS_FILE):
        with open(SESSION_EVENTS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "session_id", "event", "start_time", "end_time",
                "duration_seconds"
            ])

    if not os.path.exists(SESSION_SUMMARY_FILE):
        with open(SESSION_SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "session_id", "session_start", "session_end",
                "session_duration_seconds", "phone_events",
                "phone_total_seconds", "book_events", "book_total_seconds",
                "head_turn_events", "head_turn_total_seconds",
                "multiple_people_events", "multiple_people_total_seconds"
            ])


def estimate_head_pose(landmarks, width, height):
    points_2d = np.array([
        (landmarks[i].x * width, landmarks[i].y * height)
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = width
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    distortion = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        FACE_3D, points_2d, camera_matrix, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    return float(angles[1]), float(angles[0])


class SessionLogger:
    def __init__(self):
        ensure_csv_headers()
        self.session_id = str(uuid.uuid4())[:8]
        self.started_at = datetime.now()
        self.start_clock = time.monotonic()
        self.active_events = {}
        self.event_stats = {
            event: {"count": 0, "total_seconds": 0.0}
            for event in EVENT_TYPES
        }
        self.closed = False

    def update(self, current_events):
        if self.closed:
            return
        now_clock = time.monotonic()
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for event, is_active in current_events.items():
            if is_active and event not in self.active_events:
                self.active_events[event] = {
                    "start_clock": now_clock,
                    "start_time": now_text,
                }
            elif not is_active and event in self.active_events:
                self._finish_event(event, now_clock, now_text)

    def _finish_event(self, event, end_clock, end_text):
        details = self.active_events.pop(event, None)
        if details is None:
            return

        duration = max(0.0, end_clock - details["start_clock"])
        with open(SESSION_EVENTS_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                self.session_id, event, details["start_time"],
                end_text, round(duration, 2)
            ])

        self.event_stats[event]["count"] += 1
        self.event_stats[event]["total_seconds"] += duration

    def close(self):
        if self.closed:
            return
        end_clock = time.monotonic()
        end_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for event in list(self.active_events):
            self._finish_event(event, end_clock, end_text)

        duration = end_clock - self.start_clock
        s = self.event_stats
        row = [
            self.session_id,
            self.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            end_text,
            round(duration, 2),
            s["PHONE DETECTED"]["count"],
            round(s["PHONE DETECTED"]["total_seconds"], 2),
            s["BOOK DETECTED"]["count"],
            round(s["BOOK DETECTED"]["total_seconds"], 2),
            s["SUSTAINED HEAD TURN - REVIEW"]["count"],
            round(s["SUSTAINED HEAD TURN - REVIEW"]["total_seconds"], 2),
            s["MULTIPLE PEOPLE DETECTED"]["count"],
            round(s["MULTIPLE PEOPLE DETECTED"]["total_seconds"], 2),
        ]
        with open(SESSION_SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        self.closed = True

    def summary_dict(self):
        duration = time.monotonic() - self.start_clock
        result = {
            "session_id": self.session_id,
            "session_start": self.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration_seconds": round(duration, 1),
        }
        for event, values in self.event_stats.items():
            result[event] = values["count"]
            result[event + " seconds"] = round(values["total_seconds"], 1)
        return result


class ExamMonitorProcessor:
    """
    streamlit-webrtc video processor.
    A lock protects model/landmark processing and stopping the session.
    """

    def __init__(self):
        ensure_csv_headers()
        self.lock = threading.RLock()
        self.model = YOLO("yolov8n.pt")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.logger = SessionLogger()
        self.last_alert_time = {}
        self.calibration_started = None
        self.calibration_samples = []
        self.baseline = None
        self.turn_started = None
        self.multiple_people_started = None
        self.yaw_history = []
        self.pitch_history = []
        self.last_status = "Starting"
        self.last_metrics = {}
        self.stopped = False

    def _report_alert(self, message, details=""):
        now = time.monotonic()
        if now - self.last_alert_time.get(message, 0) < EVENT_COOLDOWN:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timestamp, message, details])
        self.last_alert_time[message] = now

    def recv(self, frame):
        # Required by streamlit-webrtc; return a video frame.
        image = frame.to_ndarray(format="bgr24")
        processed = self.process_image(image)
        from av import VideoFrame
        return VideoFrame.from_ndarray(processed, format="bgr24")

    def process_image(self, image):
        with self.lock:
            if self.stopped:
                return image

            now = time.monotonic()
            height, width = image.shape[:2]

            result = self.model(
                image,
                classes=[PERSON_CLASS, PHONE_CLASS, BOOK_CLASS],
                conf=YOLO_CONFIDENCE,
                iou=0.50,
                verbose=False
            )[0]

            display = result.plot()
            person_count = 0
            phone_detected = False
            book_detected = False

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == PERSON_CLASS:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        if (
                            confidence >= PERSON_CONFIDENCE
                            and area / (width * height) >= MIN_PERSON_AREA
                        ):
                            person_count += 1

                    elif class_id == PHONE_CLASS:
                        phone_detected = True
                        self._report_alert(
                            "PHONE DETECTED", f"confidence={confidence:.2f}"
                        )

                    elif class_id == BOOK_CLASS:
                        book_detected = True
                        self._report_alert(
                            "BOOK DETECTED", f"confidence={confidence:.2f}"
                        )

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb)
            face_count = 0
            pose = None

            if face_results.multi_face_landmarks:
                face_count = len(face_results.multi_face_landmarks)
                pose = estimate_head_pose(
                    face_results.multi_face_landmarks[0].landmark,
                    width, height
                )

            head_status = "Face not detected"
            yaw_delta = 0.0
            pitch_delta = 0.0

            if pose is not None:
                yaw, pitch = pose
                self.yaw_history.append(yaw)
                self.pitch_history.append(pitch)

                if len(self.yaw_history) > SMOOTHING_FRAMES:
                    self.yaw_history.pop(0)
                    self.pitch_history.pop(0)

                smooth_yaw = float(np.median(self.yaw_history))
                smooth_pitch = float(np.median(self.pitch_history))

                if self.baseline is None:
                    head_status = "Calibrating: look at screen"
                    if self.calibration_started is None:
                        self.calibration_started = now
                    self.calibration_samples.append((smooth_yaw, smooth_pitch))

                    if now - self.calibration_started >= CALIBRATION_SECONDS:
                        if len(self.calibration_samples) >= 15:
                            self.baseline = np.median(
                                np.array(self.calibration_samples), axis=0
                            )
                            head_status = "Head position normal"
                        else:
                            self.calibration_started = now
                            self.calibration_samples.clear()
                else:
                    yaw_delta = smooth_yaw - self.baseline[0]
                    pitch_delta = smooth_pitch - self.baseline[1]
                    turned = (
                        abs(yaw_delta) > TURN_THRESHOLD
                        or abs(pitch_delta) > TURN_THRESHOLD
                    )

                    if turned:
                        if self.turn_started is None:
                            self.turn_started = now
                        if now - self.turn_started >= SUSTAIN_SECONDS:
                            head_status = "HEAD TURN: review"
                            self._report_alert(
                                "SUSTAINED HEAD TURN - REVIEW",
                                f"yaw_delta={yaw_delta:.1f}, "
                                f"pitch_delta={pitch_delta:.1f}"
                            )
                        else:
                            head_status = "Checking head movement"
                    else:
                        self.turn_started = None
                        head_status = "Head position normal"
            else:
                self.turn_started = None
                self.yaw_history.clear()
                self.pitch_history.clear()

            multiple_people = person_count > 1 or face_count > 1
            if multiple_people:
                if self.multiple_people_started is None:
                    self.multiple_people_started = now
            else:
                self.multiple_people_started = None

            multiple_people_alert = (
                self.multiple_people_started is not None
                and now - self.multiple_people_started >= MULTIPLE_PEOPLE_SECONDS
            )

            if multiple_people_alert:
                self._report_alert(
                    "MULTIPLE PEOPLE DETECTED",
                    f"people={person_count}, faces={face_count}"
                )

            self.logger.update({
                "PHONE DETECTED": phone_detected,
                "BOOK DETECTED": book_detected,
                "SUSTAINED HEAD TURN - REVIEW": (
                    head_status == "HEAD TURN: review"
                ),
                "MULTIPLE PEOPLE DETECTED": multiple_people_alert,
            })

            if phone_detected:
                status, color = "PHONE DETECTED - REVIEW", (0, 0, 255)
            elif multiple_people_alert:
                status, color = "MULTIPLE PEOPLE - REVIEW", (0, 165, 255)
            elif head_status == "HEAD TURN: review":
                status, color = head_status, (0, 165, 255)
            else:
                status, color = "MONITORING", (0, 255, 0)

            cv2.putText(
                display, f"People: {person_count} | Faces: {face_count}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2
            )
            cv2.putText(
                display, f"Head: {head_status}",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2
            )
            cv2.putText(
                display, f"Book detected: {'Yes' if book_detected else 'No'}",
                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            cv2.putText(
                display, status, (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2
            )

            self.last_status = status
            self.last_metrics = {
                "People": person_count,
                "Faces": face_count,
                "Phone detected": phone_detected,
                "Book detected": book_detected,
                "Head status": head_status,
            }
            return display

    def get_live_summary(self):
        with self.lock:
            return self.logger.summary_dict()

    def stop(self):
        with self.lock:
            if self.stopped:
                return
            self.logger.close()
            self.face_mesh.close()
            self.stopped = True
