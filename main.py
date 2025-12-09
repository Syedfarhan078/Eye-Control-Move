# backend.py - Flask Backend for Eye-Controlled Mouse
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time
import json
import os
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import base64
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'eye-mouse-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")


class EyeControlledMouse:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.screen_w, self.screen_h = pyautogui.size()
        self.smoothing_frames = 5
        self.gaze_buffer_x = deque(maxlen=self.smoothing_frames)
        self.gaze_buffer_y = deque(maxlen=self.smoothing_frames)

        self.EAR_THRESHOLD = 0.21
        self.BLINK_FRAMES = 2
        self.left_blink_counter = 0
        self.right_blink_counter = 0
        self.last_click_time = 0
        self.click_cooldown = 0.5

        self.dwell_enabled = False
        self.dwell_time = 1.5
        self.dwell_start_time = None
        self.dwell_position = None
        self.dwell_threshold = 30

        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_data = []
        self.calibration_index = 0
        self.calibration_file = "calibration_data.json"
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.sensitivity = 1.5
        self.deadzone = 5
        self.paused = False
        self.running = False
        self.click_mode = "blink"

        self.fps = 0
        self.frame_times = deque(maxlen=30)
        self.click_count = 0
        self.session_start = time.time()
        self.face_detected = False

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        self.load_calibration()

    def calculate_ear(self, eye_points, landmarks, frame_w, frame_h):
        def get_distance(p1, p2):
            x1, y1 = int(p1.x * frame_w), int(p1.y * frame_h)
            x2, y2 = int(p2.x * frame_w), int(p2.y * frame_h)
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        v1 = get_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
        v2 = get_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
        h = get_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])

        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_blink(self, landmarks, frame_w, frame_h):
        left_eye = [362, 385, 387, 263, 373, 380]
        right_eye = [33, 160, 158, 133, 153, 144]

        left_ear = self.calculate_ear(left_eye, landmarks, frame_w, frame_h)
        right_ear = self.calculate_ear(right_eye, landmarks, frame_w, frame_h)

        left_blink = left_ear < self.EAR_THRESHOLD
        right_blink = right_ear < self.EAR_THRESHOLD

        return left_blink, right_blink, left_ear, right_ear

    def smooth_gaze(self, x, y):
        self.gaze_buffer_x.append(x)
        self.gaze_buffer_y.append(y)

        smooth_x = sum(self.gaze_buffer_x) / len(self.gaze_buffer_x)
        smooth_y = sum(self.gaze_buffer_y) / len(self.gaze_buffer_y)

        return smooth_x, smooth_y

    def apply_calibration(self, x, y):
        cal_x = (x * self.scale_x + self.offset_x) * self.sensitivity
        cal_y = (y * self.scale_y + self.offset_y) * self.sensitivity
        return cal_x, cal_y

    def start_calibration(self):
        self.calibration_mode = True
        self.calibration_index = 0
        self.calibration_data = []

        margin = 100
        self.calibration_points = [
            (margin, margin),
            (self.screen_w // 2, margin),
            (self.screen_w - margin, margin),
            (margin, self.screen_h // 2),
            (self.screen_w // 2, self.screen_h // 2),
            (self.screen_w - margin, self.screen_h // 2),
            (margin, self.screen_h - margin),
            (self.screen_w // 2, self.screen_h - margin),
            (self.screen_w - margin, self.screen_h - margin),
        ]

        socketio.emit('calibration_started', {
            'total_points': len(self.calibration_points)
        })

    def process_calibration(self, gaze_x, gaze_y):
        if self.calibration_index < len(self.calibration_points):
            target_x, target_y = self.calibration_points[self.calibration_index]
            self.calibration_data.append({
                'target': (target_x, target_y),
                'gaze': (gaze_x, gaze_y)
            })

            socketio.emit('calibration_point', {
                'index': self.calibration_index,
                'total': len(self.calibration_points),
                'x': target_x,
                'y': target_y
            })

            self.calibration_index += 1
            time.sleep(0.8)

            if self.calibration_index >= len(self.calibration_points):
                self.finish_calibration()

    def finish_calibration(self):
        if len(self.calibration_data) > 0:
            gaze_points = np.array([d['gaze'] for d in self.calibration_data])
            target_points = np.array([d['target'] for d in self.calibration_data])

            gaze_x_range = gaze_points[:, 0].max() - gaze_points[:, 0].min()
            gaze_y_range = gaze_points[:, 1].max() - gaze_points[:, 1].min()

            target_x_range = target_points[:, 0].max() - target_points[:, 0].min()
            target_y_range = target_points[:, 1].max() - target_points[:, 1].min()

            self.scale_x = target_x_range / gaze_x_range if gaze_x_range > 0 else 1.0
            self.scale_y = target_y_range / gaze_y_range if gaze_y_range > 0 else 1.0

            self.offset_x = target_points[:, 0].mean() - gaze_points[:, 0].mean() * self.scale_x
            self.offset_y = target_points[:, 1].mean() - gaze_points[:, 1].mean() * self.scale_y

            self.save_calibration()

        self.calibration_mode = False
        socketio.emit('calibration_completed', {
            'success': True,
            'message': 'Calibration completed successfully!'
        })

    def save_calibration(self):
        data = {
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f)

    def load_calibration(self):
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                    self.scale_x = data['scale_x']
                    self.scale_y = data['scale_y']
                    self.offset_x = data['offset_x']
                    self.offset_y = data['offset_y']
                return True
            except:
                return False
        return False

    def get_stats(self):
        return {
            'fps': round(self.fps, 1),
            'clicks': self.click_count,
            'session_time': int(time.time() - self.session_start),
            'sensitivity': round(self.sensitivity, 1),
            'click_mode': self.click_mode,
            'paused': self.paused,
            'face_detected': self.face_detected,
            'calibrated': os.path.exists(self.calibration_file)
        }

    def process_frame(self):
        frame_start = time.time()

        ret, frame = self.cam.read()
        if not ret:
            return None, None

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.face_mesh.process(rgb_frame)

        frame_h, frame_w, _ = frame.shape

        if output.multi_face_landmarks and not self.paused:
            self.face_detected = True
            landmarks = output.multi_face_landmarks[0].landmark

            # Draw eye landmarks
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                if id == 1:
                    raw_x = self.screen_w * landmark.x
                    raw_y = self.screen_h * landmark.y

                    smooth_x, smooth_y = self.smooth_gaze(raw_x, raw_y)
                    cal_x, cal_y = self.apply_calibration(smooth_x, smooth_y)

                    screen_x = max(0, min(self.screen_w - 1, cal_x))
                    screen_y = max(0, min(self.screen_h - 1, cal_y))

                    if self.calibration_mode:
                        self.process_calibration(raw_x, raw_y)
                    else:
                        current_x, current_y = pyautogui.position()
                        distance = np.sqrt((screen_x - current_x) ** 2 + (screen_y - current_y) ** 2)

                        if distance > self.deadzone:
                            pyautogui.moveTo(screen_x, screen_y)

                        # Dwell click
                        if self.dwell_enabled:
                            if self.dwell_position is None:
                                self.dwell_position = (screen_x, screen_y)
                                self.dwell_start_time = time.time()
                            else:
                                dwell_distance = np.sqrt(
                                    (screen_x - self.dwell_position[0]) ** 2 +
                                    (screen_y - self.dwell_position[1]) ** 2
                                )

                                if dwell_distance < self.dwell_threshold:
                                    elapsed = time.time() - self.dwell_start_time
                                    if elapsed >= self.dwell_time:
                                        pyautogui.click()
                                        self.click_count += 1
                                        self.dwell_position = None
                                        self.dwell_start_time = None
                                        socketio.emit('click_performed', {'type': 'dwell'})
                                    else:
                                        progress = int((elapsed / self.dwell_time) * 360)
                                        center_x = int(landmark.x * frame_w)
                                        center_y = int(landmark.y * frame_h)
                                        cv2.ellipse(frame, (center_x, center_y), (30, 30),
                                                    0, 0, progress, (0, 255, 255), 3)
                                else:
                                    self.dwell_position = (screen_x, screen_y)
                                    self.dwell_start_time = time.time()

            # Blink detection
            if self.click_mode == "blink":
                left_blink, right_blink, left_ear, right_ear = self.detect_blink(
                    landmarks, frame_w, frame_h
                )

                cv2.putText(frame, f"L: {left_ear:.2f}", (10, frame_h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"R: {right_ear:.2f}", (10, frame_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                current_time = time.time()

                if left_blink and not right_blink:
                    self.left_blink_counter += 1
                    if (self.left_blink_counter >= self.BLINK_FRAMES and
                            current_time - self.last_click_time > self.click_cooldown):
                        pyautogui.click()
                        self.click_count += 1
                        self.last_click_time = current_time
                        self.left_blink_counter = 0
                        socketio.emit('click_performed', {'type': 'left_blink'})
                else:
                    self.left_blink_counter = 0

                if right_blink and not left_blink:
                    self.right_blink_counter += 1
                    if (self.right_blink_counter >= self.BLINK_FRAMES and
                            current_time - self.last_click_time > self.click_cooldown):
                        pyautogui.rightClick()
                        self.click_count += 1
                        self.last_click_time = current_time
                        self.right_blink_counter = 0
                        socketio.emit('click_performed', {'type': 'right_blink'})
                else:
                    self.right_blink_counter = 0
        else:
            self.face_detected = False

        # Status overlay
        status_color = (0, 255, 0) if not self.paused else (0, 0, 255)
        status_text = "ACTIVE" if not self.paused else "PAUSED"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        if self.calibration_mode:
            cv2.putText(frame, "CALIBRATING...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Calculate FPS
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 0:
            self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))

        return frame, self.get_stats()


# Global controller instance
controller = EyeControlledMouse()


def generate_frames():
    while controller.running:
        frame, stats = controller.process_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if stats:
                socketio.emit('stats_update', stats)

        time.sleep(0.01)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('stats_update', controller.get_stats())


@socketio.on('start_tracking')
def handle_start():
    controller.running = True
    controller.paused = False
    threading.Thread(target=generate_frames, daemon=True).start()
    emit('tracking_started', {'success': True})


@socketio.on('pause_tracking')
def handle_pause():
    controller.paused = not controller.paused
    emit('tracking_paused', {'paused': controller.paused})


@socketio.on('stop_tracking')
def handle_stop():
    controller.running = False
    emit('tracking_stopped', {'success': True})


@socketio.on('start_calibration')
def handle_calibration():
    controller.start_calibration()


@socketio.on('change_click_mode')
def handle_click_mode(data):
    controller.click_mode = data['mode']
    controller.dwell_enabled = (controller.click_mode == 'dwell')
    emit('click_mode_changed', {'mode': controller.click_mode})


@socketio.on('change_sensitivity')
def handle_sensitivity(data):
    controller.sensitivity = float(data['value'])
    emit('sensitivity_changed', {'value': controller.sensitivity})


@socketio.on('change_ear_threshold')
def handle_ear_threshold(data):
    controller.EAR_THRESHOLD = float(data['value'])
    emit('ear_threshold_changed', {'value': controller.EAR_THRESHOLD})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)