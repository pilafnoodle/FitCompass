from flask import Flask, render_template, request, redirect, url_for, session, Response
import sqlite3
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
from flask import jsonify
import time

LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_HEEL = 29
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_HEEL = 30

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

app = Flask(__name__)
#there needs to be a secreet key to ensure sessions are secure and tamper proof
app.secret_key = "fitcompass_secret_key"

currentDirectory = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(currentDirectory, "UserLogins.db")

def get_db_connection():
    return sqlite3.connect(db_path)

connection = get_db_connection()
cursor = connection.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS UserLogins(
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
connection.commit()
connection.close()

# -------------------------
# Webcam setup
# -------------------------
camera = cv2.VideoCapture(0)

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def angleBetweenLines(a,b,c):
    a = np.array(a) # end
    b = np.array(b) # vertext 
    c = np.array(c) # End 
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class SquatState:
    IDLE="IDLE"
    BEGIN = "BEGIN"
    DOWN = "DOWN"
    RISE="RISE"

class SquatStateMachine:
    def __init__(self):
        self.state = SquatState.IDLE
        self.count = 0
        self.knee_angle = 0
        self.heel_anchor = None
        self.down_start_time = None

    def update(self, detection_result, image_shape):
        if not detection_result or not detection_result.pose_landmarks:
            return
        landmarks = detection_result.pose_landmarks[0] #only get the first person
        pixel_landmarks = landmarks_to_pixels(landmarks, image_shape)  
        left_hip = pixel_landmarks[LEFT_HIP] #both share a point at knee
        left_knee = pixel_landmarks[LEFT_KNEE]
        left_heel = pixel_landmarks[LEFT_HEEL]

        self.left_knee_angle=angleBetweenLines(left_hip,left_knee,left_heel)

        right_hip = pixel_landmarks[RIGHT_HIP] #both share a point at knee
        right_knee = pixel_landmarks[RIGHT_KNEE]
        right_heel = pixel_landmarks[RIGHT_HEEL]

        self.right_knee_angle=angleBetweenLines(right_hip,right_knee,right_heel)
              
        if self.state == SquatState.IDLE:
            if self.left_knee_angle>140 and self.right_knee_angle >140:
                self.heel_anchor = np.array(left_heel)

            if self.left_knee_angle<120 and self.right_knee_angle <120:
                self.state=SquatState.BEGIN
                return
            
        elif self.state==SquatState.BEGIN:
            if self.heel_anchor is None:

                self.state = SquatState.IDLE
                return
            current_heel = np.array(left_heel)
            heel_displacement = np.linalg.norm(current_heel - self.heel_anchor)
            if heel_displacement > 80:
                self.state=SquatState.IDLE
                return

            if self.left_knee_angle<80 and self.right_knee_angle <80 : #80 degree squat
                self.state = SquatState.DOWN
                self.down_start_time = time.time()
                return

        elif self.state == SquatState.DOWN:
            if self.left_knee_angle > 100 or self.right_knee_angle> 100: # User started rising too early
                if (time.time() - self.down_start_time) >= 1.0:
                    self.state =  SquatState.RISE
                else:
                    self.state = SquatState.RISE

        elif self.state ==  SquatState.RISE:
            if self.left_knee_angle <160  and self.right_knee_angle<160 :
                self.count += 1
                self.state = SquatState.IDLE
                print(f"Count: {self.count}")

squatController = SquatStateMachine()

def draw_squat_lines(image, detection_result):
    annotated_image = image.copy()
    if not detection_result.pose_landmarks:
        return annotated_image
    h, w, _ = image.shape

    for pose_landmarks in detection_result.pose_landmarks:
        def to_pixel(lm):
            return int(lm.x * w), int(lm.y * h)
        left_hip = to_pixel(pose_landmarks[LEFT_HIP])
        left_knee = to_pixel(pose_landmarks[LEFT_KNEE])
        left_ankle = to_pixel(pose_landmarks[LEFT_HEEL])
        right_hip = to_pixel(pose_landmarks[RIGHT_HIP])
        right_knee = to_pixel(pose_landmarks[RIGHT_KNEE])
        right_ankle = to_pixel(pose_landmarks[RIGHT_HEEL])
        cv2.line(annotated_image, left_hip, left_knee, (0, 255, 0), 2)
        cv2.line(annotated_image, left_knee, left_ankle, (0, 255, 0), 2)
        cv2.line(annotated_image, right_hip, right_knee, (0, 255, 0), 2)
        cv2.line(annotated_image, right_knee, right_ankle, (0, 255, 0), 2)
    return annotated_image

def landmarks_to_pixels(pose_landmarks, image_shape):
    h, w, _ = image_shape

    pixel_landmarks = []

    for lm in pose_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pixel_landmarks.append((x, y))

    return pixel_landmarks

@app.route('/get_squat_data')
def get_squat_data():
    # Return the count and the state from your squatController
    return jsonify(
        count=squatController.count,
        state=squatController.state,
        angle=round(squatController.left_knee_angle, 1)
    )

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        squatController.update(detection_result, frame.shape)

        annotated_image = draw_squat_lines(frame, detection_result)
        ret, buffer = cv2.imencode('.jpg', annotated_image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# -------------------------
# Login
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            "SELECT * FROM UserLogins WHERE username=? AND password=?",
            (username, password)
        )
        user = cursor.fetchone()
        connection.close()

        if user:
            session['username'] = username
            return redirect(url_for('workoutSession'))

        else:
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

# -------------------------
# Register
# -------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO UserLogins(username, password) VALUES (?, ?)",
            (username, password)
        )
        connection.commit()
        connection.close()

        return redirect(url_for('login'))

    return render_template('register.html')

# -------------------------
# Home (personalized)
# -------------------------
@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']

    # Placeholder personalized data (replace later with DB queries)
    points = 120
    goal_percent = 62

    return render_template(
        'homePage.html',
        username=username,
        points=points,
        goal_percent=goal_percent
    )


@app.route('/workoutSession')
def workoutSession():
    squat_count=0
    knee_angle=0

    return render_template("workoutSession.html",squat_count=squat_count,knee_angle=knee_angle)

# -------------------------
# Placeholder routes
# -------------------------
@app.route('/profile')
def profile():
    return "Profile page coming soon"

@app.route('/history')
def history():
    return "History page coming soon"

@app.route('/shop')
def shop():
    return "Shop page coming soon"

@app.route('/library')
def library():
    return "Library page coming soon"

@app.route('/settings')
def settings():
    return "Settings page coming soon"

@app.route('/more')
def more():
    return "More page coming soon"

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
