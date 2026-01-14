from flask import Flask, render_template, request, redirect, url_for, session, Response
import sqlite3
import os
import cv2

app = Flask(__name__)
#there needs to be a secreet key to ensure sessions are secure and tamper proof
app.secret_key = "fitcompass_secret_key"

# -------------------------
# Database setup
# -------------------------
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

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
            return redirect(url_for('home'))

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
    return render_template("workoutSession.html")

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

def generate_frames():
    #mp_drawing = mp.solutions.drawing_utils
    #mp_holistic = mp.solutions.holistic
    #with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            ret, buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




#app.run is called on start
if __name__=="__main__":
    app.run(debug=True, use_reloader=False,threaded=True)
