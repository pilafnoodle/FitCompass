from flask import Flask, render_template, Response, request, redirect, url_for
import sqlite3
import os
import cv2
app  = Flask(__name__)

camera=cv2.VideoCapture(0)
currentDirectory=os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(currentDirectory, "UserLogins.db")

connection = sqlite3.connect(db_path)
cursor = connection.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS UserLogins(username text,password text)")
connection.commit()
connection.close()
conn = sqlite3.connect(db_path)
cursor = conn.cursor()


@app.route('/',methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']

        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        cursor.execute(
            "SELECT * FROM UserLogins WHERE username = ? AND password = ?",
            (username, password)
        )
        user = cursor.fetchone()
        connection.close()

        print("Fetched user:", user)
        if user:
            return render_template("workoutSession.html")
        else:
            return "Invalid login"
    else:
        request.method='GET'
        return render_template('login.html')
    
@app.route('/register',methods=['POST','GET'])
def register():

    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']

        connection=sqlite3.connect(db_path)
        cursor=connection.cursor()

        cursor.execute("INSERT INTO UserLogins (username,password) VALUES (?,?)",(username,password))   
        connection.commit()
        connection.close()
        return redirect(url_for('login'))


    return render_template('register.html')


def generate_frames():
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            ret, buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

#this returns streaming response
@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#app.run is called on start
if __name__=="__main__":
    app.run(debug=True)
