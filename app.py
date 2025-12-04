from flask import Flask, render_template, Response
import cv2
app  = Flask(__name__)

camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            ret, buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

#this says whenever a user visits our website domain, run index.html
@app.route('/')
def index():
    return render_template('index.html')

#this returns streaming response
@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#app.run is called on start
if __name__=="__main__":
    app.run(debug=True)
