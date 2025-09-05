import cv2
import numpy as np
from flask import Flask, Response, request
import tflite_runtime.interpreter as tflite
from PIL import Image
import threading

# Initialize Flask
app = Flask(__name__)

# Global variables for thread-safe frame/prediction sharing
latest_frame = None
latest_prediction = "Initializing..."
lock = threading.Lock()

# TFLite Model Setup
interpreter = tflite.Interpreter(model_path='fine_tuned_efficientnet.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names (update with your labels)
class_names = [
    'Cardboard', 'E-Waste', 'General Waste',
    'Glass', 'Metals', 'Organic',
    'Paper', 'Plastics', 'Textiles'
]

def preprocess_frame(frame):
    """Prepares frame for TFLite model"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame).resize(
        (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    return np.array(frame, dtype=np.float32) / 255.0

def classify_frame(frame):
    """Runs inference on a single frame"""
    input_data = np.expand_dims(preprocess_frame(frame), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(predictions), np.max(predictions)

def generate_frames():
    """Yields video frames with prediction overlays"""
    global latest_frame, latest_prediction
    
    # Kinect setup (adjust for v1/v2)
    cap = cv2.VideoCapture(0)  # Or use freenect for Kinect v1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run classification every 5th frame to reduce load
        if threading.active_count() < 3:  # Prevent thread pileup
            threading.Thread(target=update_prediction, args=(frame.copy(),)).start()
        
        # Add prediction text to frame
        with lock:
            cv2.putText(frame, latest_prediction, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            latest_frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')

def update_prediction(frame):
    """Updates prediction in background thread"""
    global latest_prediction
    class_id, confidence = classify_frame(frame)
    with lock:
        latest_prediction = f"{class_names[class_id]} ({confidence:.2f})"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Kinect Waste Classifier</title>
      </head>
      <body>
        <h1>Live Classification</h1>
        <img src="/video_feed" width="640">
      </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
