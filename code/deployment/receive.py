# pi_receiver_tflite.py
import cv2
import numpy as np
import time
from collections import Counter
from PIL import Image
import imagezmq
from tflite_runtime.interpreter import Interpreter

class VideoClassifierPiTFLite:
    def __init__(self, model_path, class_names, window_size=2, fps=10):
        # Load TFLite model
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Class info
        self.class_names = class_names
        self.window_size = window_size
        self.fps = fps
        self.frames_per_window = window_size * fps
        self.frame_buffer = []

        # Input shape
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

    def preprocess_frame(self, frame):
        """Resize and normalize for TFLite input"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame).resize((self.input_width, self.input_height))
        frame = np.array(frame, dtype=np.float32) / 255.0
        return np.expand_dims(frame, axis=0).astype(np.float32)

    def classify_frame(self, frame):
        """Run TFLite inference"""
        input_data = self.preprocess_frame(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        class_id = int(np.argmax(output_data))
        confidence = float(np.max(output_data))
        return class_id, confidence

    def aggregate_predictions(self, predictions):
        """Majority voting"""
        if not predictions:
            return None
        return Counter(predictions).most_common(1)[0][0]

    def process_stream(self):
        """Receive and classify video stream from PC"""
        receiver = imagezmq.ImageHub(open_port='tcp://*:5555')
        print("[INFO] Waiting for incoming frames...")

        try:
            while True:
                self.frame_buffer = []
                start_time = time.time()

                for _ in range(self.frames_per_window):
                    cam_id, frame = receiver.recv_image()
                    receiver.send_reply(b'OK')

                    class_id, confidence = self.classify_frame(frame)
                    self.frame_buffer.append(class_id)

                    # Display with prediction
                    display = cv2.resize(frame, (640, 480))
                    label = f"{self.class_names[class_id]} ({confidence:.2f})"
                    cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Pi Classifier", display)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return

                    elapsed = time.time() - start_time
                    wait = max(0, (1/self.fps - elapsed))
                    time.sleep(wait)

                final_prediction = self.aggregate_predictions(self.frame_buffer)
                print(f"[RESULT] Window Prediction: {self.class_names[final_prediction]}")

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    class_names = [
        'Cardboard',
        'Electronic Waste',
        'General Waste',
        'Glass',
        'Metals',
        'Organic Waste',
        'Paper',
        'Plastics',
        'Textiles'
    ]
    
    model_path = 'fine_tuned_efficientnet.tflite'  # Update to your model's filename

    classifier = VideoClassifierPiTFLite(
        model_path=model_path,
        class_names=class_names,
        window_size=2,
        fps=10
    )
    classifier.process_stream()
