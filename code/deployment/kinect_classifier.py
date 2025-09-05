import cv2
import numpy as np
import time
from collections import Counter
from PIL import Image
import tflite_runtime.interpreter as tflite

class KinectClassifier:
    def __init__(self, model_path, window_size=2, fps=5, aggregation='majority'):
        """
        Initialize the classifier for Kinect on Raspberry Pi.

        Args:
            model_path: Path to the TFLite model
            window_size: Time window in seconds
            fps: Frames per second to capture (lower for RPi)
            aggregation: 'majority' voting
        """
        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.window_size = window_size
        self.fps = fps
        self.aggregation = aggregation
        self.frames_per_window = window_size * fps
        self.frame_buffer = []

        # Update with your actual class names
        self.class_names = [
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

    def preprocess_frame(self, frame):
        """Preprocess a frame for the model"""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame = Image.fromarray(frame)
        frame = frame.resize((self.input_details[0]['shape'][2],
                             self.input_details[0]['shape'][1]))
        frame = np.array(frame, dtype=np.float32)
        # Normalize if your model expects it
        frame = frame / 255.0
        return np.expand_dims(frame, axis=0)

    def classify_frame(self, frame):
        """Classify a single frame using TFLite"""
        processed_frame = self.preprocess_frame(frame)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_frame)

        # Run inference
        self.interpreter.invoke()

        # Get output
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        return predicted_class, confidence, predictions

    def aggregate_predictions(self, predictions):
        """Aggregate predictions using majority voting"""
        if not predictions:
            return None
        class_counts = Counter(predictions)
        majority_class = class_counts.most_common(1)[0][0]
        return majority_class

    def process_kinect_stream(self):
        """Process video stream from Kinect"""
        # Initialize Kinect
        # For Kinect v1:
        try:
            import freenect
            ctx = freenect.init()
            dev = freenect.open_device(ctx, 0)
            print("Kinect v1 initialized")

            def get_video():
                return freenect.sync_get_video()[0]

        except:
            # Fallback to OpenCV if freenect not available
            print("Freenect not available, trying OpenCV")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open video stream")
                return

        print(f"Starting processing with {self.fps} FPS and {self.window_size}s windows...")

        try:
            while True:
                start_time = time.time()
                self.frame_buffer = []

                # Collect frames for the current window
                for _ in range(self.frames_per_window):
                    try:
                        if 'freenect' in locals():
                            frame = get_video()
                        else:
                            ret, frame = cap.read()
                            if not ret:
                                print("Error: Could not read frame")
                                break
                    except Exception as e:
                        print(f"Error capturing frame: {e}")
                        continue

                    # Classify frame
                    try:
                        class_id, confidence, _ = self.classify_frame(frame)
                        self.frame_buffer.append(class_id)

                        # Print current prediction (since we don't have display)
                        print(f"Current: {self.class_names[class_id]} ({confidence:.2f})")

                    except Exception as e:
                        print(f"Error classifying frame: {e}")
                        continue

                    # Control frame rate
                    elapsed = time.time() - start_time
                    time.sleep(max(0, (1/self.fps) - elapsed))

                # Aggregate predictions after each window
                if self.frame_buffer:
                    final_prediction = self.aggregate_predictions(self.frame_buffer)
                    print(f"\\nWindow prediction: {self.class_names[final_prediction]}\\n")

        except KeyboardInterrupt:
            print("Stopping processing...")
        finally:
            if 'cap' in locals():
                cap.release()

if __name__ == "__main__":
    # Initialize classifier
    classifier = KinectClassifier(
        model_path='fine_tuned_efficientnet.tflite',
        window_size=3,  # Longer window for RPi
        fps=3,         # Lower FPS for RPi
        aggregation='majority'
    )

    # Start processing
    classifier.process_kinect_stream()
