import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Initialize interpreter
interpreter = tflite.Interpreter(model_path="fine_tuned_efficientnet.tflite")
interpreter.allocate_tensors()

# Get I/O details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class mapping
class_indices = {
    0: "Cardboard",
    1: "Electronic Waste",
    2: "General Waste",
    3: "Glass",
    4: "Metals",
    5: "Organic Waste",
    6: "Paper",
    7: "Plastics",
    8: "Textiles"
}

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]    # ImageNet std
    img = (img / 255.0 - mean) / std
    return np.expand_dims(img, axis=0).astype(np.float32)

# Main execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    args = parser.parse_args()
    
    # Run inference
    input_tensor = preprocess_image(args.image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Process results
    probabilities = np.squeeze(output)
    predicted_class_idx = np.argmax(probabilities)
    
    print("\n--- Waste Classification Results ---")
    print(f"Predicted: {class_indices[predicted_class_idx]} ({probabilities[predicted_class_idx]:.2%} confidence)")
    print("\nDetailed probabilities:")
    for class_idx, prob in enumerate(probabilities):
        print(f"  {class_indices[class_idx]:<15}: {prob:.2%}")
