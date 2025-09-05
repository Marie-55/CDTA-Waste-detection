import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
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
    5: "Organic Waste"
}

def preprocess_image(img_path):

    # Load image with PIL
    img = Image.open(img_path)

    # Resize and convert to RGB if needed
    img = img.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    return img_array

def evaluate_test_set(test_csv_path, image_folder=""):
    # Load test set
    test_df = pd.read_csv(test_csv_path)
    
    true_labels = []
    pred_labels = []
    inference_times = []
    
    for idx, row in test_df.iterrows():
        filename = os.path.basename(row['filepath'])  # Fix here
        img_path = os.path.join(image_folder, filename)
        true_label = row['label']
        
        try:
            # Measure inference time
            start_time = time.perf_counter()
            
            # Preprocess and predict
            input_tensor = preprocess_image(img_path)
#            print(f"just preprocessed image : {img_path}")
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Record time
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Get prediction
            probabilities = np.squeeze(output)
            predicted_class_idx = np.argmax(probabilities)

            print(f"[{idx+1}] Predicted: {class_indices[predicted_class_idx]}, True: {class_indices.get(true_label, true_label)}, Inference time: {inference_time:.3f} sec")

            # Store results
            true_labels.append(true_label)
            pred_labels.append(predicted_class_idx)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    # Performance stats
    avg_inference_time = np.mean(inference_times) * 1000  # in milliseconds
    fps = 1 / np.mean(inference_times)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Total images processed: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nPerformance Metrics:")
    print(f"Average inference time: {avg_inference_time:.2f} ms per image")
    print(f"Throughput: {fps:.2f} FPS (frames per second)")
    print("="*50 + "\n")
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_indices.values(),
                yticklabels=class_indices.values())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Return all metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time_ms': avg_inference_time,
        'fps': fps,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    import argparse
    print("parsing ...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, required=True, 
                       help="Path to test CSV file")
    parser.add_argument("--image_folder", type=str, default="",
                       help="Optional folder prefix for image paths")
    args = parser.parse_args()
    
    # Run evaluation
    print("evaluating..")
    metrics = evaluate_test_set(args.test_csv, args.image_folder)
    
    # Optionally save metrics to file
    with open("evaluation_results.txt", "w") as f:
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                f.write(f"{key}: {value}\n")
