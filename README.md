# CDTA-Waste-detection
This repository contains the code and data for an internship project conducted at CDTA. The project focuses on developing and deploying an AI model on a Raspberry Pi 5 for the purpose of classifying waste into recyclable categories to be used in robotic sorting systems, specifically tailored for the Algerian context.

## Project Structure

```
.
├── data/                    # Dataset and annotation files
│   ├── annotations/           # COCO-formatted annotation files
│   └── images/               # All training and validation images
├── code/                   # All source code
│   ├── training/             # Scripts for model training and evaluation
│   └── deployment/           # Scripts for Raspberry Pi deployment
├── requirements.txt       # Python dependencies
└── README.md              # This file

```

# Set up

## 1. Prequisities

- A Raspberry Pi 5 (for deployment)
- A compatible camera (e.g., Kinect, Raspberry Pi Camera Module)

### 2. Installation

1. Clone the repository:
    
    ```bash
    git clone <https://github.com/your-username/ai-waste-classification.git>
    cd ai-waste-classification
    
    ```
    
2. Install the required Python packages:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    

### 3. Using the Pre-trained Model

1. Download the pre-trained TFLite model from the [Releases](https://github.com/your-username/ai-waste-classification/releases) page.
2. Place the model file (e.g., `best_model.tflite`) in the `code/deployment/` directory.
3. Connect your camera to the Raspberry Pi.
4. Run the real-time inference script:
    
    ```bash
    cd code/deployment
    python pi_inference.py
    
    ```
    

## Dataset

The model is trained on a merged dataset of over **35,000 images** across **9 waste categories**, specifically curated and adapted for material-based recycling.

**Categories:**

- `Cardboard`
- `Electronic Waste`
- `Glass`
- `Metals`
- `Organic Waste`
- `Paper`
- `Plastics`
- `Textiles`
- `General Waste`

**Location:** `data/`

- `data/images/`: Contains all image files.
- `data/annotations/`: Contains the `instances_train.json` and `instances_val.json` files in COCO format.

## Model Training

The code for training and evaluating the models is located in `code/training/`.

### Key Features:

- Supports multiple architectures: **MobileNetV2, ResNet50, EfficientNetB3, EfficientNetV2B0**.
- Implements a two-phase transfer learning strategy (frozen base + fine-tuning).
- Includes comprehensive data augmentation and class imbalance handling (class weighting).
- Outputs trained models in `.h5` and converted `.tflite` formats.

## Deployment on Raspberry Pi

The deployment scripts are in `code/deployment/`.

### Hardware Used:

- **Raspberry Pi 5**
- **Kinect Camera** (or any compatible USB camera)

### Key Scripts:

- `pi_inference.py`: Main script for running real-time inference using a camera feed. It displays the classification result and latency metrics on the screen.
- `model_utils.py`: Contains helper functions for loading the TFLite model and processing images.
- `test_single_image.py`: Script to test the model on a single image file.

### Performance on Raspberry Pi 5:

- **Inference Speed:** ~85-90 ms per image (≈11-12 FPS)
- **Model:** Quantized TensorFlow Lite (`.tflite`) version of MobileNetV2
- **Accuracy:** >96% on the test set

## Results

Our best models achieved the following performance:

| Model | Accuracy | F1-Score | Params | Best For |
| --- | --- | --- | --- | --- |
| **ResNet50 + SVM** | **98.00%** | **97.99%** | 25M + SVM | Highest Accuracy |
| **MobileNetV2** | 96.83% | 96.80% | **3.5M** | **Edge Deployment** |
| EfficientNetB3 | 96.10% | 96.05% | 12M | Balanced Performance |

## Key Features

- **Edge Optimized:** Model is compressed and quantized for fast inference on resource-constrained devices.
- **Real-Time Capable:** Achieves near real-time performance on a Raspberry Pi 5.
- **Algerian Context:** Dataset and classification system are adapted for local waste types (e.g., high organic content, Arabic text).
- **Comprehensive Training:** Includes advanced techniques like data augmentation, class weighting, and transfer learning.

## Authors

- **HADJ MESSAOUD Maria**
- **LITIM Rania**

*Supervised by: **Mrs. AKLI Isma***

## License

This project is licensed for academic and research purposes. Please contact the authors for commercial use.

## Acknowledgments

- Advanced Technology Development Centre (CDTA) for hosting the internship.
- The creators of all the public datasets (TACO, TrashNet, etc.) used to build our merged dataset.
- The open-source communities behind TensorFlow, Keras, and OpenCV.
