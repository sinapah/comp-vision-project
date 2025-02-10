# Emotion Detection with Convolutional Neural Networks (CNN)
This project focuses on extending an existing code base (https://github.com/atulapra/Emotion-detection?tab=readme-ov-file) to improve the number of emotions detected, image detection in static images and real-time image detection in videos.

# Table of Contents
- [Emotion Detection with Convolutional Neural Networks (CNN)](#emotion-detection-with-convolutional-neural-networks-cnn)
- [Key Features](#key-features)
  - [1. Improved dataset](#1-improved-dataset)
  - [2. Static Image Classification](#2-static-image-classification)
  - [3. Emotion Detection in Videos](#3-emotion-detection-in-videos)
- [Demo](#demo)
  - [Emotion Detection in Images](#emotion-detection-in-images)
  - [Emotion Detection in Videos](#emotion-detection-in-videos)
    - [Example 1](#example-1)
    - [Example 2](#example-2)
    - [Example 3](#example-3)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)


## Key Features

### 1. Improved dataset:
- We have increased the dataset by including images from both FER-2013 and AffectNet, increasing diversity and coverage. This expansion also allows for the detection of an additional emotion, enhancing the model’s classification capabilities.
- In addition, image augmentation techniques like Gaussian noise, slight rotations, brightness/contrast adjustments, and image flipping, are employed to help the model generalize better to new images, improve robustness, decrease class imbalance for emotions with fewer samples (i.e. disgust and contempt) and increase accuracy.
### 2. Static Image Classification:
- The code gives the option of using the model to classify the emotions in a static image. 
### 3. Emotion Detection in Videos:
- The model is able to interact with a provided video to detect the emotions in it. It then displays the breakdown of the detected emotions in a bar graph.
- We optimized video processing by adjusting video quality and frame rate to improve the speed and efficiency of emotion detection. These optimizations ensure real-time performance without sacrificing accuracy.
- The system can handle multiple faces within a single video frame, ensuring that emotions can be detected simultaneously (for up to 3 faces on the screen at a time). It also adapts to varying video frame dimensions, making it versatile for different types of video input.
---

## Demo
See our project in action!

### Emotion Detection in Images

https://github.com/user-attachments/assets/e668aed6-5280-4425-a29c-3d80dd4cae74

### Emotion Detection in Videos

The examples below highlight our project's ability to analyze diverse video inputs, accurately mapping emotions to corresponding bar graphs in real-time. Our model can track multiple faces within a frame, adapt to various video frame dimensions, and label each face with its detected emotions.

#### Example 1
https://github.com/user-attachments/assets/7cdfa0b5-7696-4997-8fab-b9ff5cb3df16
#### Example 2
https://github.com/user-attachments/assets/4062ac09-c291-4e4d-991b-7600ce7f8eb3
#### Example 3
https://github.com/user-attachments/assets/7b5b7930-d46d-4fd8-9c21-9000cec0e1d2

---

## Project Structure</h3></summary>

- **dataset_prepare.py** - Prepares the FER-2013 dataset by reading fer2013.csv, converting each data row into an image, and organizing images by emotion labels for training and testing.
- **emotions.py** - Contains the main CNN model for training and testing emotion detection. It also includes code for testing the trained model on sample images and videos.
- **fer2013.csv** - The dataset file containing pixel data and emotion labels for each image.
- **model.h5 and model.weights.h5** - Saved model files that allow loading a pre-trained model instead of training from scratch.
- **haarcascade_frontalface_default.xml** - A pre-trained face detector from OpenCV, used to locate faces in images.
- **plot.png** - A visualization of the model's accuracy and loss over training epochs.
- **data** - Directory where processed images are stored after running dataset_prepare.py.
- **requirements.txt** - Specifies the necessary requirements for running the code.
- **code_references.txt** - Lists the resources used for the code.
- **data_augmentations.ipynb** - Code used for augmenting images in the dataset. Does not need to be run since augmented images are included. Provided for completeness.

---
## Setup Instructions
   
- Refer to the installation guide for this.
- You can download the PDF here: [Installation Guide](Installation%20Guide.pdf)
  
---

## Model Architecture

The CNN model in emotions.py is structured as follows:
* **Convolutional Layers** - For detecting spatial features in images.
* **MaxPooling Layers** - For reducing spatial dimensions and computation.
* **Dropout Layers** - To prevent overfitting by randomly disabling neurons during training.
* **Dense Layers** - Fully connected layers for final emotion classification.
* **Softmax Output** - For multiclass classification of emotions and their intensities.

## Results
The model's training performance, including accuracy and loss, is visualized in plot.png. With the applied data augmentation techniques, the model achieves improved generalization and robustness in emotion detection.

![plot](https://github.com/user-attachments/assets/ac584777-a0e5-4bf1-8b38-9884da9857f3)

## Dependencies

* Python 3
* TensorFlow
* Keras
* Pandas
* NumPy
* OpenCV
* Matplotlib
* TQDM


## License
This project is open-source and available under the MIT License.
