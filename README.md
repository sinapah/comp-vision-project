<h2>Emotion Detection with Convolutional Neural Networks (CNN)</h2>
This project focuses on extending an existing code base (https://github.com/atulapra/Emotion-detection?tab=readme-ov-file) to improve the number of emotions detected, image detection in static images and real-time image detection in videos.

<h3>Key Features</h3>

<h4>1. Improved dataset:</h4>
    * We have increased the dataset by including images from both FER-2013 and AffectNet, increasing diversity and coverage. This expansion also allows for the detection of an additional emotion, enhancing the model’s classification capabilities.
    * In addition, image augmentation techniques like Gaussian noise, slight rotations, brightness/contrast adjustments, and image flipping, are employed to help the model generalize better to new images, improve robustness, decrease class imbalance for emotions with fewer samples (i.e. disgust and contempt) and increase accuracy.
<h4>2. Static Image Classification:</h4>
    * The code gives the option of using the model to classify the emotions in a static image. 
<h4>3. Emotion Detection in Videos:</h4>
    * The model is able to interact with a provided video to detect the emotions in it. It then displays the breakdown of the detected emotions in a bar graph.
    * We optimized video processing by adjusting video quality and frame rate to improve the speed and efficiency of emotion detection. These optimizations ensure real-time performance without sacrificing accuracy.
    * The system can handle multiple faces within a single video frame, ensuring that emotions can be detected simultaneously (for up to 3 faces on the screen at a time). It also adapts to varying video frame dimensions, making it versatile for different types of video input.

<details>
<summary><h3>Project Structure</h3></summary>

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
</details>

<details>
<summary><h3>Setup Instructions</h3></summary>
   
* Refer to the installation guide for this.
* You can download the PDF here: [Example PDF](https://github.com/username/repository-name/raw/main/docs/example.pdf)

</details>

<details>
<summary><h3>Model Architecture</h3></summary>

The CNN model in emotions.py is structured as follows:
* **Convolutional Layers** - For detecting spatial features in images.
* **MaxPooling Layers** - For reducing spatial dimensions and computation.
* **Dropout Layers** - To prevent overfitting by randomly disabling neurons during training.
* **Dense Layers** - Fully connected layers for final emotion classification.
* **Softmax Output** - For multiclass classification of emotions and their intensities.
</details>

<details>
<summary><h3>Results</h3></summary>
The model's training performance, including accuracy and loss, is visualized in plot.png. With the applied data augmentation techniques, the model achieves improved generalization and robustness in emotion detection.

![An image with two graphs showing model loss and accuracy.]([URL](https://github.com/sinapah/comp-vision-project/blob/final-version/src/plot.png))

</details>

<h3>Dependencies</h3>

* Python 3
* TensorFlow
* Keras
* Pandas
* NumPy
* OpenCV
* Matplotlib
* TQDM


<h3>License</h3>
This project is open-source and available under the MIT License.
