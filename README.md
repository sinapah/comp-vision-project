<h2>Emotion Detection with Convolutional Neural Networks (CNN)</h2>
This project focuses on extending an existing code base (https://github.com/atulapra/Emotion-detection?tab=readme-ov-file) to improve the number of emotions detected and it's applicability. 

<h3>Project Structure</h3>

* dataset_prepare.py - Prepares the FER-2013 dataset by reading fer2013.csv, converting each data row into an image, and organizing images by emotion labels for training and testing.
* emotions.py - Contains the main CNN model for training and testing emotion detection. It also includes code for testing the trained model on sample images and videos.
* fer2013.csv - The dataset file containing pixel data and emotion labels for each image.
* model.h5 and model.weights.h5 - Saved model files that allow loading a pre-trained model instead of training from scratch.
* haarcascade_frontalface_default.xml - A pre-trained face detector from OpenCV, used to locate faces in images.
* plot.png - A visualization of the model's accuracy and loss over training epochs.
* data - Directory where processed images are stored after running dataset_prepare.py.
* requirements.txt - Specify the necessary requirements for running the code.
* code_references.txt - Lists the resources used for the code.
* transformations.ipynb - Code used for augmenting images in the dataset. Does not need to be run since augmented images are included. Provided for completeness. 


<h3>Key Features</h3>

1. Improved dataset:
    * We have increased the dataset by including images from both FER-2013 and AffectNet. In addition, through image augmentation we have decreased the class imbalance with emotions with fewer samples (i.e. disgust and contempt). This helps the model generalize better to new images, improve robustness and increase accuracy.
2. Static Image Classification:
    * The code gives the option of using the model to classify the emotions in a static image. 
3. Emotion Detection in Videos:
    * The model is able to interact with a provided video to detect the emotions in it. It then displays the breakdown of the detected emotions in a bar graph. Currently the code is able to classify up to 3 faces in the screen at a time. 


<h3>Setup Instructions</h3>

1. Clone the Repository:
            git clone https://github.com/sinapah/comp-vision-project.git cd <repository-folder> 
2. Set Up Virtual Environment:
            python3 -m venv emotion_env source emotion_env/bin/activate 
3. Install Dependencies:
             pip install numpy pandas pillow opencv-python tensorflow matplotlib tqdm 
4. Download and Place fer2013.csv:
    * Ensure fer2013.csv is in the main project directory.
    * If necessary, modify the file path in dataset_prepare.py to reflect the location of your CSV file.


<h3>Running the Project</h3>

1. Prepare the Dataset:
    * Run the data preparation script to convert fer2013.csv into images and organize them for training and testing: bash Copy code   python3 src/dataset_prepare.py
2. Train the Model:
    * To train the CNN model with augmented data, run emotions.py in training mode: bash Copy code   python3 src/emotions.py --mode train
3. Test the Model::
    * After training, evaluate the model using test images: bash Copy code   python3 src/emotions.py --mode test
4. Launch the GUI:
   * Once the model is trained, you can launch the GUI to detect emotions in real-time. The GUI allows loading images or using a webcam feed to display detected emotions with intensity levels.



<h3>Model Architecture</h3>

The CNN model in emotions.py is structured as follows:
* Convolutional Layers - For detecting spatial features in images.
* MaxPooling Layers - For reducing spatial dimensions and computation.
* Dropout Layers - To prevent overfitting by randomly disabling neurons during training.
* Dense Layers - Fully connected layers for final emotion classification.
* Softmax Output - For multiclass classification of emotions and their intensities.


<h3>Results</h3>
The model's training performance, including accuracy and loss, is visualized in plot.png. With the applied data augmentation techniques, the model achieves improved generalization and robustness in emotion detection.

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
