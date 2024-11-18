import importlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import cv2
import joblib
from sklearn.neighbors import KNeighborsClassifier
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("***Please wait while the keras libraries are loaded***")
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

###############################################=========FUNCTIONS=============

def test_model():
    
    setup_NN()
    val_datagen = ImageDataGenerator(rescale=1./255) # It assumes the images are already in place
    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=intBatch_size,
            color_mode="grayscale",
            class_mode='categorical')
    
    
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
    model.load_weights('model.weights.h5')
    testing_generator = val_datagen.flow_from_directory(
            testing_dir,
            target_size=(48,48),
            batch_size=intBatch_size,
            color_mode="grayscale",
            class_mode='categorical')

    
    num_test = 42
    batch_size = 32
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(testing_generator, steps=num_test // batch_size)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    

def load_and_preprocess_image(image_path):
    """
    Loads and preprocess the image: resize it to 48x48 and convert to grayscale
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale image
    img = cv2.resize(img, (48, 48))  # Resize the image to 48x48
    img = np.expand_dims(img, axis=-1)  # Add a channel dimension (48x48x1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 48, 48, 1)
    img = img / 255.0  # Rescale pixel values to [0, 1]
    
    return img

def setup_NN():
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))#, input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    


def train_NN_model():
    setup_NN()
    
    
    # Getting around the decay argument, deprecated from Adam
    learning_rate_schedule =ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=intBatch_size,
            color_mode="grayscale",
            class_mode='categorical')
    
    val_datagen = ImageDataGenerator(rescale=1./255) # It assumes the images are already in place
    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=intBatch_size,
            color_mode="grayscale",
            class_mode='categorical')

    
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=learning_rate_schedule),metrics=['accuracy'])
    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // intBatch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // intBatch_size)
    
    model.save_weights('model.weights.h5')
    
    

def predict_emotion(image_path):
    
    setup_NN()
    model.load_weights(model_file)
    clf=joblib.load('knn_model.job') # Pre-processed KNN model
    
    
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    
    # Predict the emotion using the model
    predictions = model.predict(img)
    
    # Get the index of the highest probability (the predicted class)
    predicted_class = np.argmax(predictions, axis=-1)
    
    # Map the predicted class to the corresponding emotion label
    predicted_emotion = emotion_labels[predicted_class[0]]
    
    print(f"Predicted Base Emotion: {predicted_emotion}")
    
    #============ Intensity prediction
    
    y = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale image
    y = cv2.resize(y, (48, 48))  # Resize the image to 48x48
    y=y.flatten()
    # print(y, y.size)
    
    y_pred=clf.predict([y])
    y_pred=float(y_pred[0])
    if y_pred>0.6:
        intensity="Very"
    elif y_pred<0.3:
        intensity="Slightly"
    else:
        intensity="Regular"
    
    print("The enhanced emotion was predicted as: ", intensity, predicted_emotion)


def emotion_intensity_training(image_dataset):    
       
  
    # USE THE CSV IMAGE DATA ITSELF FOR THE X TRAIN AND THE EMOTION DECIMALS FOR THE Y_TRAIN
    
    dfCSV = pd.read_csv(image_dataset, header=None)
    dfCSV_Train=dfCSV[dfCSV[2]=="Training"]
    dfCSV_Test=dfCSV[dfCSV[2]=="PublicTest"]
    X_Train=dfCSV_Train[1].str.split(' ', expand=True)
    X_Test=dfCSV_Test[1].str.split(' ', expand=True)
    
    y_Train=dfCSV_Train[0]
    y_Test=dfCSV_Test[0]
    
    
    # Train the regression model
    clf.fit(X_Train, y_Train)
    joblib.dump(clf, 'knn_model.job')
    
    

def show_menu():
    print("*** PROGRAM START ***\n")
    print(" Welcome to the Image Emotion predictor \n")
    print("PLEASE SELECT AN OPTION: \n")
    print(" 1. Train the model \n \
2. Test the model trained\n \
3. Predict an emotion\n \
4. Exit\n")


    
if __name__=='__main__':    #==================== START OF SCRIPT
    
    
    
    #============================================ VARIABLE DECLARATION
    
    # Dictionary to map the output to the corresponding empotion labels
    emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        
    # Define data generators
    train_dir = 'data/train'
    val_dir = 'data/test' # this is for validation
    testing_dir = 'data/testing-images'  # this is for testing
    image_dataset ='D:/src/fer2013_range_test.csv' # This is the orignal CSV file with the emotion and intensities classification, and the flattened image
    model_file='model.weights.h5'
    knn_model_file='knn_model.job'
    #learning_rate_schedule=''
    
    # Define model parameters
    
    num_train = 28709
    num_val = 7136
    intBatch_size = 64
    num_epoch = 50 
    
    
    #=============================================== CMD LINE MENU
    
    while True:
        show_menu()
        choice = input("Choose an option (1-3): ")
        
        #=========SETTING UP THE MODEL VARIABLES
        
        if choice in ['1','2','3']:
           
            
            # Create the model
            model = Sequential([layers.Input(shape=(48,48,1))])
           
            # Setup the  for emotion intensities
            clf = KNeighborsClassifier(n_neighbors=5)
        
        if choice == '1': # Training
            print("You chose option 1")
            num_epoch=int(input("This process takes a significant amount of time per epoch (cycle), please enter the desired number of epochs:_"))
            train_NN_model()
            emotion_intensity_training(image_dataset)
            print('*** Training is complete ***')
        elif choice == '2': # Testing
            print("You chose option 2")
            test_model()
        elif choice == '3': # Predict
            
            if os.path.isfile(model_file) and os.path.isfile(knn_model_file):
                image_path=input("Please enter the file path and name of the image file_")
                print("Analyzing ", image_path)
                predict_emotion(image_path)
                print('*** DONE ***')
            else:
                print("Model has not been trained, please select option 1")
            
        elif choice == '4': # Exit
            print("Exiting the menu. Goodbye!")
            break
        else:
            print("Invalid choice! Please select a valid option (1-3).")

