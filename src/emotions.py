import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/test/predict")
ap.add_argument("-image_name",help="image name.png")
mode = ap.parse_args().mode
image_path=ap.parse_args().image_name

# Dictionary to map the output to the corresponding empotion labels
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def load_and_preprocess_image(image_path):
    """
    Load and preprocess the image: resize it to 48x48 and convert to grayscale
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale image
    img = cv2.resize(img, (48, 48))  # Resize the image to 48x48
    img = np.expand_dims(img, axis=-1)  # Add a channel dimension (48x48x1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 48, 48, 1)
    img = img / 255.0  # Rescale pixel values to [0, 1]
    
    return img


def predict_emotion(image_path):
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    
    # Predict the emotion using the model
    predictions = model.predict(img)
    
    # Get the index of the highest probability (the predicted class)
    predicted_class = np.argmax(predictions, axis=-1)
    
    # Map the predicted class to the corresponding emotion label
    predicted_emotion = emotion_labels[predicted_class[0]]
    
    print(f"Predicted Emotion: {predicted_emotion}")



'''
# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    #axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1)) #Chaged above line to this
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    #axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1))
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()
'''
# Define data generators
train_dir = 'data/train'
val_dir = 'data/test' # this is for validation
testing_dir = 'data/testing-images'  # this is for testing

num_train = 28709
num_val = 7136
batch_size = 64
num_epoch = 50 #changed from 50 to 5

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
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

# Getting around the decay argument, deprecated from Adam
learning_rate_schedule =ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

# The next line assumes the training already took place and loads the generated model weights
model.load_weights('model.weights.h5')


# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=learning_rate_schedule),metrics=['accuracy'])
    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    #plot_model_history(model_info)
    #model.save_weights('model.h5')
    model.save_weights('model.weights.h5')


# emotions will be displayed on your face from the webcam feed
elif mode == "test":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
    model.load_weights('model.weights.h5')
    testing_generator = val_datagen.flow_from_directory(
            testing_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    
    num_test = 42
    batch_size = 32
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(testing_generator, steps=num_test // batch_size)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
# predict an image emption    
if mode == "predict":
    #image_path = 'happy_test.png'  # Specify the path to your image
    predict_emotion(image_path)