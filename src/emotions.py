import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import face_recognition
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
mode = ap.parse_args().mode

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
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1)) #Chaged above line to this
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1))
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

def create_emotion_graph(title, values):
  categories = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
  colours = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink']
  px = 1/plt.rcParams['figure.dpi']  # pixel in inches
  fig, ax = plt.subplots(figsize=(9, 4), dpi=200) #width, height
  ax.bar(categories, values, color=colours)
  ax.set_xlabel('Emotions', fontsize=18)
  ax.set_ylabel('Probability of Emotion in Current Frame (for current frame)', fontsize=18)
  ax.set_title('Emotion Probability: ' + title, fontsize=18)
  ax.tick_params(axis='x', labelsize=15)  
  ax.tick_params(axis='y', labelsize=14)  
  return fig

def image_padding(image):
  top_padding = (3000-image.shape[0]) // 2
  bottom_padding = 3000 - top_padding - image.shape[0]
  image = cv2.copyMakeBorder(image, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, None, value = 0) 
  return image

def graph_padding(graph):
  image = cv2.copyMakeBorder(graph, 0, (3000-(graph.shape[0])), 0, 0, cv2.BORDER_CONSTANT, None, value = 0) 
  return image

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709 + 436 #Adding 436 to account for newly added images to disgust folder. 
num_val = 7178
batch_size = 64
num_epoch = 40 #changed from 50 to 5

#Preparing images for training:
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

#Load images and prep them to be used by deep learning model 
train_generator = train_datagen.flow_from_directory(
        train_dir,                  #Folder images are being loaded from
        target_size=(48,48),        #Resizing images
        batch_size=batch_size,      #Batch size
        color_mode="grayscale",     #Load images in grayscale
        class_mode='categorical')   #Classification type

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        shuffle=False,
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

# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    model.save_weights('model.weights.h5')

    loss, accuracy = model.evaluate(train_generator, steps=num_val // batch_size)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")


    """
    train_accuracy = model_info.history['accuracy']
    val_accuracy = model_info.history['val_accuracy']
    print("The train accuracy is " + str(train_accuracy))
    print("The test accuracy is " + str(val_accuracy))

    #Creating confusion matrix:
    y_pred = model.predict(validation_generator)#Predicted values
    y_actual = validation_generator.classes     #Actual labels
    class_labels = list(validation_generator.class_indices.keys())

    cm = confusion_matrix(y_true = y_actual, y_pred = np.argmax(y_pred, axis=1)) #Np.argmax to get the class with highest probability
    sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.show()
    """

elif mode == "test":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
    model.load_weights('model.weights.h5')
    testing_generator = val_datagen.flow_from_directory(
            val_dir,
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



# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    """
    model.load_weights('model.weights.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    """
    print("Webcam would have deployed, commented out temporarily.")
    i = 0 
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    #Video application features from here:
    cap = cv2.VideoCapture("videos/sad.mp4")
    face_locations = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()

        #Get current height and width of the image
        height = frame.shape[0]
        width = frame.shape[1]
        print(frame.shape)
        #Resize frame to 2/3 of the height and width to increase processing speed while still retaining quality
        frame = cv2.resize(frame, (int(width*2/3), int(height*2/3)))

        if process_this_frame:
            """
            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            for top, right, bottom, left in face_locations:
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            """

            # Load the Haar Cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # Frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Find all faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            #Counter to keep track of the number of faces in the frame
            counter = 0 
            #returned_graphs will take emotions graph for each face and append them.
            result_graphs = None

            # For each face detected, draw a rectangle and get the emotion breakdown for it.
            for (x, y, w, h) in faces:
                if counter < 3:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face" + str(counter+1), (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    #Co-ordinates of the selected rectangle
                    roi_gray = gray[y:y + h, x:x + w] 
                    #Resize the rectangle
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    #Make a prediction (will return the probability breakdown of each emotion)
                    prediction = model.predict(cropped_img)
                    #Convert prediciton to numpy array 
                    prediction = np.array(prediction)
                    #print(prediction) #ex.[[7.2752759e-01, 2.3053539e-05, 5.0827023e-04, 9.4853419e-07, 9.6426476e-05,9.9926704e-01, 9.7008124e-05]]

                    #Used for writing out the breakdown of emotions in legible format
                    emotion_breakdown = ""
                    for j in range(len(prediction[0])):
                        emotion_breakdown = emotion_breakdown + " " + emotion_dict[j] + ": " + str(prediction[0][j])
                    #Returns emotion and corresponding percentage:
                    print(emotion_breakdown) 
                    #ex.  Angry: 8.481411e-09 Disgusted: 0.00028628838 Fearful: 1.0516198e-05 Happy: 0.94973195 Neutral: 9.047412e-09 Sad: 0.04997055 Surprised: 6.466511e-07

                    #We plot the returned emotion prediction
                    fig = create_emotion_graph("Face " + str(counter+1), prediction[0])
                    fig.canvas.draw()

                    #Convert the plot to an image (code taken from URL included below)
                    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
                    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR) #Convert to RGB for matplotlib


                    if counter < 1:
                        result_graphs = plot
                    elif counter >= 1:
                        #Append both graphs one on top of the other. 
                        result_graphs = np.vstack([result_graphs, plot])
                    counter = counter + 1
                
            print(str(counter))
            plot = graph_padding(result_graphs)
            #plot = image_padding(result_graphs)
            frame = image_padding(frame)
            result_img = np.hstack([frame, plot])
            print(result_img.shape) #Height, width, 3000 x 3040
        
        process_this_frame = not process_this_frame #Process every other frame (due to resource constraints)
        

        # Display the resulting image
        cv2.imshow("Image", result_img)

        #Printing out which frame we're on:
        print("This is frame: ", i)
        i = i + 1
        
        # Wait for 'q'' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





#Resources:
# https://stackoverflow.com/questions/53351963/mnist-get-confusion-matrix 
# https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
#https://stackoverflow.com/questions/58887056/resize-frame-of-cv2-videocapture
#https://github.com/ageitgey/face_recognition/issues/1336
#https://note.nkmk.me/en/python-opencv-pillow-image-size/ #Getting height, width of image
#https://pyimagesearch.com/2021/01/20/opencv-resize-image-cv2-resize/ #Resizing an image
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html #Adjusting figure height in pixels
# https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/ #For adding padding
# https://stackoverflow.com/questions/6390393/how-to-change-tick-label-font-size

#References for videos used:
#Video by Mikhail Nilov: https://www.pexels.com/video/a-couple-looking-at-a-smartphone-screen-6963479/
#Video by cottonbro studio: https://www.pexels.com/video/an-excited-young-female-at-game-arcade-5767473/ 
# Video by fauxels: https://www.pexels.com/video/close-up-video-of-man-wearing-red-hoodie-3249935/ 