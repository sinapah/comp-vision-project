import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import seaborn as sns
import face_recognition

#from sklearn.neighbors import KNeighborsClassifier
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("***Please wait while the keras libraries are loaded***")
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


###############################################=========FUNCTIONS=============

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


def test_model(): # Model testing Train vs Testing intances to provide accuracy
    
    setup_NN()  # Sets up the model layers
    
    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
    model.load_weights('model.weights.h5') # loads the previously trained model
    
    val_datagen = ImageDataGenerator(rescale=1./255) # It assumes the images are already in place
    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=intBatch_size,
            color_mode="grayscale",
            class_mode='categorical')
    
  
    testing_generator = val_datagen.flow_from_directory(
            testing_dir,
            target_size=(48,48),
            batch_size=intBatch_size,
            color_mode="grayscale",
            class_mode='categorical')

    
    #num_test = 3589
    #batch_size = 32
    # Evaluate the model on the testing set
    loss, accuracy = model.evaluate(testing_generator)
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

def setup_NN(): # Sets up the Artificial Neural Network layers, kernel size and type of activation
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
    model.add(Dense(8, activation='softmax')) # 8 for 8 emotions (original 7 plus contempt)
    


def train_NN_model(): # This function trains the Artificial Neural Network
    setup_NN() # This function sets up the ANN details
    
    
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
    plot_model_history(model_info)
    

def predict_emotion(image_path):
    
    setup_NN()
    model.load_weights(model_file)
    weights, biases = model.layers[1].get_weights()
    
    # Load and preprocess the image
    img = load_and_preprocess_image(image_path)
    
    # Predict the emotion using the model
    predictions = model.predict(img)
    
    # Get the index of the highest probability (the predicted class)
    predicted_class = np.argmax(predictions, axis=-1)
    
    # Map the predicted class to the corresponding emotion label
    predicted_emotion = emotion_labels[predicted_class[0]]
    
    print(f"Predicted Base Emotion: {predicted_emotion}")
    #print('Predictions: ', predictions)

def create_emotion_graph(title, values):
  #Define categories and colours for bars:
  categories = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised', 'Contempt']
  colours = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink']
  
  #Define the width and height of the graph. 
  fig, ax = plt.subplots(figsize=(10, 4), dpi=200) #width, height

  #Create the graph
  ax.bar(categories, values, color=colours)
  ax.set_xlabel('Emotions', fontsize=18)
  ax.set_ylabel('Probability of Emotion in Current Frame (for current frame)', fontsize=18)
  ax.set_title('Emotion Probability: ' + title, fontsize=18)
  ax.tick_params(axis='x', labelsize=15)  
  ax.tick_params(axis='y', labelsize=14)  
  return fig

def image_padding(image):
  #Make window 3000px, divide padding for video into top and bottom 
  top_padding = (3000-image.shape[0]) // 2
  bottom_padding = 3000 - top_padding - image.shape[0]

  #Add padding 
  image = cv2.copyMakeBorder(image, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, None, value = 0) 
  return image

def graph_padding(graph):
  #Adding padding to the graph, to make it 3000px
  image = cv2.copyMakeBorder(graph, 0, (3000-(graph.shape[0])), 0, 0, cv2.BORDER_CONSTANT, None, value = 0) 
  return image

def show_menu(): # Main menu
    print("*** PROGRAM START ***\n")
    print(" Welcome to the Image Emotion predictor \n")
    print("PLEASE SELECT AN OPTION: \n")
    print(" 1. Train the model \n \
2. Test the model trained \n \
3. Predict an emotion \n \
4. Apply model to a video \n \
5. Exit\n")


    
if __name__=='__main__':    #==================== START OF SCRIPT
    
    
    #============================================ VARIABLE DECLARATION
    
    # Dictionary to map the output to the corresponding emotion labels
    emotion_labels = ['angry', 'contempt','disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        
    # Define directories variables
    train_dir = 'data/train'
    val_dir = 'data/test' # this is for validation
    testing_dir = 'data/testing-images'  # this is for testing
    model_file='model.weights.h5' # Filename that keeps the values of the weights of the ANN model
    
    # Define model parameters
    
    num_train = 43361 # Number f training instances
    num_val = 5121 # number os validation and testing intances
    intBatch_size = 64
    num_epoch = 50 # Default number of epochs
    
    
    #=============================================== CMD LINE MENU
    
    while True:
        show_menu()
        choice = input("Choose an option (1-5): ")
        
        #=========SETTING UP THE MODEL VARIABLES
        
        if choice in ['1','2','3','4']:
           
            
            # Create the model
            model = Sequential([layers.Input(shape=(48,48,1))])
           
        
        if choice == '1': # Training
            print("You chose option 1")
            num_epoch=int(input("This process takes a significant amount of time per epoch (cycle), please enter the desired number of epochs:_"))
            train_NN_model()
            
            print('*** Training is complete ***')
        elif choice == '2': # Testing
            print("You chose option 2")
            test_model()
        elif choice == '3': # Predict
            
            if os.path.isfile(model_file):
                print("Please enter the file path for an image. \n (ex. sample-images/angry.png)")
                image_path=input("File path of image: ")
                print("Analyzing ", image_path)
                predict_emotion(image_path)
                print('*** DONE ***')
            else:
                print("Model has not been trained, please select option 1")
        elif choice == '4': # Apply Model to Video 
            print("Please enter the file path of a video \n (ex. videos/angry.mp4)")
            print("You can click q when OpenCV window is in focus to quit.")
            image_path=input("Video path:")

            #Set up model:
            setup_NN()
            
            #Load trained model weights:
            model.load_weights('model.weights.h5')

            i = 0 
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised", 7: "Contempt"}
            process_this_frame = True

            #Get video
            cap = cv2.VideoCapture(image_path)
            while True:
                # Get the next frame of the video
                ret, frame = cap.read()
                if ret == False:
                    print("No more frames in this video!")
                    break
               
                #Resize frame to 2/3 of the height and width to increase processing speed while still retaining quality
                height = frame.shape[0] #Frame height
                width = frame.shape[1] #Frame width
                frame = cv2.resize(frame, (int(width*2/3), int(height*2/3)))
                

                if process_this_frame:
                    # Using Haar Cascade for detcting faces
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    # Convert the frame to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Find all of the faces in the frame
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    #Keep track of the # of faces in the frame
                    counter = 0 
                    #Store emotion graphs 
                    result_graphs = None

                    # For each face detected, draw a rectangle and get the emotion breakdown for it.
                    #The code in the loop below is modified from the existing base code at https://github.com/atulapra/Emotion-detection?tab=readme-ov-file. 
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
                        

                            #Create a graph representing the predicted emotions
                            fig = create_emotion_graph("Face " + str(counter+1), prediction[0])
                            fig.canvas.draw()

                            #Convert the plot to an image (code taken from URL below)
                            # https://medium.com/@Mert.A/real-time-plotting-with-opencv-and-matplotlib-2a452fbbbaf9
                            plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) #,sep=''
                            plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                            plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR) #Convert to RGB for matplotlib

                            #Formatting emotions graphs for UI  
                            if counter < 1:
                                result_graphs = plot
                            elif counter >= 1:
                                #Append both graphs one on top of the other. 
                                result_graphs = np.vstack([result_graphs, plot])
                            counter = counter + 1

                    if counter == 0:
                        #If there aren't any faces in the current frame, have empty plot:
                        fig = create_emotion_graph("No Face Detected", [0,0,0,0,0,0,0,0])
                        fig.canvas.draw()

                        #Convert the plot to an image (code taken from URL below)
                        # https://medium.com/@Mert.A/real-time-plotting-with-opencv-and-matplotlib-2a452fbbbaf9
                        plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) #,sep=''
                        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR) #Convert to RGB for matplotlib
                        result_graphs = plot
                        
                    #Adding padding to make it 3000px 
                    plot = graph_padding(result_graphs) 

                    #Change frame width and height if needed (for visibility in video window):
                    if height < 1500 and width < 1000:
                        frame = cv2.resize(frame, (int(width*1.5), int(height*1.5)))   
                    #Adding padding to the image to make it's height 3000px  
                    frame = image_padding(frame)

                    #Append graph and image together
                    result_img = np.hstack([frame, plot])
                
                #Process every other frame (due to resource constraints)
                process_this_frame = not process_this_frame 

                # Display the resulting image
                cv2.imshow("Video", result_img)
                
                # Wait for 'q'' key to stop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                #Close plot after each frame (to reduce memory used)
                plt.close() 
            cap.release()
            cv2.destroyAllWindows()
        elif choice == '5': # Exit
            print("Exiting the menu. Goodbye! 👋🏽")
            break
        else:
            print("Invalid choice! Please select a valid option (1-5).")