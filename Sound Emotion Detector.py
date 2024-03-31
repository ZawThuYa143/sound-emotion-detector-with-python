import pygame
import tkinter as tk
from tkinter.filedialog import askdirectory
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
from tkinter import messagebox
import numpy as np
from matplotlib import pyplot as plt
#import IPython.display as ipd
import pandas as pd
from tkinter import *
from PIL import ImageTk, Image

song_path = ""
predition_string = ""

#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#create the gui window
prototye_frame = tk.Tk()

#set the title of the window
prototye_frame.title("Speech Emotion Detector ( Version.1.0.0 )")

#set the size of the window
prototye_frame.geometry("750x500")

img_lab = tk.Label(prototye_frame)

#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name,mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:        
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

# Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#Emotions to observe
#observed_emotions=['calm', 'happy', 'fearful', 'disgust']
#observed_emotions=['neutral', 'happy', 'sad', 'angry']
observed_emotions=['neutral', 'calm', 'happy','sad', 'angry', 'fearful', 'disgust', 'surprised']

#Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("E:/Python Projects/TestAudio/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#open the audio folder
os.chdir("E:/Python Projects/TestAudio")
audio_list = os.listdir()

#show all existing files in the list
playlist = tk.Listbox(prototye_frame, height=15, width=25, font="arial 12 italic", bg ='#ADD8E6', selectmode=tk.SINGLE)

#for loop to play all audio
for item in audio_list:
    pos = 0
    playlist.insert(pos,item)
    pos+=1

#initialize the pygmae mixer
pygame.init()
pygame.mixer.init()

#function to play the audio
def play():
    clear_label_image()
    pygame.mixer.music.load(playlist.get(tk.ACTIVE))
    var.set(playlist.get(tk.ACTIVE))
    pre.set(" ")
    pygame.mixer.music.play()
    #create parameter for file path of selected song
    fn_wav = "E:/Python Projects/TestAudio/" + var.get()   
    x, Fs = librosa.load(fn_wav, sr=None)    
    print_plot_play(x=x, Fs=Fs, text='WAV file: ')
    

    
#to show song wave
def print_plot_play(x, Fs, text=''):
    """1. Prints information about an audio singal, 2. plots the waveform, and 3. Creates player
    
    Notebook: C1/B_PythonAudio.ipynb
    
    Args: 
        x: Input signal
        Fs: Sampling rate of x    
        text: Text to print
    """
    
    print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, Fs, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='blue')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    #ipd.display(ipd.Audio(data=x, rate=Fs))

#create function to train data
def train():
    #Split the dataset
    x_train,x_test,y_train,y_test=load_data(test_size=0.25)

    #Get the shape of the training and testing datasets
    #print((x_train.shape[0], x_test.shape[0]))

    #Get the number of features extracted
    #print(f'Features extracted: {x_train.shape[1]}')

    #Train the model
    model.fit(x_train,y_train)

    #Predict for the test set
    y_pred=model.predict(x_test)

    #Calculate the accuracy of our model
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

    #Print the accuracy
    accu.set("Accuracy: {:.2f}%".format(accuracy*100))

    # from sklearn.metrics import accuracy_score, f1_score

    # f1_score(y_test, y_pred,average=None)

    # import pandas as pd
    # df=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
    # df.head(20)
    messagebox.showinfo("Alert", "Data Training Successfully Done!")

#create final prediction result function
def final_result():
    import pickle
    # Writing different model files to file
    with open( 'modelForPrediction1.sav', 'wb') as f:
        pickle.dump(model,f)

    filename = 'modelForPrediction1.sav'
    loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

    #create parameter for file path of selected song
    song_path = "E:/Python Projects/TestAudio/" + var.get()

    feature=extract_feature(song_path, mfcc=True, chroma=True, mel=True)

    feature=feature.reshape(1,-1)

    prediction=loaded_model.predict(feature)
    #print(prediction)

    # Convert all the elements of the array to strings
    prediction = [str(element) for element in prediction]

    # Convert the array to a string, using a space as a separator
    predition_string = ' '.join(prediction)

    pre.set("The Predition Result is " + "\"" + predition_string +"\"") 
    # Print the resulting string
    # print(type(predition_string))

    #assigning emotion images
    if(predition_string == "neutral"):
        #Load the image
        image = Image.open("E:/Python Projects/Images/neutral.png")

        # Resize the image using resize() method
        resize_image = image.resize((250,250))
        
        photo = ImageTk.PhotoImage(resize_image)

        #Label widget to display the image
        img_lab = tk.Label(prototye_frame, image=photo)
        img_lab.image= photo
        img_lab.grid(row=7, column=3)

    elif(predition_string == "calm"):
        #Load the image
        image = Image.open("E:/Python Projects/Images/calm.png")

        # Resize the image using resize() method
        resize_image = image.resize((250,250))
        
        photo = ImageTk.PhotoImage(resize_image)

        #Label widget to display the image
        img_lab = tk.Label(prototye_frame, image=photo)
        img_lab.image= photo
        img_lab.grid(row=7, column=3)

    elif(predition_string == "happy"):
        #Load the image
        image = Image.open("E:/Python Projects/Images/happy.png")

        # Resize the image using resize() method
        resize_image = image.resize((250,250))
        
        photo = ImageTk.PhotoImage(resize_image)

        #Label widget to display the image
        img_lab = tk.Label(prototye_frame, image=photo)
        img_lab.image= photo
        img_lab.grid(row=7, column=3)

    elif(predition_string == "sad"):
        #Load the image
        image = Image.open("E:/Python Projects/Images/sad.png")

        # Resize the image using resize() method
        resize_image = image.resize((250,250))
        
        photo = ImageTk.PhotoImage(resize_image)

        #Label widget to display the image
        img_lab = tk.Label(prototye_frame, image=photo)
        img_lab.image= photo
        img_lab.grid(row=7, column=3)

    elif(predition_string == "angry"):
        #Load the image
        image = Image.open("E:/Python Projects/Images/angry.png")

        # Resize the image using resize() method
        resize_image = image.resize((250,250))
        
        photo = ImageTk.PhotoImage(resize_image)

        #Label widget to display the image
        img_lab = tk.Label(prototye_frame, image=photo)
        img_lab.image= photo
        img_lab.grid(row=7, column=3)

    elif(predition_string == "fearful"):
        #Load the image
        image = Image.open("E:/Python Projects/Images/fearful.png")

        # Resize the image using resize() method
        resize_image = image.resize((250,250))
        
        photo = ImageTk.PhotoImage(resize_image)

        #Label widget to display the image
        img_lab = tk.Label(prototye_frame, image=photo)
        img_lab.image= photo
        img_lab.grid(row=7, column=3)

    elif(predition_string == "disgust"):
        #Load the image
        image = Image.open("E:/Python Projects/Images/disgust.png")

        # Resize the image using resize() method
        resize_image = image.resize((250,250))
        
        photo = ImageTk.PhotoImage(resize_image)

        #Label widget to display the image
        img_lab = tk.Label(prototye_frame, image=photo)
        img_lab.image= photo
        img_lab.grid(row=7, column=3)

    elif(predition_string == "surprised"):
        #Load the image
        image = Image.open("E:/Python Projects/Images/surprised.png")

        # Resize the image using resize() method
        resize_image = image.resize((250,250))
        
        photo = ImageTk.PhotoImage(resize_image)

        #Label widget to display the image
        img_lab = tk.Label(prototye_frame, image=photo)
        img_lab.image= photo
        img_lab.grid(row=7, column=3)

#making clear image function
def clear_label_image():
    #Load the image
        image = Image.open("E:/Python Projects/Images/nothing.png")
        # Resize the image using resize() method
        resize_image = image.resize((250,250))
        
        photo = ImageTk.PhotoImage(resize_image)

        #Label widget to display the image
        img_lab = tk.Label(prototye_frame, image=photo)
        img_lab.image= photo
        img_lab.grid(row=7, column=3)

#creating buttons
Button1 = tk.Button(prototye_frame,width=18,height=1,font="arial 12 bold",text="Play",command=play,bg="#549610",fg="white")
Button2 = tk.Button(prototye_frame,width=18,height=1,font="arial 12 bold",text="Train Data",command=train,bg="#FF4949",fg="white")
Button3 = tk.Button(prototye_frame,width=18,height=1,font="arial 12 bold",text="Prediction",command=final_result,bg="#FF8D29",fg="white")

#creating free space fro rows
tk.Label(prototye_frame, text=" " ,width=5, height=1).grid(row=0, column=0)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=0, column=1)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=0, column=2)
tk.Label(prototye_frame, text=" " ,width=38, height=1).grid(row=0, column=3)

tk.Label(prototye_frame, text=" " ,width=5, height=1).grid(row=2, column=0)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=2, column=1)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=2, column=2)
tk.Label(prototye_frame, text=" " ,width=38, height=1).grid(row=2, column=3)

tk.Label(prototye_frame, text=" " ,width=5, height=1).grid(row=4, column=0)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=4, column=1)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=4, column=2)
tk.Label(prototye_frame, text=" " ,width=38, height=1).grid(row=4, column=3)

tk.Label(prototye_frame, text=" " ,width=5, height=1).grid(row=6, column=0)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=6, column=1)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=6, column=2)
tk.Label(prototye_frame, text=" " ,width=38, height=1).grid(row=6, column=3)

#creating free space fro columns
tk.Label(prototye_frame, text=" " ,width=5, height=1).grid(row=1, column=0)
tk.Label(prototye_frame, text=" " ,width=5, height=1).grid(row=3, column=0)
tk.Label(prototye_frame, text=" " ,width=5, height=1).grid(row=5, column=0)
tk.Label(prototye_frame, text=" " ,width=5, height=1).grid(row=7, column=0)

tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=1, column=2)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=3, column=2)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=5, column=2)
tk.Label(prototye_frame, text=" " ,width=20, height=1).grid(row=7, column=2)

#StringVar() holds store string data to sent text value 
var = tk.StringVar()
pre = tk.StringVar()
accu = tk.StringVar()

#create label to display the title of selected song
song_title = tk.Label(prototye_frame,font="arial 12 bold",textvariable=var)

#create label to display the title of selected song
predict_result = tk.Label(prototye_frame,font="arial 12 bold",textvariable=pre)

#create label to display the accuraccy
accuracy_result = tk.Label(prototye_frame,font="arial 12 bold",textvariable=accu)

#the pack() geometry manage the widgets
Button2.grid(row=1, column=1)
accuracy_result.grid(row=1, column=3)
Button1.grid(row=3, column=1)
song_title.grid(row=3, column=3)
Button3.grid(row=5, column=1)
predict_result.grid(row=5, column=3)
playlist.grid(row=7, column=1)

#run the gui
prototye_frame.mainloop()








