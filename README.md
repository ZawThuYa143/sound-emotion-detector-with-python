# Speech Emotion Recognition (SER) using Machine Learning

Data Science Project
Mg Zaw Thu Ya
 
# Abstract
	Speech Emotion Recognition, abbreviated as SER, is the act of attempting to recognize human emotion and affective states from speech. The goal is to determine the emotional state of a speaker, such as happiness, anger, sadness, or frustration, from their speech patterns, such as prosody, pitch, and rhythm. This is also the phenomenon that animals like dogs and horses employ to be able to understand human emotion. SER is tough because emotions are subjective and annotating audio is challenging. The best example of it can be seen at call centers. If you ever noticed, call centers employees never talk in the same manner, their way of pitching/talking to the customers changes with customers. Now, this does happen with common people too, but how is this relevant to call centers? Here is answer, the employees recognize customers’ emotions from speech, so they can improve their service and convert more people. In this way, they are using speech emotion recognition. In this project, 8 emotions are divided by using mainly three features. And, over 1500 audio data set are used to train and test for emotions classification.

# Introduction
	Speech emotion recognition is a simple Python mini-project, which is used to practice for data science project. The 8 kinds of emotions are aimed to predict from the system. They are 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust' and 'surprised'. This project is developed by creating GUI prototype in order to use easily and show clear work. It was able to recognize emotion from audio files. The data are loaded and extracted features from it, then split the dataset into training and testing sets. Then, an MLPClassifier is initialized and trained the model. After that, the accuracy of model is calculated and showed. Any desired audio file is opened and the prototype showed the frequency of the selected sound file. Finally, based on the selected audio file the system predicted the features and showed the emotion of that audio. Over 1500 audio data of 24 actors (also male and female) are used as a data set of this project. From the audio data, three key features have extracted which have been used in this study, MFCC (Mel Frequency Cepstral Coefficients), Mel Spectrogram and Chroma. Librosa (Python Library) was used in those features extraction. Training and testing data set are divided into 75% train and 25% test to create prediction model and calculate accuracy. 

# Data Source
For this project, the RAVDESS dataset is used and that is the Ryerson Audio-Visual Database of Emotional Speech and Song dataset, and is free to download. This dataset has 24 files rated by 10 individuals on emotional validity, intensity, and genuineness. The entire dataset is 162MB from 24 actors, but have lowered the sample rate on all the files.


# Library Used
The libraries librosa, soundfile, pygame, numpy, matplotlib, tkinter (GUI for python) and sklearn (among others) to build a model using an MLPClassifier are mainly used to build entire project.

# Features Used
MFCC : MFCC was by far the most researched about and utilized feature in this dataset. It represents the short-term power spectrum of a sound.
Mel Spectrogram : This is just a spectrogram that depicts amplitude which is mapped on a Mel scale.
Chroma : A Chroma vector is typically a 12-element feature vector indicating how much energy of each pitch class is present in the signal in a standard chromatic scale.

![image](https://github.com/ZawThuYa143/sound-emotion-detector-with-python/assets/152624230/d649f396-ae2c-4a44-9f5b-a8449d27b5db)
 
Figure 1 : Create a function to extract features base on three features of selected audio
 
# Project Prototype
	First of all, the simple GUI is appeared and the buttons and audios are showed to make clear command.
 
![image](https://github.com/ZawThuYa143/sound-emotion-detector-with-python/assets/152624230/3257c0f4-ed0a-48b0-b81b-2f16f692e8fd)

Figure 2 : GUI of Speech Emotion Recognition Prototype

	After the “Train Data” button is clicked, the system make process on added audio data and show accuracy result. The time to train data can be long according to the data set that used.

![image](https://github.com/ZawThuYa143/sound-emotion-detector-with-python/assets/152624230/dd9531c2-5851-4807-9178-fcc814aa70da)

Figure 3 : Train data and show accuracy value

  While the audio file is selected and click “Play” button, the system played audio and showed the frequency feature of that audio file with selected file name.

![image](https://github.com/ZawThuYa143/sound-emotion-detector-with-python/assets/152624230/4498127c-a031-4203-9800-5214aaa35ca2)


Figure 4 : Play selected audio file and show frequency

	Finally, the “Prediction” button gave the prediction emotion result of the selected audio file.

![image](https://github.com/ZawThuYa143/sound-emotion-detector-with-python/assets/152624230/2bd4c1f4-3f9c-4d38-9a7f-ec10de841b9c)

Figure 5 : Predict the selected audio’s emotion

# Observation
	Firstly, the accuracy value is unstable while the system is run and it always show different value. The accuracy is lower than 70% while 8 emotions are extracted. If 4 or lower than 4 emotions are extracted, the higher value of accuracy is founded. But there is some issue. While lower than 4 emotions (For example, 2 emotions) are extracted features, the training and testing data sets are also changed and a warning is shown because of lower than 500 data sets are used. This may be the impact of low amount of audio data set are used in this project. For bigger and better projects, the reduction on emotions extracting can be the way to make higher accuracy.
	Secondly, while 8 emotions are tried to predict on different audio data, it showed unstable prediction and sometime wrong prediction. This is because of low accuracy on this data set. So, another solution to get higher accuracy is using bigger audio data set for training and testing the system. According to some research, better experience of extracting emotions is based on the use of higher performance data set.
	Finally, the trueness of data prediction is based on the accuracy value and the accuracy is changed while the change in dividing of the training and testing data set values. For this project, training data is 75% and testing data is 25% was used. When the training set to 80% and the testing to 20% was changed, the change in accuracy value is founded. 
	
# Conclusion
	In recent years, Speech Emotion Recognition (SER) technology as one of the key technologies in human-computer interaction systems, has received a lot of attention from researchers at home and abroad for its ability to accurately recognize emotions and thus improve the quality of human-computer interaction. In this project, we saw how we can use Machine learning to find the emotion from speech audio data and some perception on the human expression of emotion through voice. This system can be set up in a variety of organizations like Call Centre for complaints or marketing, in voice-based virtual assistants or chatbots, in linguistic research, etc.

# References
https://www.projectpro.io/article/speech-emotion-recognition-project-using-machine-learning/573
https://medium.com/analytics-vidhya/speech-emotion-recognition-using-machine-learning-df31f6fa8404
https://www.irjmets.com/uploadedfiles/paper/volume_3/issue_12_december_2021/17485/final/fin_irjmets1638949366.pdf
https://youtu.be/-VQL8ynOdVg?si=irnu35SLaC9rvqAa
