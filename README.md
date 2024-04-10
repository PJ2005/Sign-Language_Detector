# Sign-Language_Detector
This repository contains the necessary files to build and run a Sign Language Detector<br/>
> Goal: To Create a real time **Sign Language Detection** using sequences.<br/><br/>

This project facilitates seamless interaction for individuals with hearing impairments and speech disabilities by enabling the detection and interpretation of sign language gestures through the lens of Machine Learning.

## Dependencies
<ul>
  <li>OpenCV</li>
  <li>Numpy</li>
  <li>Matplotlib</li>
  <li>Mediapipe</li>
  <li>TensorFlow</li>
</ul>

## Basic Working
&emsp;**1. Data Processing:** Video frames are extracted and processed using **OpenCV and Mediapipe**. This involves ***detecting key points*** indicative of&emsp; sign language gestures.<br/>
&emsp;**2. Model Training:** A **Long Short-Term Memory (LSTM) model**, facilitated by Keras, is trained on the processed data. This model learns to &emsp;***recognize patterns*** in the key points extracted from sign language gestures.<br/>
&emsp;**3. Real-time Prediction:** With the trained model, ***real-time sign language gestures*** are captured via webcam and passed through the model &emsp;for interpretation. The model predicts the ***most probable gesture*** performed by the user.<br/>

## Detailed WorkFlow
<ol>
  <li>Structure and label the dataset from [Kaggle] containing sign language gestures.</li>
  <li>Use [OpenCV] to extract frames from the video dataset, converting the color code from RGB to BGR.</li>
  <li>Employ [Mediapipe-Holistic] to detect essential keypoints in the extracted frames.</li>
  <li>Visualize and validate the detected keypoints on the frames using [Mediapipe-Drawings].</li>
  <li>Store the extracted keypoint results in an array [numpy], considering them as features for analysis.</li>
  <li>Develop an [LSTM model] using [Keras] for sequence processing of the extracted features.</li>
  <li>Develop an [LSTM model] using [Keras] for sequence processing of the extracted features.</li>
  <li>Configure the LSTM model with three layers: two LSTM layers returning sequences and another LSTM layer followed by three [dense layers].</li>
  <li>Dense layers: two with [ReLU] activation, followed by a [softmax] activation output layer, providing class probabilities.</li>
  <li>Compile and train the model. Save the adjusted weights of the trained model.</li>
  <li>Utilize [scikit-learn] to derive a confusion matrix and calculate the accuracy score for model evaluation.</li>
  <li>Capture real-time video from the webcam using [OpenCV].</li>
  <li>Pass the captured video frames to the trained model to predict the most probable sign gesture performed by the user.</li>
</ol>
