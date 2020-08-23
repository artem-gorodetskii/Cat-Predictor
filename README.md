# Cat-Predictor
This project represents the integration of TensorFlow Lite into Android application. 
The application allows to take photos and predict probability of cats on them. The picture below shows two examples of application screenshots:

<p align="center">
  <img src="images/android_results.png" width="500" />
</p>

## Project structure
* The "Android App" directory contains Android Studio project.
* The "Tensorflow model" directory contains Jupyter notebooks with TensorFlow implementation of the model.

## Model architecture and training process
The MobileNetv2 architecture was used as a base model.
Firstly, the MobileNetv2 model was trained on dataset containing 17 different classes including: guitar, flower, car, motobike, airplane, face, 
ship, dog, house, bottle, bird, background, cat, camel, watch, chair and panda.
