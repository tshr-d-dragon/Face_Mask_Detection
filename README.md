# Face_Mask_Detection

### Face_Mask_Detection  A Deep Learning project that checks whether a person is wearing face mask or not live on webcam

This projects helps detecting Face Mask. I used MobilenetV2 because of comparatively lesser size and good performance.

It uses Google's mediapipe library to detect faces. For more information, please visit https://google.github.io/mediapipe/solutions/face_detection.html.

## Project Structure
1. Model_training.ipynb file gives the walkthrough over the training part of classification of the Face Mask. Weights for MobilenetV2 model trained for 10 epochs is MobileNetV2.h5 file.   
2. Face_Mask_Detection.py file contains opencv code for face detection, classification as well as prediction on webcam.
3. requirements.txt file contains all the dependencies.

## To run the prject, follow below steps
1. Ensure that you are in the project home directory
2. Create anaconda environment
3. Activate environment
4. >pip install -r requirement.txt
5. >python Face_Mask_Detection.py

## Please feel free to connect for any suggestions or doubts!!!

## Credits
1. The credits for dataset used for training goes to https://github.com/balajisrinivas/Face-Mask-Detection


##### For better prediction, we need better image quality dataset for training.
