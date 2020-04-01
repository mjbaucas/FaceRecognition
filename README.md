# FaceRecognition
Facial Recognition implementation using Tensorflow and OpenCV2 on RaspberryPi4 (testcam.py)

## Setting up
1. Make sure to install OpenCV. (Atleast version 3.4.6, which is what the code uses) 
2. Have the Pi Camera connected to the Pi via the Camera Module Port. (more help in setting that up can be found [here](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera))
3. Make sure to have Python 3.7 installed (with pip).
4. On the terminal, run 'pip install -r requirements.txt'.

## Running
1a. If you want to run through the Pi, run 'testcam.py' on the terminal
1b. If you want to run it through Windows 10, run 'testcam_tx2.py' (note that doing this would require other libraries and applications that are not documented here)

## Training
1. To add a data entry to the model, download the dataset prepared by [Dan Becker](https://www.linkedin.com/in/dansbecker/) [here](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset). (You will need to make an account to download)
2. Follow the format of the dataset by providing folders for both train and val directories.
3. Run 'process_dataset.py' to compress the newly created dataset. 
4. Create your own code to load the model via Keras and train using the 'model.fit' function. You can choose to save the fitted model using the 'model.save' function.

## Credits
Credits to the dataset training and using code and the facenet model used by this project to Jason Brownlee and his blogpost on [How to Develop a Face Recognition System Using FaceNet in Keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) and his sources. The facenet model that was used as the foundation of this project can be downloaded [here](https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn). 
