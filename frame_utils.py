import cv2

from os import listdir
from os.path import isdir

from PIL import Image
from numpy import asarray, savez_compressed, expand_dims

def readjust_coordinates(x,y,w,h):
	x1 = abs(x)
	x2 = x1 + w
	y1 = abs(y)
	y2 = x2 + h
	
	return x1, y1, x2, y2
		
def extract_face_from_frame(face_detector, frame, face_size=(160,160)):
	# Detect face
	face_coord = face_detector.detectMultiScale(
		frame,
		scaleFactor = 1.2,
		minNeighbors = 4,
		minSize = (int(30), int(30)) 	
	)
	if len(face_coord) > 0:
		x1, y1, width, height = face_coord[0]
		
		# Adjust coordinates
		x1 = abs(x1)
		x2 = x1 + width
		y1 = abs(y1)
		y2 = y1 + height

		# Get face coordinates
		face = frame[y1:y2, x1:x2]
		img = Image.fromarray(face)
		img = img.resize(face_size)
		return 0, asarray(img), (x1, x2, y1, y2)
	return 1, None, None



def load_faces(directory):
	faces = []
	for filename in listdir(directory):
		path = directory + filename
		face = extract_face_from_img(path)
		faces.append(face)
	return faces

def load_dataset(directory):
	x = []
	y = []

	for subdirectory in listdir(directory):
		path = directory + subdirectory + '/'
		if not isdir(path):
			continue

		faces = load_faces(path)
		labels = [subdirectory for i in range(len(faces))]
		x.extend(faces)
		y.extend(labels)
	return asarray(x), asarray(y)

def get_embedding(model, face_arr):
	face_arr = face_arr.astype('float32')
	mean = face_arr.mean()
	std = face_arr.std()
	face_arr = (face_arr - mean)/std
	face_samples = expand_dims(face_arr, axis=0)
	prediction = model.predict(face_samples)
	return prediction