from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray

# Based on tutorial and code from 
# https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
def extract_face_from_img(filename, face_size=(160,160)):
	
	# Pre-process image
	img = Image.open(filename)
	img = img.convert('RGB')
	img_arr = asarray(img)

	# Detect face
	face_detector = MTCNN()
	face_coord = face_detector.detect_faces(img_arr)
	x1, y1, width, height = face_coord[0]['box']
	
	# Adjust coordinates
	x1 = abs(x1)
	x2 = x1 + width
	y1 = abs(y1)
	y2 = x2 + height

	# Get face coordinates
	face = img_arr[y1:y2, x1:x2]
	img = Image.fromarray(face)
	img = img.resize(face_size)
	return asarray(img)