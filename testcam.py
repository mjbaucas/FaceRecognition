import cv2
import time 

from numpy import load, expand_dims, asarray
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from random import choice
from frame_utils import get_embedding, extract_face_from_frame, readjust_coordinates
from keras.models import load_model

from picamera.array import PiRGBArray
from picamera import PiCamera

#################################################################
# Pre-load Model 
################################################################# 
data = load('5-celebrity-faces-embeddings.npz')
trainX = data['arr_0']
trainY = data['arr_1']

encode_in = Normalizer(norm='l2')

nsamples, nx, ny = trainX.shape
trainX = trainX.reshape((nsamples,nx*ny))

model_facenet = load_model('facenet_keras.h5')
trainX = encode_in.transform(trainX)

encode_out = LabelEncoder()
encode_out.fit(trainY)

trainY = encode_out.transform(trainY)
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

#################################################################
# Open Camera 
################################################################# 
xml_path = "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(xml_path)

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
capture = PiRGBArray(camera, size=(640, 480))

time.sleep(0)

min_width = 30
min_height = 30

font = cv2.FONT_HERSHEY_DUPLEX
rec_color = (0, 255, 0)

#################################################################
# Detection and Classification Loop
#################################################################
predicted_name = "Stranger"
frame_count = 0
start_time = time.time()
sum_acc = 0
correct_count = 0
det_count = 0

for capture in camera.capture_continuous(capture, format="bgr", use_video_port=True):
	frame = capture.array
	ret, img, coord = extract_face_from_frame(face_detector, frame)
	if ret == 0:
		x1, x2, y1, y2 = coord
		testX = asarray(get_embedding(model_facenet, img))

		nx, ny = testX.shape
		testX = testX.reshape((-1, nx*ny))

		yhat_class = model.predict(testX)
		yhat_prob = model.predict_proba(testX)

		class_index = yhat_class[0]
		class_probability = yhat_prob[0,class_index] * 100
		predicted_name = encode_out.inverse_transform(yhat_class)[0]
		#print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
		
		if predicted_name == 'marc_baucas':
			sum_acc+=class_probability
			correct_count+=1
		det_count+=1
		
		# Draw
		cv2.rectangle(frame, (x1, y1), (x2, y2), rec_color, 2)
		
		# Put label
		cv2.rectangle(frame, (x1, y1 - 10), (x2, y1), (0, 0, 255), cv2.FILLED)
		cv2.putText(frame, predicted_name, (x1 + 3, y1 - 3), font, 0.3, (255, 255, 255), 1)

	cv2.imshow('Video', frame)
	key = cv2.waitKey(1) & 0xFF
	frame_count+=1
	capture.truncate(0)
	# Exit
	if key == ord('q'):
		break

# Release webcam handle
end_time = time.time()
elapsed_time = end_time - start_time
print("Frames %d - Time %d" % (frame_count, elapsed_time))
print("Average Classifier Confidence %.3f" % (sum_acc/det_count))
print("Actual Precision %.3f" % ((correct_count/det_count)*100))
capture.release()
cv2.destroyAllWindows()
