from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
from frame_utils import isolate_faces, readjust_coordinates

xml_path = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(xml_path)

#gst_str = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
#capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
capture = PiRGBArray(camera, size=(640, 480))

time.sleep(0)

min_width = 30
min_height = 30

name = 'person'
font = cv2.FONT_HERSHEY_DUPLEX
rec_color = (0, 255, 0)

for capture in camera.capture_continuous(capture, format="bgr", use_video_port=True):
	frame = capture.array
	faces = detector.detectMultiScale(
		frame,
		scaleFactor = 1.2,
		minNeighbors = 4,
		minSize = (int(min_width), int(min_height)) 	
	)
	
	isolated_faces = isolate_faces(frame, faces)
	for i, face in enumerate(isolated_faces):
		cv2.imwrite('test{}.jpeg'.format(i), face)
	
	for (x, y, w, h) in faces:
		x1, y1, x2, y2 = readjust_coordinates(x, y, w, h)

		# Draw
		cv2.rectangle(frame, (x1, y1), (x2, y2), rec_color, 2)
		
		# Put label
		cv2.rectangle(frame, (x1, y1 - 10), (x2, y1), (0, 0, 255), cv2.FILLED)
		cv2.putText(frame, name, (x1 + 3, y1 - 3), font, 0.3, (255, 255, 255), 1)

	cv2.imshow('Video', frame)
	key = cv2.waitKey(1) & 0xFF
	
	capture.truncate(0)
	# Exit
	if key == ord('q'):
		break

# Release webcam handle
capture.release()
cv2.destroyAllWindows()
