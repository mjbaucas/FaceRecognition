import cv2

def readjust_coordinates(x,y,w,h):
	w_rm = int(0.1 * w/2)
		
	x1 = x + w_rm
	x2 = x + w - w_rm
	y1 = y
	y2 = y + h	
	
	return x1, y1, x2, y2
		
def crop_faces(frame, coordinates):
	faces = []
	
	for (x, y, w, h) in coordinates:
		x1, y1, x2, y2 = readjust_coordinates(x, y, w, h)
		faces.append(frame[y1: y2, x1: x2])

	return faces

def resize_faces(faces, size=(128,128)):
	normalized = []
	for face in faces:
		if face.shape < size:
			face_norm = cv2.resize(face, size, interpolation=cv2.INTER_AREA)
		else:
			face_norm = cv2.resize(face, size, interpolation=cv2.INTER_CUBIC)
		normalized.append(face_norm)

	return normalized

def isolate_faces(frame, coordinates):
	cropped_faces = crop_faces(frame, coordinates)
	resized_faces = resize_faces(cropped_faces)

	return resized_faces
