from numpy import load, expand_dims, asarray
from matplotlib import pyplot
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from random import choice
from frame_utils import get_embedding
from frame_utils_tf import extract_face_from_img
from keras.models import load_model

#data = load('5-celebrity-faces.npz')
#testX_faces = data['arr_2']

data = load('5-celebrity-faces-embeddings.npz')
trainX = data['arr_0']
trainY = data['arr_1']
#testX = data['arr_2']
#testY = data['arr_3']

encode_in = Normalizer(norm='l2')

nsamples, nx, ny = trainX.shape
trainX = trainX.reshape((nsamples,nx*ny))

model = load_model('facenet_keras.h5')
img = extract_face_from_img('kaling1.jpg')
testX = asarray(get_embedding(model, img))

nx, ny = testX.shape
testX = testX.reshape((-1, nx*ny))

trainX = encode_in.transform(trainX)
testX = encode_in.transform(testX)

encode_out = LabelEncoder()
encode_out.fit(trainY)

trainY = encode_out.transform(trainY)
# testY = encode_out.transform(testY)

model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainY)

#selection = choice([i for i in range(testX.shape[0])])
#random_face_pixels = testX_faces[selection]
#random_face_emb = testX[selection]
#random_face_class = testY[selection]
#random_face_name = encode_out.inverse_transform([random_face_class])

yhat_class = model.predict(testX)
yhat_prob = model.predict_proba(testX)

class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = encode_out.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
#print('Expected: %s' % random_face_name[0])

pyplot.imshow(testX)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()