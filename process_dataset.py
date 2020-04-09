from frame_utils import get_embedding
from frame_utils_tf import load_dataset
from keras.models import load_model
from numpy import asarray, savez_compressed

trainX, trainY = load_dataset('5-celebrity-faces-dataset/train/')
testX, testY = load_dataset('5-celebrity-faces-dataset/val/')

savez_compressed('5-celebrity-faces.npz', trainX, trainY, testX, testY)

model = load_model('facenet_keras.h5')
newTrainX = []
for face_coords in trainX:
    embedding = get_embedding(model, face_coords)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)

newTestX = []
for face_coords in testX:
    embedding = get_embedding(model, face_coords)
    newTestX.append(embedding)
newTestX = asarray(newTestX)

savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainY, newTestX, testY)
