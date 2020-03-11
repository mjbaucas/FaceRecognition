import tensorflow as tf

from keras import backend
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

def inception_block(x):
	dformat = 'channels_first'

	o = x			

	o = Conv2D(96, (1, 1), activation='relu', data_format=dformat, name='inception_3x3_conv')(o)
	o = BatchNormalization(axis=1, epsilon=0.0001, name='inception_3x3_norm')(o)
 	o = ZeroPadding2D(padding=(1, 1), data_format=dformat)(o)
	
	

