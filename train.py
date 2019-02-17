import pandas as pd
import numpy as np
from keras.applications.resnet50 import ResNet50

from keras.applications.densenet import DenseNet201
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input


from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
import scipy.misc
import matplotlib.pyplot as plt
IMG_WIDTH = 96
IMG_CHANNEL = 3
EPOCH_NUM = 50
SPLIT_NUM = 100

def LoadData( split_idx, DATA_DIR_PATH, LABEL_PATH ):
	y = pd.read_csv( LABEL_PATH )
	

	DATA_NUM = int(y.shape[0] / SPLIT_NUM )
	
	y = y[ split_idx * DATA_NUM : (split_idx + 1) * DATA_NUM ]

	print( y.head() )
	y = y.reset_index(drop=True)
	data = np.zeros( shape = ( DATA_NUM, IMG_WIDTH, IMG_WIDTH, IMG_CHANNEL ) )
	
	for idx in range( DATA_NUM ):
		img_path = DATA_DIR_PATH + y['id'][ idx ] +'.tif'
		# print( img_path )
		img = image.load_img( img_path )
		# img.show()
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		data[ idx, :, :, : ] = x
	
	return data, y['label']


# def LoadData( DATA_DIR_PATH, LABEL_PATH ):
# 	y = pd.read_csv( LABEL_PATH )
	
# 	DATA_NUM = int(y.shape[0] / 100)
# 	y = y[ : DATA_NUM ]
# 	print( y.head() )
# 	data = np.zeros( shape = ( DATA_NUM, IMG_WIDTH, IMG_WIDTH, IMG_CHANNEL ) )
	
# 	for idx in range( DATA_NUM ):
# 		img_path = DATA_DIR_PATH + y['id'][ idx ] +'.tif'
# 		# print( img_path )
# 		img = image.load_img( img_path )
# 		# img.show()
# 		x = image.img_to_array(img)
# 		x = np.expand_dims(x, axis=0)
# 		x = preprocess_input(x)
# 		data[ idx, :, :, : ] = x
	
# 	return data, y['label']

def GetDenseNet201( shape = ( IMG_WIDTH, IMG_WIDTH, IMG_CHANNEL ), num_classes = 2 ):
	base_model = DenseNet201( weights='imagenet', include_top=False )

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	x = Dense(512, activation='relu')(x)
	x = Dense(256, activation='relu')(x)
	x = Dense(128, activation='relu')(x)
	x = Dense(64, activation='relu')(x)

	predictions = Dense( num_classes, activation='softmax' )(x)

	model = Model( inputs=base_model.input, outputs=predictions )
	my_optim = Adam(lr=0.0001,beta_1=0.9,beta_2=0.99,epsilon=1e-8)
	model.compile( loss='categorical_crossentropy', optimizer=my_optim,metrics=['accuracy'] )
	return model

def GetResnet50( shape = ( IMG_WIDTH, IMG_WIDTH, IMG_CHANNEL ), num_classes = 2 ):
	base_model = ResNet50(weights='imagenet', include_top=False)

	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	x = Dense(512, activation='relu')(x)
	x = Dense(256, activation='relu')(x)
	x = Dense(128, activation='relu')(x)
	x = Dense(64, activation='relu')(x)
	# and a logistic layer -- let's say we have 200 classes
	predictions = Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)
	my_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
	model.compile(loss='categorical_crossentropy',optimizer= my_optim,metrics=['accuracy'])
	
	return model


# def GetResnet50( shape = ( IMG_WIDTH, IMG_WIDTH, IMG_CHANNEL ), num_classes = 2 ):
# 	base_model = ResNet50(weights='imagenet', include_top=False)

# 	# add a global spatial average pooling layer
# 	x = base_model.output
# 	x = GlobalAveragePooling2D()(x)
# 	# let's add a fully-connected layer
# 	x = Dense(1024, activation='relu')(x)
# 	# and a logistic layer -- let's say we have 200 classes
# 	predictions = Dense(num_classes, activation='softmax')(x)

# 	# this is the model we will train
# 	model = Model(inputs=base_model.input, outputs=predictions)
# 	my_optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# 	model.compile(loss='categorical_crossentropy',optimizer= my_optim,metrics=['accuracy'])
	
# 	return model




def TrainModel( epoch, model, trainX, trainY ):
	trainY = pd.get_dummies( trainY )
	# print( trainY.head() )
	print( 'Epoch In All Training:', epoch )
	model.fit( trainX, trainY, epochs = 1, batch_size=32, validation_split = 0.2 )
	return model







# def TrainModel( trainX, trainY ):

# 	X_train, X_test, y_train, y_test = train_test_split( trainX, trainY, test_size = 0.2 )
# 	trainY = pd.get_dummies( trainY )
# 	# print( trainY.head() )
# 	model_resnet50 = GetResnet50()
# 	model_resnet50.fit( trainX, trainY, epochs = 50, batch_size=32, validation_split = 0.2 )


# 	return model_resnet50



def main():
	# model = GetResnet50()
	model = GetDenseNet201()
	for epoch in range( EPOCH_NUM ):
		for split_idx in range( SPLIT_NUM ):
			trainX, trainY = LoadData( split_idx, u'../train/data/', u'../train/label/train_labels.csv' )
			model = TrainModel( epoch, model, trainX, trainY )
	model.save( '../model/densenet201.h5' )
	return


# def main():
# 	trainX, trainY = LoadData( u'../train/data/', u'../train/label/train_labels.csv' )
# 	model = TrainModel( trainX, trainY )
# 	return


if __name__ == '__main__':
	main()


