
# Architecture of covid 19 neural network
# is based vgg16 default network
# contains 5 shells,
# image are converted to 224 pixels w, 224 pixels h, and 3 color deeps
# import the necessary packages
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

class COVID19Net:
	@staticmethod
	def build():
          # load the VGG16 network, ensuring the head FC layer sets are left off
          baseModel = VGG16(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))
          # construct the head of the model that will be placed on top of the
          # the base model
          headModel = baseModel.output
          headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
          headModel = Flatten(name="flatten")(headModel)
          headModel = Dense(64, activation="relu")(headModel)
          headModel = Dropout(0.5)(headModel)
          headModel = Dense(2, activation="softmax")(headModel)
          
          # place the head FC model on top of the base model (this will become
          # the actual model we will train)
          model = Model(inputs=baseModel.input, outputs=headModel)
          # return the constructed network architecture
          # loop over all layers in the base model and freeze them so they will
          # *not* be updated during the first training process
          for layer in baseModel.layers:
            layer.trainable = False
          return model