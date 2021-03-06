# GOAL
# classify input image, using neural network saved to file
# USAGE
# python classify_covid19.py --model model/vggnet.model --labelbin labels/labels --image test/[malignant|benign]/*.png

# import the necessary packages
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras

import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (224, 224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained VGG16 neural network and the label
# binarizer
print("[INFO] loading network...")
lb = pickle.loads(open(args["labelbin"], "rb").read())

model = tf.keras.models.load_model(args["model"])
# classify the input image
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# we'll mark our prediction as "correct" of the input image filename
# contains the predicted label text (obviously this makes the
# assumption that you have named your testing image files this way)
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
print(filename,label)
correct = "correct" if filename.rfind(label) != -1 else "to revision"

# build the label and draw the label on the image
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)

