from pyimagesearch.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
from keras import models

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg",
	choices=("vgg", "resnet", "catdog"),
	help="model to be used")
args = vars(ap.parse_args())

# initialize the model to be VGG16
Model = models.load_model('catdog.h5')
Model.save_weights('save_weights.h5')
# check to see if we are using ResNet
if args["model"] == "resnet":
	Model = ResNet50
# load the pre-trained CNN from disk
'''print("[INFO] loading model...")
model = Model(weights="imagenet")'''

model = Model.load_weights('save_weights.h5')

orig = cv2.imread(args["image"])
resized = cv2.resize(orig, (64,64))

image = load_img(args["image"], target_size=(64, 64))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)

preds = Model.predict(image)
i = np.argmax(preds[0][0])
'''decoded = imagenet_utils.decode_predictions(preds)
(imagenetID, label, prob) = decoded[0][0]
label = "{}: {:.2f}%".format(label, prob * 100)
print("[INFO] {}".format(label))'''

cam = GradCAM(Model, i)
heatmap = cam.compute_heatmap(image)

heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)