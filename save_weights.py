from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from keras import models
model = models.load_model('catdog.h5')
model.save_weights('save_weights.h5')