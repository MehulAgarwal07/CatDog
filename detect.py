import cv2
import numpy as np
from PIL import Image
from keras import models

#Load the saved model
model = models.load_model('catdog.h5')
video = cv2.VideoCapture(0)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

while True:
        _, frame = video.read()
        
        '''print (frame.shape)'''
        
        im = cv2.resize(frame, (64,64), fx=0,fy=0)

        #Convert the captured frame into RGB
        '''im = Image.fromarray(frame, 'RGB')'''

        #Resizing into 128x128 because we trained the model with this image size.
        '''im = cv2.resize(frame,(64,64))'''
        '''img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        prediction = model.predict(img_array)[0][0]
        training_set.class_indices
        if prediction[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
            
        print(prediction)'''
        import numpy as np
        from keras.preprocessing import image
        test_image = image.img_to_array(im)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
        
        print(prediction)

        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        '''if prediction == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)'''

        cv2.imshow("Capturing", frame)                                                                                                                                                      
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
print(result)
video.release()
cv2.destroyAllWindows()