from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import model_from_json


# Just disables the warning, doesn't enable AVX/FMA
#import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# input image dimensions
img_width, img_height = 150, 150

# the CIFAR10 images are RGB
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
#input
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape,padding='same'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#first convo
model.add(Conv2D(32, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#second convo
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


#third convo
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#fully connected
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])


# this will do preprocessing and realtime data augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training = train_datagen.flow_from_directory(
    'C:/Users/Senura Fernando/PycharmProjects/E-Security/Dataset/train',
    target_size=(img_width, img_height),
    batch_size=10,
    class_mode='categorical',
    # color_mode='grayscale',
    # save_to_dir='preview',
    # save_format='jpeg'
    )

testing = test_datagen.flow_from_directory(
        'C:/Users/Senura Fernando/PycharmProjects/E-Security/Dataset/test',
        target_size=(img_width, img_height),
        batch_size=10,
        class_mode='categorical',
        # color_mode='grayscale'

)

model.fit_generator(
        training,
        steps_per_epoch=10,
        epochs=5,
        validation_data=testing,
        validation_steps=2)

# to save model weights
model.save_weights('./models/ai_model.h5')
# ... code
K.clear_session()
# config = model.to_json()
# open("face_recog_model_structure.json", "wb").write(config)


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)