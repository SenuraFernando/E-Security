from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K



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
model.add(Dense(93600))
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
    'C:/Users/Senura Fernando/PycharmProjects/E-Security/Dataset',
    target_size=(img_width, img_height),
    batch_size=20,
    class_mode='categorical',
    # color_mode='grayscale',
    # save_to_dir='preview',
    # save_format='jpeg'
    )

testing = test_datagen.flow_from_directory(
        'C:/Users/Senura Fernando/PycharmProjects/E-Security/Dataset',
        target_size=(img_width, img_height),
        batch_size=128,
        class_mode='categorical',
        # color_mode='grayscale'

)

model.fit_generator(
        training,
        steps_per_epoch=610,
        epochs=30,
        validation_data=testing,
        validation_steps=20)

# to save model weights
model.save_weights('./models/trained_model_3.h5')
