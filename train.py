# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
from keras.preprocessing.image import ImageDataGenerator
import os
import config

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Optional: Control GPU usage

sz = config.IMG_SIZE

# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.40))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=27, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

# Step 2 - Preparing the train/test data and training the model
classifier.summary()

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(config.TRAIN2_DIR,
                                                 target_size=(sz, sz),
                                                 batch_size=config.BATCH_SIZE,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(config.TEST2_DIR,
                                            target_size=(sz , sz),
                                            batch_size=config.BATCH_SIZE,
                                            color_mode='grayscale',
                                            class_mode='categorical') 

# Calculate steps per epoch dynamically
steps_per_epoch = max(1, training_set.samples // config.BATCH_SIZE)
validation_steps = max(1, test_set.samples // config.BATCH_SIZE)

classifier.fit(
        training_set,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        validation_data=test_set,
        validation_steps=validation_steps)

# Saving the model
if not os.path.exists(config.MODEL_DIR):
    os.makedirs(config.MODEL_DIR)

model_json = classifier.to_json()
with open(config.MODEL_BW_JSON, "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights(config.MODEL_BW_H5)
print('Weights saved')

