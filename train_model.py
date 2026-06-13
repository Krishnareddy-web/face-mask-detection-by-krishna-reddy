import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Dataset path
DATASET_DIR = 'dataset'

def load_data():
    data = []
    labels = []
    classes = ['with_mask', 'without_mask']
    for idx, name in enumerate(classes):
        path = os.path.join(DATASET_DIR, name)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            image = load_img(img_path, target_size=(224,224))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(idx)
    return np.array(data), np.array(labels)

print("[INFO] Loading images...")
data, labels = load_data()
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                        shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"])
print("[INFO] Training...")
model.fit(aug.flow(trainX, trainY, batch_size=32),
          steps_per_epoch=len(trainX)//32,
          validation_data=(testX, testY),
          validation_steps=len(testX)//32, epochs=10)
print("[INFO] Saving mask detector model...")
if not os.path.exists('model'):
    os.makedirs('model')
model.save('model/mask_detector.h5')
