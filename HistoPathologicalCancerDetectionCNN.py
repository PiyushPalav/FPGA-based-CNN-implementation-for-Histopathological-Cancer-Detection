# Importing libraries- numpy for linear algebra, pandas for data processing, etc
import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
from glob import glob 
from skimage.io import imread
import gc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

#os.system('pip install visualkeras')
import visualkeras

base_dir = '/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/train'
df = pd.DataFrame({'path': glob(os.path.join(base_dir,'*.tif'))})
df['id'] = df.path.map(lambda x: x.split('/')[-1].split(".")[0])
labels = pd.read_csv("/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/train_labels.csv")
df_data = df.merge(labels, on = "id")

# removing this image because it caused a training error
df_data = df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec164be2']

# removing this image because it's black
df_data = df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
df_data.head(3)

# Load 80k postive and negative examples
SAMPLE_SIZE = 80000 

# take a random sample of label 0 with size equal to num samples in label 1
df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)
# filter out label 1
df_1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)

# concatenation of the dataframes
df_data = shuffle(pd.concat([df_0, df_1], axis=0).reset_index(drop=True))

# Split data into training and test # stratify=y creates a balanced validation set.
y = df_data['label']
df_train, df_val = train_test_split(df_data, test_size=0.30, random_state=101, stratify=y)

# Create directories
train_path = '/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/train'
valid_path = '/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/valid'
test_path = '/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/test'
for fold in [train_path, valid_path]:
    for subf in ["0", "1"]:
        os.makedirs(os.path.join(fold, subf))

# Set the id as the index in df_data
df_data.set_index('id', inplace=True)
df_data.head()

for image in df_train['id'].values:
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    label = str(df_data.loc[image,'label']) # get the label for a certain image
    src = os.path.join('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/train', fname)
    dst = os.path.join(train_path, label, fname)
    shutil.copyfile(src, dst)

for image in df_val['id'].values:
    fname = image + '.tif'
    label = str(df_data.loc[image,'label']) # get the label for a certain image
    src = os.path.join('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/train', fname)
    dst = os.path.join(valid_path, label, fname)
    shutil.copyfile(src, dst)

from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 32
num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32
val_batch_size = 32

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

# Data preprocessing and augmentation
datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                            horizontal_flip=True,
                            vertical_flip=True)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(32,32),
                                        batch_size=train_batch_size,
                                        color_mode="grayscale",
                                        class_mode='binary')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(32,32),
                                        batch_size=val_batch_size,
                                        color_mode="grayscale",
                                        class_mode='binary')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(32,32),
                                        batch_size=1,
                                        color_mode="grayscale",
                                        class_mode='binary',
                                        shuffle=False)

# Importing libraries for CNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam

# Defining hyperparamters
kernel_size = (3,3)
pool_size= (2,2)

model_auc = tf.keras.metrics.AUC()
model = Sequential()

model.add(Conv2D(filters=8, kernel_size=kernel_size, activation='relu', input_shape = (32, 32, 1)))
model.add(MaxPool2D(pool_size=pool_size))

model.add(Conv2D(filters=16, kernel_size=kernel_size, activation='relu'))
model.add(MaxPool2D(pool_size=pool_size))

model.add(Conv2D(filters=32, kernel_size=kernel_size, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
visualkeras.layered_view(model, type_ignore=[ZeroPadding2D, Flatten], legend=True)

# Compile the model
model.compile(Adam(0.001), loss = "binary_crossentropy", metrics=["accuracy",model_auc])

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)
history_model = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=10,
                   callbacks=[reducel, earlystopper])

plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show();

plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show();

plt.plot(history_model.history['auc'])
plt.plot(history_model.history['val_auc'])
plt.title('Model AUC ROC vs Epoch')
plt.ylabel('ROC')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show();

# Print the parameters of each layer
for layer in model.layers:
    print(layer.name)
    print(layer.trainable_variables)

conv2d_weights = model.layers[0].get_weights()[0]
conv2d_weights_file = open('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/parameters/conv2d_1_weights_updated_32.txt','w')
# iterate through the array
for i in range(3):
    for j in range(3):
        for k in range(1):
            for l in range(8):
                conv2d_weights_file.write(str(conv2d_weights[i][j][k][l])+",")
conv2d_weights_file.close()

conv2d_biases_file = open('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/parameters/conv2d_1_biases_updated_32.txt','w')
conv2d_biases  = model.layers[0].get_weights()[1]
for i in range(8):
    conv2d_biases_file.write(str(conv2d_biases[i])+",")
conv2d_biases_file.close()

conv2d_1_weights = model.layers[2].get_weights()[0]
conv2d_1_weights_file = open('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/parameters/conv2d_2_weights_updated_32.txt','w')
# iterate through the array
for i in range(3):
    for j in range(3):
        for k in range(8):
            for l in range(16):
                conv2d_1_weights_file.write(str(conv2d_1_weights[i][j][k][l])+",")
conv2d_1_weights_file.close()

conv2d_1_biases_file = open('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/parameters/conv2d_2_biases_updated_32.txt','w')
conv2d_1_biases  = model.layers[2].get_weights()[1]
for i in range(16):
    conv2d_1_biases_file.write(str(conv2d_1_biases[i])+",")
conv2d_1_biases_file.close()

conv2d_2_weights = model.layers[4].get_weights()[0]
conv2d_2_weights_file = open('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/parameters/conv2d_3_weights_updated_32.txt','w')
# iterate through the array
for i in range(3):
    for j in range(3):
        for k in range(16):
            for l in range(32):
                conv2d_2_weights_file.write(str(conv2d_2_weights[i][j][k][l])+",")
conv2d_2_weights_file.close()

conv2d_2_biases_file = open('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/parameters/conv2d_3_biases_updated_32.txt','w')
conv2d_2_biases  = model.layers[4].get_weights()[1]
for i in range(32):
    conv2d_2_biases_file.write(str(conv2d_2_biases[i])+",")
conv2d_2_biases_file.close()

fc_weights = model.layers[6].get_weights()[0]
fc_weights_file = open('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/parameters/fc_weights_updated_32.txt','w')
# iterate through the array
for i in range(512):
    for j in range(1):
                fc_weights_file.write(str(fc_weights[i][j])+",")
fc_weights_file.close()

fc_biases_file = open('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/parameters/fc_biases_updated_32.txt','w')
fc_biases  = model.layers[6].get_weights()[1]
for i in range(1):
    fc_biases_file.write(str(fc_biases[i])+",")
fc_biases_file.close()

from sklearn.metrics import roc_curve, auc, roc_auc_score

# make a prediction
y_pred_keras = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_gen.classes, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
auc_keras

# Plot ROC
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

model.save_weights('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/checkpoint_updated_32')

# Restoring model weights in a new model and predicting
# Importing libraries- numpy for linear algebra, pandas for data processing, etc
import numpy as np 
import pandas as pd 
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
from glob import glob 
from skimage.io import imread
import gc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import visualkeras


base_dir = '/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/train'
df = pd.DataFrame({'path': glob(os.path.join(base_dir,'*.tif'))})
df['id'] = df.path.map(lambda x: x.split('/')[-1].split(".")[0])
labels = pd.read_csv("/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/train_labels.csv")
df_data = df.merge(labels, on = "id")

# removing this image because it caused a training error previously
df_data = df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec164be2']

# removing this image because it's black
df_data = df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
df_data.head(3)

# Load 80k postive and negative examples
SAMPLE_SIZE = 80000 

# take a random sample of label 0 with size equal to num samples in label 1
df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)
# filter out label 1
df_1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)

# concatenation of the dataframes
df_data = shuffle(pd.concat([df_0, df_1], axis=0).reset_index(drop=True))

# Split data into training and test # stratify=y creates a balanced validation set.
y = df_data['label']
df_train, df_val = train_test_split(df_data, test_size=0.30, random_state=101, stratify=y)

# Create directories
train_path = '/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/train'
valid_path = '/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/valid'
test_path = '/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/test'

# Set the id as the index in df_data
df_data.set_index('id', inplace=True)
df_data.head()

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img

IMAGE_SIZE = 32
num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32
val_batch_size = 32

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

# Data preprocessing and augmentation
datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                            horizontal_flip=True,
                            vertical_flip=True)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(32,32),
                                        batch_size=train_batch_size,
                                        color_mode="grayscale",
                                        class_mode='binary')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(32,32),
                                        batch_size=val_batch_size,
                                        color_mode="grayscale",
                                        class_mode='binary')


for label in os.listdir(valid_path):
    i=0
    image_files = os.listdir(valid_path+f'/{label}')
    for file in image_files:
        image_name = file.split('.')[0]
        img = cv2.imread(valid_path+f'/{label}/'+file,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         
        img = cv2.resize(img,(32,32))
        img = img.reshape(1,32,32,1)

        # get the augmented image
        aug_img_iterator = datagen.flow(img,batch_size=1)
        aug_img=next(aug_img_iterator)
    
        # save the augmented image
        cv2.imwrite(f'/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/valid_augmented/{label}/{image_name}_updated_32.tif',aug_img[0,:,:,:])
        np.savetxt(f'/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/valid_augmented/{label}/{image_name}_updated_array_32',aug_img.reshape(32,32), delimiter=',', fmt='%f')
        i += 1
        if i > 20: # save 20 images
            break  # otherwise the generator would loop indefinitely

# Create a new model instance
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam

IMAGE_SIZE = 32
kernel_size = (3,3)
pool_size= (2,2)

restoredmodel_auc = tf.keras.metrics.AUC()
restoredmodel = Sequential()

restoredmodel.add(Conv2D(filters=8, kernel_size=kernel_size, activation='relu', input_shape = (32, 32, 1)))
restoredmodel.add(MaxPool2D(pool_size=pool_size))

restoredmodel.add(Conv2D(filters=16, kernel_size=kernel_size, activation='relu'))
restoredmodel.add(MaxPool2D(pool_size=pool_size))

restoredmodel.add(Conv2D(filters=32, kernel_size=kernel_size, activation='relu'))

restoredmodel.add(Flatten())
restoredmodel.add(Dense(1, activation='sigmoid'))

restoredmodel.compile(Adam(0.001), loss = "binary_crossentropy", metrics=["accuracy",restoredmodel_auc])

# Restore the weights
restoredmodel.load_weights('/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/checkpoint_updated_32')

restoredmodel.summary()
visualkeras.layered_view(restoredmodel, type_ignore=[ZeroPadding2D, Flatten], legend=True)

# Print the parameters of each layer
for layer in restoredmodel.layers:
    print(layer.name)
    print(layer.trainable_variables)

label = '1'
img_name = '9c717143448ae0cd7021c3f00cb09eec7dcfaf34'
image = np.loadtxt(f'/home/piyush/IITBombay/Sem4/EmbeddedSystemsDesign/CourseProject/histopathologic-cancer-detection/valid_augmented/{label}/{img_name}_updated_array_32',delimiter=',').reshape(1,32, 32, 1)
print(image)
print(image.shape)
print(np.max(image))
print(restoredmodel.predict(image))