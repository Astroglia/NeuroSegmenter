
import pyfolder.imageProcessingFunctions as IPF

## General UNET? Salute! ## reference: https://tenor.com/view/himym-how-imet-your-mother-salute-ted-robin-gif-7960412
import keras
import tensorflow
import numpy as np
import sys
import time
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.merge import concatenate, add
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.convolutional import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

def insert_dim(image):
    return np.expand_dims(image, 0)

#image_arr is a list of (512,512) images. original image shape is a tuple.
def predict_multiple_images(image_arr, original_image_shape, model, status_options=None): ##status options to output time to screen.

    final_image_stack = []
    total_images = int(len(image_arr) * image_arr[0].shape[0] )
    images_left = total_images
    time_taken = []
    curr_image = 1

    for i in range( len(image_arr) ):
        temp_img_arr = []
        for j in (range( image_arr[i].shape[0] )):
            current_image = image_arr[i][j, :, :] # image is now (512, 512)
            start_time = time.time()

            predicted_image = predict_image(current_image, model)

            temp_img_arr.append( predicted_image[0,0,:,:] )

            #Updates fr user
            curr_image += 1
            time_taken.append(time.time() - start_time )
            avg_time = sum(time_taken)/len(time_taken)
            time_to_predict = int(avg_time*images_left)
            images_left = images_left - 1
            if status_options is not None:
                print("predicting image number:  ", curr_image , "  of  ", total_images, "."," Time left is Approximately " , time_to_predict ," seconds.")

        #rewrap and append
        rewrapped_image = IPF.rewrap_image( temp_img_arr, original_image_shape, image_arr[0].shape[0])
        final_image_stack.append( rewrapped_image )
    return final_image_stack


def predict_image(image, model): ##predicts image. assumes 2D input. (512, 512)
    image = insert_dim(image)
    image = insert_dim(image)
    return model.predict(image)

## e.g. difference in number of filters (2-->4-->8 or 2-->8-->16)
def find_dividers(size, between_encoding_lengths):
    divider_arr = []
    previous_append = -between_encoding_lengths
    ## if we have a number that is divisible by two and the image size, and is less than half the size of the image length, 
    # and we haven't filled up all the spots yet, and it is greater than the encoding length limit, then append to the array.
    for i in range(size):
        if i > 0:
            if (i % 2) == 0:
                if (size % i) == 0:
                    if (size/i) >= 2:
                        if len(divider_arr) < 4:
                            if (i - previous_append) >= between_encoding_lengths:
                                divider_arr.append(i)
                                previous_append = i
    
    return divider_arr                                

def GEN_UNET(size, image_size):
    
    inputs = Input((1, image_size , image_size))
    
   # div_arr = find_dividers(size, divider_length)
   # print("Filter Numbers: ", div_arr)
    
    div16 = int(size/16)#int(size/div_arr[3])
    div8  = int(size/8)#int(size/div_arr[2])
    div4  = int(size/4)#int(size/div_arr[1])
    div2  = int(size/2)#int(size/div_arr[0])

    conv1 = Conv2D(div16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(div16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(div8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Dropout(.2)(conv2)
    conv2 = Conv2D(div8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(div4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Dropout(.2)(conv3)
    conv3 = Conv2D(div4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(div2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Dropout(.2)(conv4)
    conv4 = Conv2D(div2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(size, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Dropout(.3)(conv5)
    conv5 = Conv2D(size, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)

    up6 = concatenate([Conv2DTranspose(size, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv5), conv4], axis=1)
    conv6 = Conv2D(div2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
    conv6 = Dropout(.3)(conv6)
    conv6 = Conv2D(div2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)

    up7 = concatenate([Conv2DTranspose(div2, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv6), conv3], axis=1)
    conv7 = Conv2D(div4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
    conv7 = Dropout(.3)(conv7)
    conv7 = Conv2D(div4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)

    up8 = concatenate([Conv2DTranspose(div4, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv7), conv2], axis=1)
    conv8 = Conv2D(div8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up8)
    conv8 = Dropout(.3)(conv8)
    conv8 = Conv2D(div8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)

    up9 = concatenate([Conv2DTranspose(div8, (2, 2), strides=(2, 2), padding='same',kernel_initializer='he_normal')(conv8), conv1], axis=1)
    conv9 = Conv2D(div16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up9)
    conv9 = Dropout(.3)(conv9)
    conv9 = Conv2D(div16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid',kernel_initializer='he_normal')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model