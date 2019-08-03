import pyfolder.imageProcessingFunctions as imageProcessingFunctions# unwrap_images, normalize_numpy_array
import pyfolder.GeneralUNET as UNET #UNET for processing
import time
import sys
import numpy as np
from skimage import io
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.merge import concatenate, add
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.convolutional import *
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
K.set_image_dim_ordering('th')

wanted_xy = 512 #size for network input.

##find out wanted model
print("Loading Network...")
RibbonSegmenter = UNET.GEN_UNET(wanted_xy,wanted_xy)
if(sys.argv[2] == "STRING"):
    RibbonSegmenter.load_weights('networks/STRINGS-512-512-150Epochs.hdf5')
else:
    RibbonSegmenter = UNET.GEN_UNET(1024, wanted_xy)
    RibbonSegmenter.load_weights('networks/1024-512-70Percent-DICE.hdf5')

standard_sleep = .8 #make display a bit less computery
#read in image from javascript variable.
image_path = sys.argv[1]
image = io.imread(image_path)
print("Successfully read image!") ##TODO:: check for error.
time.sleep(standard_sleep)
print("Original image shape: ", image.shape)
time.sleep(standard_sleep)

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
#    IMAGE PREPARATION #
###################################################################################################################################

## Modify image to be standardized for network input, and normalize.
image = imageProcessingFunctions.extract_or_insert_dims(image) ## Check for 4 or 2 dimensional image.
image = imageProcessingFunctions.move_channels(image) #standardize to be image of the form (1, size, size)
image = imageProcessingFunctions.normalize_numpy_array(image)
###Images need to be cropped if they arent a multiple of 512 (or whatever network size). But, cropping needs to be added on at the end so the segmentation
###fits into the image size.

dims = image.shape
# check dimensionality input - 1024/512 -- > 2 , but 1100/512 --> 2.14. crop input to largest possible.
x_cap = int(dims[1]/wanted_xy)
y_cap = int(dims[2]/wanted_xy)


image_cropped_x, image_cropped_y = imageProcessingFunctions.get_cropped_regions(image, wanted_xy, y_cap, x_cap, dims)
image = imageProcessingFunctions.crop_image(image, wanted_xy, x_cap, y_cap)

#unwrap the padded images for analysis.
padded_x_unwrap = None
padded_y_unwrap = None
padded_x = None
padded_y = None
if image_cropped_x is not None:
    padded_x = imageProcessingFunctions.pad_cropped_area(image_cropped_x, wanted_xy, pad_dim = 1)
    x_cap_crop_x, y_cap_crop_x = int(padded_x.shape[1]/wanted_xy), int(padded_x.shape[2]/wanted_xy)
    padded_x_unwrap = imageProcessingFunctions.unwrap_multiple_images(padded_x, padded_x.shape, wanted_xy, x_cap_crop_x, y_cap_crop_x)
if image_cropped_y is not None:
    padded_y = imageProcessingFunctions.pad_cropped_area(image_cropped_y, wanted_xy, pad_dim = 2)
    x_cap_crop_y, y_cap_crop_y = int(padded_y.shape[1]/wanted_xy), int(padded_y.shape[2]/wanted_xy) 
    padded_y_unwrap = imageProcessingFunctions.unwrap_multiple_images(padded_y, padded_y.shape, wanted_xy, x_cap_crop_y, y_cap_crop_y)

#print("Modified Image shape for network input...  ", image.shape, ". This will be converted back to original shape after segmentation.")
#time.sleep(standard_sleep)

## unwrap images from (30, 512, 1024) to ---> [ (2, 512, 512), (2, 512, 512) ..... ], or: a list of items, each item is a full image split into (n, 512, 512)
image_arr = []
for i in range( dims[0] ):
    current_image = image[i,:,:]
    image_arr.append( imageProcessingFunctions.unwrap_images( current_image, wanted_xy, x_cap, y_cap, dims) )

print("Tiled image size conversion.... Converted : " , image.shape , "  to : ", image_arr[0].shape, " x " , len(image_arr)) 
time.sleep(standard_sleep)

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
#    IMAGE PROCESSING #
###################################################################################################################################
print("PROCESSING") #!identifier for javascript - changing this will introduce errors for console output. ##TODO:: try to avoid this.

final_image_stack = []
total_images = int(len(image_arr) * image_arr[0].shape[0] )
images_left = total_images
time_taken = []
curr_image = 1

final_image_stack = UNET.predict_multiple_images(image_arr, dims, RibbonSegmenter, status_options=True) ##status options to output time to screen.


##Iterate through images and predict. every time you finish an image, stitch it back together and put into an updated array.
for i in range( len(image_arr) ):
    temp_img_arr = []
    for j in (range( image_arr[i].shape[0] )):
        current_image = image_arr[i][j, :, :] # image is now (512, 512)
        start_time = time.time()

        current_image = np.expand_dims(current_image, 0)         #expansion to tensor for network.
        current_image = np.expand_dims(current_image, 0)
        #predicted_image = RibbonSegmenter.predict(current_image) #Predict
        predicted_image = current_image

        temp_img_arr.append( predicted_image[0,0,:,:] )
        curr_image += 1


        #Updates for user
        time_taken.append(time.time() - start_time )
        avg_time = sum(time_taken)/len(time_taken)
        time_to_predict = int(avg_time*images_left)
        images_left = images_left - 1
        print("predicting image number:  ", curr_image , "  of  ", total_images, "."," Time left is Approximately " , time_to_predict ," seconds.")
    rewrapped_image = imageProcessingFunctions.rewrap_image( temp_img_arr, dims, wanted_xy)
    final_image_stack.append( rewrapped_image )

#same thing for X.
final_padded_x_stack = None
if padded_x_unwrap is not None:
    print("Parts of the image had to be cropped for network input... predicting those areas now.")
    final_padded_x_stack = []
    for i in range( len(padded_x_unwrap) ):
        temp_img_arr = []
        for j in (range( padded_x_unwrap[i].shape[0] )):
            current_image = padded_x_unwrap[i][j, :, :]
            current_image = np.expand_dims(current_image, 0)
            current_image = np.expand_dims(current_image, 0)
            #predicted_image = RibbonSegmenter.predict(current_image)
            predicted_image = current_image
            temp_img_arr.append( predicted_image[0,0,:,:] )
        rewrapped_image = imageProcessingFunctions.rewrap_image(temp_img_arr, padded_x.shape, wanted_xy)
        final_padded_x_stack.append( rewrapped_image )

#same thing for Y.
final_padded_y_stack = None
if padded_y_unwrap is not None:
    final_padded_y_stack = []
    for i in range( len(padded_y_unwrap) ):
        temp_img_arr = []
        for j in (range( padded_y_unwrap[i].shape[0] )):
            current_image = padded_y_unwrap[i][j, :, :]
            current_image = np.expand_dims(current_image, 0)
            current_image = np.expand_dims(current_image, 0)
            #predicted_image = RibbonSegmenter.predict(current_image)
            predicted_image = current_image
            temp_img_arr.append( predicted_image[0,0,:,:] )
        rewrapped_image = imageProcessingFunctions.rewrap_image(temp_img_arr, padded_y.shape, wanted_xy)
        final_padded_y_stack.append( rewrapped_image )



#the images are put back together, but not in their original z-stack. do it now.
dstacked_final_image = np.dstack(final_image_stack)
dstacked_final_image = np.moveaxis(dstacked_final_image, 2, 0)

unpadded_x_stack = None
unpadded_y_stack = None
if final_padded_x_stack is not None:
    dstacked_x_stack = np.dstack(final_padded_x_stack)
    unpadded_x_stack = imageProcessingFunctions.unpad_cropped_area(dstacked_x_stack, image_cropped_x.shape, padded_dim=1)
if final_padded_y_stack is not None:
    dstacked_y_stack = np.dstack(final_padded_y_stack)
    unpadded_y_stack = imageProcessingFunctions.unpad_cropped_area(dstacked_y_stack, image_cropped_y.shape, padded_dim=2)

uncropped_final_image_stack = imageProcessingFunctions.uncrop_images( dstacked_final_image, unpadded_x_stack, unpadded_y_stack, x_cap, y_cap, wanted_xy)

print("Done Processing!")

#####################################################################################################################################

#Save image.

import scipy.misc
from skimage.external import tifffile as tif

save_image = np.dstack(uncropped_final_image_stack)
save_image = np.moveaxis(save_image, 2, 0)
tif.imsave('segmented-Image.tif', save_image, bigtiff=True )
print("Image saved!")
