import numpy as np

#normalizes image to range between 0 and 1. If cap_outliers=True, it ignores values higher than 2 sdevs from the mean.
def normalize_numpy_array(numpy_img, cap_outliers=False):
    if(cap_outliers):
        mean = np.mean(numpy_img.flatten()) #Get mean/sdev
        sdev = np.std(numpy_img.flatten())
        if(cap_outliers):
            thresh = mean + sdev + sdev
            numpy_img_threshold = numpy_img > thresh #Exclude large values > 2 sdev
            numpy_img[numpy_img_threshold] = thresh
    
    numpy_img = (numpy_img-np.min(numpy_img))/(np.max(numpy_img)-np.min(numpy_img))    
    return numpy_img

#just calls unwrap_images for an entire image set, to clean up the main code.
def unwrap_multiple_images(image, dims, wanted_xy, x_cap, y_cap):
    image_arr = []
    for i in range( dims[0] ):
        current_image = image[i,:,:]
        image_arr.append( unwrap_images( current_image, wanted_xy, x_cap, y_cap, dims) )
    return image_arr


def unwrap_images(numpy, wanted_xy, x_cap, y_cap, dims):
    x1 = 0
    x2 = int(wanted_xy)
    y1 = 0
    y2 = int(wanted_xy)
    img_slices = []
    
    for x_iter in range( x_cap ): ## 1536/256 --> 6
        for y_iter in range( y_cap ):            
            curr_image = numpy[x1:x2, y1:y2]
            curr_image = curr_image.astype(np.float) ##conversion to floats
            img_slices.append(curr_image)
            
            y1 = y1 + wanted_xy
            y2 = y2 + wanted_xy
        
        y1 = 0
        y2 = wanted_xy
        x1 = x1 + wanted_xy
        x2 = x2 + wanted_xy
    return np.array(img_slices)

def get_larger_dim_3D(shape):
    if(shape[1] > shape[2]):
        return 1
    else:
        return 2

def rewrap_multiple_images(image_list, OGshape, wanted_xy):
    img_arr = []
    for i in range(len(image_list)):
        img_arr.append(rewrap_image( image_list[i], OGshape, wanted_xy))
    return np.asarray(img_arr)

def rewrap_image(image, OGshape, wanted_xy=512):
    if isinstance(image, list): #check for list input.
        image = np.stack(image, axis = 0 )

    y_portion_to_take = int( OGshape[2] / wanted_xy )

    img_append = []
    J = 0
    for i in range(0, image.shape[0], y_portion_to_take ): # from zero to the image shape, in steps of 2, 3.. whatever y_portion_to_take is.
        J = y_portion_to_take + J
        cross_append = np.concatenate( (image[i:J, : , :]), axis = 1  ) #axis is 1 because image[0, :,:] * y_portion_to_take is what's returned by image[i:J, :,:]
        img_append.append(cross_append)

    final_image = np.concatenate(img_append, axis=0)
    return final_image

def rewrap_images(image_array, original_shape, wanted_xy=512):
    if image_array[0].ndim == 2: ##if we have a list of: [ (512, 512), (512, 512) ...], dstack the image (processing step puts array into a list like this)
        temp_array = np.stack(image_array, axis=0)
        image_array = []
        image_array.append(temp_array)

    if 5 == 5:
   # if (len(image_array) % 2) != 0: #if we have a non-square image (this works for square images too.)
        #unwrap_images traverses in the y direction, so get amount in the y direction to take.
        y_portion_to_take = int( original_shape[2] / wanted_xy )
        elongated_y_set = []
        for i in  range(len( image_array )):
            temp_img = None
            current_image_set = image_array[i]
            for j in range(y_portion_to_take):
                if j != 0:
                    temp_img = np.append( temp_img,  current_image_set[j, : , :], axis= 1)
                else:
                    temp_img = current_image_set[0, :, :]
            i = i + y_portion_to_take
            elongated_y_set.append( temp_img )

        final_image = np.stack( elongated_y_set, axis=0)
        return final_image

##takes your uncropped image, and uncrops it.
def uncrop_images(image_array, cropped_x, cropped_y, x_cap, y_cap, wanted_xy):
    uncropped_image = image_array
    ##check if there was cropping in neither dimension, just x, just y, or both.
    if (cropped_x is None) and (cropped_y is None): 
        uncropped_image = image_array
    elif cropped_x is None:
        uncropped_image = np.column_stack( (uncropped_image, cropped_y) )
    elif cropped_y is None:
        uncropped_image = np.column_stack( (uncropped_image, cropped_x) )
    else:
        image_concat_x = np.column_stack( (image_array, cropped_x))
        cropped_y = np.append( cropped_y, np.zeros([cropped_y.shape[0], image_concat_x.shape[1]-cropped_y.shape[1], cropped_y.shape[2]]), axis=1 )
        image_concat_yx = np.concatenate( (image_concat_x, cropped_y), axis=2 )
        uncropped_image = image_concat_yx

    return uncropped_image

def pad_cropped_area(cropped_area, wanted_xy, pad_dim = 1):
    amount_to_add = wanted_xy - cropped_area.shape[pad_dim]
    append_image = []

    if pad_dim == 1:
        append_image = np.zeros( [cropped_area.shape[0], amount_to_add, cropped_area.shape[2] ] )
    elif pad_dim == 2:
        append_image = np.zeros( [cropped_area.shape[0], cropped_area.shape[1], amount_to_add ] )

   # append_image = np.moveaxis(append_image, 1, pad_dim)  ##generalizes so this works for x or y (moves amount to add to other axis). 
    padded_cropped_image = np.append( cropped_area, append_image , axis=pad_dim)
    return padded_cropped_image

def unpad_cropped_area(image, original_shape, padded_dim=1):
    image = move_channels(image) #rewrap images puts the image number as the last axis, woops.
    unpadded_image = None
    if padded_dim == 1:
        unpadded_image = image[:, 0:original_shape[1] , :]
    if padded_dim == 2:
        unpadded_image = image[:, :, 0:original_shape[2]]
    return unpadded_image

def extract_or_insert_dims(image):
    ##check for 3/4 dims
    if image.ndim == 4:
        image = image[:,:,:,0] ##TODO:: let user specify which channel to use. 

    ##check for single slice input, add dim for processing
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    return image

#figure out the cropping in x/y. there's overlap between the two.. hard to describe with words. check readme for details.
def get_cropped_regions(image, wanted_xy, y_cap, x_cap, dims):
    image_cropped_y = None
    image_cropped_x = None

    if( wanted_xy*x_cap == dims[1] ):
        pass
    else:
        image_cropped_x = image[:, (wanted_xy*x_cap):, 0:(wanted_xy*y_cap)]
    if( wanted_xy*y_cap == dims[2] ):
        pass
    else:
        image_cropped_y = image[:,0:(wanted_xy*x_cap), (wanted_xy*y_cap):]
    return image_cropped_x, image_cropped_y

##this moves the image channel
def move_channels(image):
    dims = image.shape
    if dims[0] > dims[2]:
        image = np.moveaxis(image, 2, 0)
    #if image.shape[0] > 1:
    #    image = image[:1,:,:]
    return image

#this crops the images so they can be split properly
def crop_image(image, wanted_xy, x_cap, y_cap):
    image = image[:, 0:int(wanted_xy*x_cap), 0:int(wanted_xy*y_cap)] ## note that these will be integers, but the casting is done because * returns a float (I                                                                     ## think)
    return image
    




