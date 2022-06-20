import numpy as np   
from scipy import ndimage 
from time import time  
from PIL import Image as img

#load the image 
data = np.load('image.npy') 

def distance_transform_np(binary_image, voxel_dimensions):
    """ 
    An algorithm for computing the euclidean distance transform of a 3D binary image. First an array to store the euclidean distances is computed. 
    Then the indices where the binary image has a value of 1 (the foreground) are computed and placed these indices into a 3D array. The indices 
    where the image has a value of 0 (the background) are also found and split into x, y and z arrays. The algorithm then loops through the indices 
    of the foreground pixels, and creates x, y and z arrays for the size of the background arrays for each individual foreground pixel. The euclidean 
    distance of the foreground pixel from every other background pixel is then computed and the minimum value becomes the pixel value in the array 
    containing the euclidean distances at the index corresponding to the foreground pixel of the binary image.  
    
    INPUTS:
        binary_image - A binary image that will be euclidean distance transformed. 
        voxel_dimensions - The voxel dimensions of the binary image. 

    OUTPUT:
        EDT - The euclidean distance transform of the binary image. 

    """
    #create an empty array in which to store the EDT values. 
    EDT = np.zeros((voxel_dimensions[0], voxel_dimensions[1], voxel_dimensions[2])) 
    
    #find the foreground pixels 
    foreground = np.where(binary_image == 1) 
    foreground = list(zip(foreground[0], foreground[1], foreground[2]))  

    #find the background pixels 
    background = np.where(binary_image == 0) 
    background_z = np.array(background[0]) 
    background_x = np.array(background[1]) 
    background_y = np.array(background[2]) 

    #loop thorugh the foreground pixels 
    for voxel in (foreground):
        #create a arrays filled with the foreground pixel value but the same shape as the background arrays 
        foreground_z = voxel[0] * np.ones(len(background_z)) 
        foreground_x = voxel[1] * np.ones(len(background_x)) 
        foreground_y = voxel[2] * np.ones(len(background_y)) 

        #compute the EDT 
        distances = ((foreground_x - background_x) ** 2) + ((foreground_y - background_y) ** 2) + ((foreground_z - background_z) ** 2) ** 0.5

        #store the EDT in the corresponding index 
        EDT[voxel] = np.min(distances)

    return EDT           

#scipy implementation of EDT 
start = time() #begin the timing
scipy_EDT = ndimage.distance_transform_edt(data, [32, 128, 128])  
end = time() #end the timing 
result = end - start 

#numpy implementation of EDT 
start = time() 
np_EDT = distance_transform_np(data, [32, 128, 128]) 
end = time() 
result2 = end - start 

#compute the difference in time 
timeDiff = result2 - result 

#the time difference is 46.22 seconds long, the scipy implementation takes 0.07 seconds, whereas the numpy implementation takes
# 46.29 seconds. 

#compute the pixel wise difference between the two approaches 
pixel_differences = scipy_EDT - np_EDT 

#compute the mean and standard deviation of the pixel differences. 
meanPixeldiff, stdPixeldiff = np.mean(pixel_differences), np.std(pixel_differences)

#the mean pixel differnece is 3.57 and the standard deviation of the pixel differences is 21.9, which shows that the two implementations
#are quite similar. 

#save the images as PNG files 
for i in range(9, 14):
    img.fromarray((data[i, :, :]/(np.max(data[i, :, :])/255)).astype('uint8')).save('original image slice ' + str(i) +'.png')
    img.fromarray((np_EDT[i, :, :]/(np.max(np_EDT[i, :, :])/255)).astype('uint8')).save('numpy EDT slice ' + str(i) +'.png')
    img.fromarray((scipy_EDT[i, :, :]/(np.max(scipy_EDT[i, :, :])/255)).astype('uint8')).save('scipy EDT slice ' + str(i) +'.png')

 