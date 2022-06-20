from task import Vis_imgs 
from PIL import Image as img 
import numpy as np 

#create a list with the appropriate names for the rigid and affine transformation photos 
names_of_translations = ['rigid transformation (45 degree rotation)', 
                        'rigid transformation  (68 degree rotation)', 'rigid transformation  (30mm translation on the y_axis)',
                        'rigid transformation  30mm translation on the x_axis', 'rigid transformation  (0.5 scaling (across z) and a translation on the x_axis)',
                         'rigid transformation  (0.5 scaling (across z) and a translation on the y_axis)', 'rigid transformation  (-30mm translation on the y_axis and 68 degree rotation)', 'rigid transformation  (-30mm translation on the y_axis and 45 degree rotation)', 
                         'rigid transformation  (90 degree rotation about the z-axis 30mm translation on the x_axis)', 'rigid transformation  (140 degree rotation about the z-axis and a translation on the x_axis)', 'affine transformation  (45 degree rotation about z, and a 0.5 scaling in both the x and y directions)', 
                         'affine transformation  (68 degree rotation about z, and a scaling in both the x and y directions)',
                         'affine transformation  (translation and a shear in y)', 'affine transformation  (translation in x, y and a shear in y )', 
                         'affine transformation  (shear in z and translation in x)',
                         'affine transformation  (translation and a shear in y and translation in x)',
                         'affine transformation  (rotation of 68 degrees about y and a shear in y)', 
                         'affine transformation  (rotation of 68 degrees about y and a shear in z)', 
                         'affine transformation  (rotation of 45 degrees about z and x and a shear in y and z, scaling in y and a translation)',
                         'affine transformation  (rotation of 150 degrees about z and x and a shear in y and z, scaling in y and a translation)']

#save images for the rigid and affine transformations
for i in range(0, 20): 
    img.fromarray((Vis_imgs[i, :, :]/(np.max(Vis_imgs[i, :, :])/255)).astype('uint8')).save(str(names_of_translations[i]) + '.png') 

Vis_slice = 19

#save images for the Randomly generated images 
for i in range(10):   
    for j in range(26, 31):
        Vis_slice += 1
        img.fromarray((Vis_imgs[Vis_slice, :, :]/(np.max(Vis_imgs[Vis_slice, :, :])/255)).astype('uint8')).save('random transformation ' + str(i) + 'slice ' + str(j) + '.png') 

strength = np.arange(1, 6) 

#save images for the Randomly generated images of different strength 
for s in strength:
    Vis_slice += 1
    img.fromarray((Vis_imgs[Vis_slice, :, :]/(np.max(Vis_imgs[Vis_slice, :, :])/255)).astype('uint8')).save('random transformation with a strength parameter of ' + str(s) + '.png') 
