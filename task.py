from scipy.interpolate import interpn 
import numpy as np 
import matplotlib.pyplot as plt 

class Image3D:
    """ 
    Image3D
    A class which takes as input a 3d image which will have a spatial transformation applied to it.   
    """
    def __init__(self, Image): 
        """ 
    A class constructor function to intialise the Image3D object.    
    
    INPUTS:
        Image - The 3D image which will be operated upon. 
    
    OUTPUTS:
        self.Image - The image following, image precomputation. 
        self.image_shape - The shape of the image. 
        self.coordinates - The image co-ordinates. 
    """ 
        #conduct image precomputation to orient it properly. By transposing the image and swapping the axes the co-ordinate system
        # becomes X, Y, Z. Which is a much more intuitive approach.  
        self.Image = Image.T
        self.Image = np.swapaxes(self.Image, 1, 0)  

        #get the image shape  
        self.image_shape = self.Image.shape   
         
        #Normalise the image to be between 0 and 1 
        self.Image = (self.Image - np.min(self.Image))/(np.max(self.Image) - np.min(self.Image))
        
        #create the X, Y and Z meshgrid to create the coordinates this creates grids which will represent the co-ordinate for each pixel.
        XX, YY, ZZ = np.meshgrid(np.arange(0, self.image_shape[0]), np.arange(0, self.image_shape[1]), 
                                         np.arange(0, self.image_shape[2]), indexing = 'ij')

        #create the image coordinates, by concatentating the meshgrids and transposing it. By doing this the co-ordinates can then be multiplied by the transformation matrix.
        self.coordinates = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1), ZZ.reshape(-1, 1), np.ones((np.prod(self.image_shape), 1))), axis = 1).T
        
    def warp(self, TransformM):
        """ 
        warp 

        A function which computes a spatial transformation on an image. 
        
        INPUTS:
            TransformM - A transformation matrix (numpy.ndarray) which will be applied to image co-ordinates. 

        OUTPUT:
            interpImage - The transformed image following the operations of the function. 
        """
        
        #set the conditions for interpolation
        loci_i, loci_j, loci_k = np.arange(0, self.image_shape[0]), np.arange(0, self.image_shape[1]), np.arange(0, self.image_shape[2])
        loci = (loci_i, loci_j, loci_k)  
        
        
        #set the new coordinates to be intepolated 
        new_coords = np.dot(TransformM, self.coordinates)[:3].T   
    
        
        #interpolate 
        interpImage = interpn(loci, self.Image, new_coords, bounds_error = False, fill_value = 0)   
        interpImage  = interpImage.reshape(self.image_shape)  
        
        return interpImage  
    
    
class AffineTransform:
    """ 
    AffineTransform 

    A class which allows for the computation of a suitable transformation matrix. 
    """ 
    def __init__(self, trans_param, image_shape):
        """
        A class constructor function which allows for the intialisation of an AffineTransform object. 

        INPUTS:
            trans_param - A list of transformation parameters. 
            image_shape - The shape of the image being transformed. 
        
        OUPUTS:
            self.trans_param - The transformation paramaters. 
            self.vec_length - The length of the transformation parameter list. 
            self.image_shape - The shape of the image. 
            self.centreM - A matrix used to centre an image under particular interpolation conditions.
            self.centreMinv - A matrix used to centre an image under particular interpolation conditions.
        """  

        #store the transformation parameters 
        self.trans_param = trans_param 
        
        #set trans parameters to something of length 12 as not to activate a ValueError. 
        if self.trans_param == None:
            self.trans_param = np.zeros(12) 
          
        #set the appropriate lengths for the vecotr  
        self.params = [6, 7, 12]  

        #get the length of the transformation parameters list   
        self.vec_length = len(self.trans_param) 

        #set the image shape appropriately  
        self.image_shape = image_shape[1], image_shape[0], image_shape[2]   
        
        #check if the transformation parameters list is of the correct size
        if self.params.count(self.vec_length) == 0:
            raise ValueError('The input vector may only contain, 6, 7 or 12 DoFs') 

        #set the centre of the image appropriately  
        self.centre = np.array([(i-1)/2 for i in self.image_shape]) 

        #create the cetering matrices        
        self.centreM = np.array([[1, 0, 0, 0],   
                             [0, 1, 0, 0],      
                             [0, 0, 1, 0],
                             [-self.centre[0], -self.centre[2], -self.centre[1], 1]]) 
                              
        self.centreM_inv = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [self.centre[0], self.centre[2], self.centre[1], 1]]) 
        
        #transpose them and change their dtype's
        self.centreM_inv, self.centreM = (self.centreM_inv.T).astype('float32'), self.centreM.T.astype('float32') 
           
              
    def rigid_transform(self):
        """ 
        rigid_transform

        A function which computes a rigid transformation matrix 

        INPUT:
            self - used to access self.trans_parameters to input into the matrix. 
        
        OUTPUT:
            Rtrans_M - Transformation matrix to compute rigid transformations. 
        """ 

        #compute the transformation matrix using the transformation parameters 
        Rtrans_M = np.array([[(np.cos(self.trans_param[2])*np.cos(self.trans_param[1])), (np.cos(self.trans_param[2])*np.sin(self.trans_param[1])*np.sin(self.trans_param[0]))
              - (np.sin(self.trans_param[2])*np.cos(self.trans_param[0])), 
         (np.cos(self.trans_param[2])*np.sin(self.trans_param[1])*np.cos(self.trans_param[0])) + (np.sin(self.trans_param[2]) * np.sin(self.trans_param[0])), (self.trans_param[3])],

          [(np.sin(self.trans_param[2])*np.cos(self.trans_param[1])), np.sin(self.trans_param[2])*np.sin(self.trans_param[1] )*np.sin(self.trans_param[0]) + np.cos(self.trans_param[2])*np.cos(self.trans_param[0]), 
          (np.sin(self.trans_param[2])*np.sin(self.trans_param[1]))*np.cos(self.trans_param[0]) - (np.cos(self.trans_param[2])*np.sin(self.trans_param[0])), (self.trans_param[4])], 

         [np.sin(self.trans_param[1]), np.cos(self.trans_param[1])*np.sin(self.trans_param[0]), (np.cos(self.trans_param[1])*np.cos(self.trans_param[0])), self.trans_param[5]], 

         [0, 0, 0, 1]]).astype('float32')   
        
        #if there is a seventh entry then use it to scale the transformation 
        if self.vec_length == 7:  
            Scale_M = np.eye(4, 4)
            Scale_M[0, 0] = self.trans_param[6]
            Scale_M[1, 1] = self.trans_param[6]
            Scale_M[2, 2] = self.trans_param[6]
            Scale_M[3, 3] = self.trans_param[6] 
            
            #compute the appropriate matrix 
            Rtrans_M = np.dot(Scale_M, Rtrans_M).astype('float32')  

        #check whether a rotation is occuring or not     
        for i in self.trans_param[:3]: 
            if i != 0:
                #if so, centre the transformation appropriately. 
                Rtrans_M = np.dot(self.centreM_inv, np.dot(Rtrans_M, self.centreM))
                break 

        return Rtrans_M  

    def affine_transform(self): 
        """ 
        affine_transform

        A function which computes an affine transformation matrix 

        INPUT:
            self - used to access self.trans_parameters to input into the matrix. 
        
        OUTPUT:
            Atrans_M - Transformation matrix to compute rigid transformations. 
        """ 
       
       #compute the affine transformation matrix based on the transformation parameters.
        Atrans_M = np.array([[(np.cos(self.trans_param[2])*np.cos(self.trans_param[1])) * (self.trans_param[6] + (1 if self.trans_param[6] == 0 else 0)), (np.cos(self.trans_param[2])*np.sin(self.trans_param[1])*np.sin(self.trans_param[0]))
                      - (np.sin(self.trans_param[2] if self.trans_param[9][0] == 0 else np.pi/2)*np.cos(self.trans_param[0]))*(self.trans_param[9][0] + (1 if self.trans_param[9][0] == 0 else 0)), 
                 (np.cos(self.trans_param[2])*np.sin(self.trans_param[1])*np.cos(self.trans_param[0])) + (np.sin(self.trans_param[2] if self.trans_param[11][0] == 0 else np.pi/2) * np.sin(self.trans_param[0] if self.trans_param[11][0] == 0 else np.pi/2)) * (self.trans_param[11][0] + (1 if self.trans_param[11][0] == 0 else 0)), (self.trans_param[3])],
                         
                  [(np.sin(self.trans_param[2] if self.trans_param[10][0] == 0 else np.pi/2)*np.cos(self.trans_param[1]) )*(self.trans_param[10][0] + (1 if self.trans_param[10][0]==0 else 0)), np.sin(self.trans_param[2] )*np.sin(self.trans_param[1])*np.sin(self.trans_param[0]) + np.cos(self.trans_param[2] )*np.cos(self.trans_param[0] ) * (self.trans_param[7] + (1 if self.trans_param[7] == 0 else 0)), 
                  (np.sin(self.trans_param[2])*np.sin(self.trans_param[1]))*np.cos(self.trans_param[0]) - (np.cos(self.trans_param[2])*np.sin(self.trans_param[0] if self.trans_param[11][0] == 0 else np.pi/2)) * (self.trans_param[11][1] + (1 if self.trans_param[11][1] == 0 else 0)), (self.trans_param[4])], 

                 [-np.sin(self.trans_param[1] if self.trans_param[10][1] == 0 else np.pi/2)*(self.trans_param[10][1] + (1 if self.trans_param[10][1] == 0 else 0)), np.cos(self.trans_param[1])*np.sin(self.trans_param[0] if self.trans_param[9][1] == 0 else np.pi/2) * (self.trans_param[9][1] + (1 if self.trans_param[9][1] == 0 else 0 )), (np.cos(self.trans_param[1])*np.cos(self.trans_param[0])) *  (self.trans_param[8] + (1 if self.trans_param[8] == 0 else 0)), self.trans_param[5]], 

                 [0, 0, 0, 1]]).astype('float32')  
        
        #check whether a rotation is occuring or not 
        for i in self.trans_param[:3]: 
            if i != 0:
                #if so centre the image appropriately  
                Atrans_M = self.centreM_inv @ (Atrans_M @ self.centreM)
                break 
      
        return Atrans_M
                    
    def random_transform_generator(self, strength):
        """
        random_transform_generator

        A function which computes a random transformation matrix. It works by by starting with a list of length twelve with a random amount of 
        zeros and the 'strength' parameter. That list is used to decide which DoF are used as the DoF's who's corresponding index is zero will be cancelled out.
        Next the function randomly creates transformation parameters and multiplies them by the decision list. These are then passed into the random
        transformation generation matrix at appropriate indices.

        INPUT:
            self - used to access self variables.
        
        OUTPUT:
            rand_AtransM - The random transformation matrix. 
        """
        #create a list with a random amount of ones and zeros, 
        #that will be used decide which transformations take place
        randD = np.random.choice([0, strength], size = (12,)) 
        
        #randomise the values of alpha, beta and gamma to be used in the list                                                                                
        self.alpha = np.random.randint(360) * randD[0] 
        self.beta = np.random.randint(360) * randD[1]
        self.gamma = np.random.randint(360) * randD[2]
        
        #randomise translations that may or may not be used                                                                                
        trans_x = np.random.randint(-4, 4) * randD[3] 
        trans_y = np.random.randint(-4, 4) * randD[4]
        trans_z = np.random.randint(-4, 4) * randD[5] 
        
        #randomise a shear that may or may not be used                                                                                                    
        shear_x = np.random.rand(2) * randD[6]
        shear_y = np.random.rand(2) * randD[7]    
        shear_z = np.random.rand(2) * randD[8]
        
        #randomise a scaling that may or not be used                                                                                 
        scale_x = np.random.randn() * randD[9]
        scale_y = np.random.randn() * randD[10]
        scale_z = np.random.randn() * randD[11]
         
        
        #create the randomised transformation matrix
        rand_AtransM = np.array([[(np.cos(np.radians(self.alpha))*np.cos(np.radians(self.beta))) * (scale_x + (1 if scale_x == 0 else 0)), (np.cos(np.radians(self.alpha))*np.sin(np.radians(self.beta))*np.sin(np.radians(self.gamma)))
                      - (np.sin(np.radians(self.alpha) if shear_x[0] == 0 else np.pi/2)*np.cos(np.radians(self.gamma)))*(shear_x[0] + (1 if shear_x[0] == 0 else 0)), 
                 (np.cos(np.radians(self.alpha))*np.sin(np.radians(self.beta))*np.cos(np.radians(self.gamma))) + (np.sin(np.radians(self.alpha) if shear_z[0] == 0 else np.pi/2) * np.sin(np.radians(self.gamma) if shear_z[0] == 0 else np.pi/2)) * (shear_z[0] + (1 if shear_z[0] == 0 else 0)), (trans_x)],
                         
                  [(np.sin(np.radians(self.alpha) if shear_y[0] == 0 else np.pi/2)*np.cos(np.radians(self.beta)) )*(shear_y[0] + (1 if shear_y[0]==0 else 0)), np.sin(np.radians(self.alpha) )*np.sin(np.radians(self.beta))*np.sin(np.radians(self.gamma) ) + np.cos(np.radians(self.alpha) )*np.cos(np.radians(self.gamma) ) * (scale_y + (1 if scale_y == 0 else 0)), 
                  (np.sin(np.radians(self.alpha))*np.sin(np.radians(self.beta)))*np.cos(np.radians(self.gamma)) - (np.cos(np.radians(self.alpha))*np.sin(np.radians(self.gamma) if shear_z[0] == 0 else np.pi/2)) * (shear_z[1] + (1 if shear_z[1] == 0 else 0)), (trans_y)], 

                 [-np.sin(np.radians(self.beta) if shear_y[1] == 0 else np.pi/2)*shear_y[1], np.cos(np.radians(self.beta))*np.sin(np.radians(self.gamma) if shear_x[1] == 0 else np.pi/2) * (shear_x[1] + (1 if shear_x[1] == 0 else 0 )), (np.cos(np.radians(self.beta))*np.cos(np.radians(self.gamma))) *  (scale_z + (1 if scale_z == 0 else 0)), trans_z], 

                 [0, 0, 0, 1]])
        
        #check if a rotation is used 
        for i in [self.alpha, self.beta, self.gamma]: 
            if i != 0:
                #if so centre appropriately 
                rand_AtransM = self.centreM_inv @ (rand_AtransM @ self.centreM)
                break 
        
        return rand_AtransM            
                                               

#IMPLEMENTATION FOR THE VISUALISATION 

#load in the data  
img3D = np.load('image_tf.npy')

#instatiate an Image3d Class 
Data = Image3D(img3D) 

#instatiate an array to store the images within 
Vis_imgs = np.ones((100, 128, 128))

# create a rigid transformation matrix for a 45 degree rotation
rMatrix = AffineTransform([0, 0, np.radians(45), 0, 0, 0], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[0, :, :] = WarpedImg[:, :, 15] 

# create a rigid transformation matrix for a 68 degree rotation
rMatrix = AffineTransform([0, 0, np.radians(45), 0, 0, 0], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[1, :, :] = WarpedImg[:, :, 15] 

#create a rigid transformation matrix for a 30mm translation on the y_axis 
rMatrix = AffineTransform([0, 0, 0, 30, 0, 0], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[2, :, :] = WarpedImg[:, :, 15] 

#create a rigid transformation matrix for a 30mm translation on the x_axis 
rMatrix = AffineTransform([0, 0, 0, 0, 30, 0], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[3, :, :] = WarpedImg[:, :, 15] 

#create a rigid transformation matrix for a 0.5 scaling (across z) and a translation on the x_axis
rMatrix = AffineTransform([0, 0, 0, 0, -20, 0, 2], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[4, :, :] = WarpedImg[:, :, 15]  

#create a rigid transformation matrix for a 0.5 scaling (across z) and a translation on the y_axis
rMatrix = AffineTransform([0, 0, 0, -20, 0, 0, 2], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[5, :, :] = WarpedImg[:, :, 15]  


#create a rigid transformation matrix for a -30mm translation on the y_axis and 68 degree rotation
rMatrix = AffineTransform([0, 0, np.radians(68), -30, 0, 0], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[6, :, :] = WarpedImg[:, :, 15]   

#create a rigid transformation matrix for a -30mm translation on the y_axis and 45 degree rotation
rMatrix = AffineTransform([0, 0, np.radians(45), -30, 0, 0], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[7, :, :] = WarpedImg[:, :, 15]  

#create a rigid transformation matrix for a 90 degree rotation about the z-axis 30mm translation on the x_axis
rMatrix = AffineTransform([0, 0, np.radians(90), -30, 0, 0], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[8, :, :] = WarpedImg[:, :, 15]   

#create a rigid transformation matrix for a 140 degree rotation about the z-axis and a translation on the x_axis
rMatrix = AffineTransform([0, 0, np.radians(140), -15, 0, 0], img3D.shape) 
Matrix = rMatrix.rigid_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[9, :, :] = WarpedImg[:, :, 15] 

#create an affine transformation matrix for 45 degree rotation about z, and a 0.5 scaling in both the x and y directions
Amatrix = AffineTransform([0, 0, np.radians(45), 0, 0, 0, 2, 2, 0, [0, 0], [0, 0], [0, 0]], img3D.shape) 
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[10, :, :] = WarpedImg[:, :, 15]   

#create an affine transformation matrix for 68 degree rotation about z, and a scaling in both the x and y directions
Amatrix = AffineTransform([0, 0, np.radians(68), 0, 0, 0, 3, 3, 0, [0, 0], [0, 0], [0, 0]], img3D.shape) 
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[11, :, :] = WarpedImg[:, :, 15]  

#create an affine transformation matrix for a translation and a shear in y 
Amatrix = AffineTransform([0, 0, 0, 0, 0, -30, 0, 0, 0, [0.5, 0.5], [0, 0], [0, 0]], img3D.shape)
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[12, :, :] = WarpedImg[:, :, 15]   

#create an affine transformation matrix for a translation in x, y and a shear in y 
Amatrix = AffineTransform([0, 0, 0, 0, -30, -30, 0, 0, 0, [0.5, 0.5], [0, 0], [0, 0]], img3D.shape)
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[13, :, :] = WarpedImg[:, :, 15]  

#create an affine transformation matrix for a translation a and a shear in z and translation in x  
Amatrix = AffineTransform([0, 0, 0, 0, 0, 0, 0.5, 0, 0, [0, 0], [0, 0], [2, 2]], img3D.shape)
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[14, :, :] = WarpedImg[:, :, 15] 

#create an affine transformation matrix for a translation a and a shear in z and translation in x, y
Amatrix = AffineTransform([0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, [0, 0], [0, 0], [2, 2]], img3D.shape)
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[15, :, :] = WarpedImg[:, :, 15] 

#create an affine transformation matrix for a rotation of 68 degrees about y and a shear in y 
Amatrix = AffineTransform([0, np.radians(45), 0, 0, 0, 0, 0, 0, 0, [0.5, 0.5], [0, 0], [0, 0]], img3D.shape)
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[16, :, :] = WarpedImg[:, :, 15]  

#create an affine transformation matrix for a rotation of 68 degrees about y and a shear in z 
Amatrix = AffineTransform([0, np.radians(68), 0, 0, 0, 0, 0, 0, 0, [0, 0], [0, 0], [0.5, 0.5]], img3D.shape)
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[17, :, :] = WarpedImg[:, :, 15]  

#create an affine transformation matrix for a rotation of 45 degrees about z and x and a shear in y and z, scaling in y and a translation
Amatrix = AffineTransform([np.radians(45), 0, np.radians(45), 0, 0, -30, 0, 0.25, 0, [0.5, 0.5], [0, 0], [0.5, 0.5]], img3D.shape)
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[18, :, :] = WarpedImg[:, :, 15] 

#create an affine transformation matrix for a rotation of 150 degrees about z and x and a shear in y and z, scaling in y and a translation
Amatrix = AffineTransform([np.radians(45), 0, np.radians(150), 0, 0, -30, 0, 0.25, 0, [0.5, 0.5], [0, 0], [0.5, 0.5]], img3D.shape)
Matrix = Amatrix.affine_transform() 
WarpedImg = Data.warp(Matrix)
Vis_imgs[19, :, :] = WarpedImg[:, :, 15] 

#used to continue iterating through 
Vis_slice = 19

for i in range(10):
    #generate a random transformation matrix 
    randMatrix = AffineTransform(None, img3D.shape)
    matrix = randMatrix.random_transform_generator(1)
    #warp 
    warped = Data.warp(matrix)
    for j in range(26, 31):
        Vis_slice += 1
        #store the image 
        Vis_imgs[Vis_slice, :, :] = warped[:, :, j] 

strength = np.arange(1, 6) 

for str in strength:
    #create a random transformation matrix 
    randMatrix = AffineTransform(None, img3D.shape)
    matrix = randMatrix.random_transform_generator(str)
    #warp 
    warped = Data.warp(matrix)
    Vis_slice += 1
    #store the image 
    Vis_imgs[Vis_slice, :, :] = warped[:, :, 30]  
