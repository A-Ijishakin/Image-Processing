from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter 
from scipy import spatial 
import numpy as np 
import matplotlib.pyplot as plt  

def surface_normals_np(triangles, verts):
    """ 
    An algorithm for computing surface normals of a triangulated surface.  
    
    INPUTS:
        triangles - An (n x 3) numpy.ndarray of indices corresponding to the vertices (in verts array), which composes each triangle.
        verts - An (n x 3) numpy.ndarray of the co-ordinates of vertices which make up triangles. 

    OUTPUT:
        verts_normls - The vertex normals of inputed vertices. 
        tri_normls - The face normals of the inputed triangles.
    """
    #create an array which will be used to store the face normals 
    tri_normls = np.ones((triangles.shape[0], triangles.shape[1])) 

    #loop through the triangles and compute face normals 
    for idx, i in enumerate(triangles):
        tri_normls[idx] = np.cross((verts[i[0]] - verts[i[1]]), (verts[i[2]]- verts[i[1]]))

    #normalise 
    for idx, i in enumerate(tri_normls):     
        tri_normls[idx] = i/np.linalg.norm(i) 

    #initialise an empty to array which will store vertex normals 
    verts_normls = np.array([]) 
        
    for idx, i in enumerate(verts): #for index element in vertices 
        indx_list = np.where(triangles == idx) #find the triangles which use this vertex 
        indx_list = list(zip(indx_list[0], indx_list[1])) #put all the indices into a list 
        
        normal = [] #intialise an empty array to store the normals of the relevant triangle 
        
        for j in indx_list:
            normal.append(tri_normls[j[0]]) #append the relevant normals 
        
        for idx, i in enumerate(normal): 
            normal[idx] = i/np.linalg.norm(i) #set their values to them divided by their magnitude
        
        mag_norm = np.add.reduce(normal) #add the normals together 
        magnitude = np.linalg.norm(mag_norm) #compute the magnitude of the resulting vector 
        mag_norm = mag_norm/magnitude #divide by their magnitude 
        
        #add the latest normal to the list 
        verts_normls = np.append(verts_normls, mag_norm)    
    
    #reshape the array so that it is of the correct size 
    verts_normls = np.reshape(verts_normls, (-1, 3))       
    return verts_normls, tri_normls 

if __name__ == '__main__':
    #load in the image 
    data = np.load('image.npy')  

    #compute, vertices, triangles, and normals with scipy 
    verts, triangles, sp_normls, values = marching_cubes(data, spacing = [2, 0.5, 0.5]) 

    #compute face and vertex normals with numpy 
    verts_normls, tri_normls = surface_normals_np(triangles, verts) 

    #CONDUCT FIRST COMPARISON: between vertex normals:
    #reshape so that the cosine simialrity can be computed 
    verts = np.reshape(verts_normls, (verts_normls.shape[0] * verts_normls.shape[1])) 
    normls = np.reshape(sp_normls, (sp_normls.shape[0] * sp_normls.shape[1]))  

    #compute the cosine similarity of the vertex normals computed by surface_normals_np and marching cubes
    difference = 1 - spatial.distance.cosine(verts, normls)  

    #the cosine similarity yielded a result of: 0.94, which demonstrates that they are highly similar. 

    #CONDUCT SECOND COMPARISON: between vertex and face normals

    #multiply the vertex and face normals by their transposes so that they are the same shape, and then flatten them to 
    #check the cosine similarity of them 
    vertex_norms, face_norms  = (verts_normls.T @ verts_normls).flatten(), (tri_normls.T @ tri_normls).flatten()

    #compute the cosine similarity of the vertex normals and face normals 
    difference2 = 1 - spatial.distance.cosine(vertex_norms, face_norms)  

    #the cosine similarity yielded a result of 0.99, demonstrating that they are highly similar, which makes sense given that 
    #they are derived from the same surface 

    #CONDUCT SMOOTHING EXPERIMENT 
    image = data 

    #normalise the image pixels to be between 0 and 255
    image = image /(np.max(image) / 255)    

    #set the image to an unsigned integer 8-bit type 
    image = image.astype('uint8') 

    #create a list of sigma values to loop through 
    sigma_range = np.arange(1, 9)  

    #initialise an empty list to store the magnitudes of the different normal vectors
    normalsMag = []

    #create a list for the first comparison (vertex to vertex normals differences)
    comp1 = []

    #create a list for the first comparison (vertex to face normals differences)
    comp2 = []

    for i in sigma_range:
        img = gaussian_filter(image, i) #smooth the image 
        verts, triangles, sp_normals, _ = marching_cubes(img, spacing = [0.2, 0.5, 0.5]) #compute faces, vertices and normals with scipy
        verts_normls, tri_normls = surface_normals_np(triangles, verts) #compute normals with numpy 

        #flatten the arrays so that cosine simialrity can be taken 
        spNormals, vertsNormls, triNormls = sp_normals.flatten(), verts_normls.flatten(), tri_normls.flatten()

        #calculate the magnitude of the vectors 
        normalsMag.append((np.linalg.norm(spNormals), np.linalg.norm(vertsNormls), np.linalg.norm(triNormls))) 

        #comparison 1
        diff = 1 - spatial.distance.cosine(vertsNormls, spNormals)
        comp1.append(diff) 

        #multiply the vertex and face normals by their transposes so that they are the same shape, and then flatten them to 
        #check the cosine similarity of them 
        verts_normls, tri_normls  = (verts_normls.T @ verts_normls).flatten(), (tri_normls.T @ tri_normls).flatten()

        #compute the cosine similarity of the vertex normals and face normals 
        diff2 = 1 - spatial.distance.cosine(verts_normls, tri_normls)  
        comp2.append(diff2)
        
#As values of sigma increase the magnitude of the vectors which describe the three respsective normals decreases up until when 
#sigma is = 6, at which points the magnitude starts to increase. Also higher values of sigma creates a fluctuation in the similarity 
#of vertex normals computed by the numpy and scipy implementation, at certain values (2, 8) they are exactly equal, whilst in between these values
#the siilarity decreases.  A similar trend is mimicked in the comparison experient between vertex and face normals, except the similarity remains 
#very high the whole time. 
















