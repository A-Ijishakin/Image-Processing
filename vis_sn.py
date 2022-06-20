import matplotlib.pyplot as plt 
from skimage.measure import marching_cubes
import numpy as np 
from surf_norms import surface_normals_np 

#load in the data 
data = np.load('image.npy') 

#compute vertices, triangles and vertex normals 
verts, triangles, scipy_normals, values = marching_cubes(data) 

#compute vertex and face normals 
numpy_normls, numpy_Fnormls = surface_normals_np(triangles, verts) 

#VISUALISE THE NUMPY VERTEX NORMALS:
fig = plt.figure() 
ax = fig.gca(projection='3d')
ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles = triangles, color='b')  
ax.quiver(verts[:, 0], verts[:, 1], verts[:, 2], numpy_normls[:, 0], numpy_normls[:, 1], 
          numpy_normls[:, 2], 0.5, color = 'red', length = 3, alpha = 0.1, normalize = True)
#save 
fig.savefig('Numpy_vertex_normals.png') 

#VISUALISE THE SCIPY VERTEX NORMALS:
fig = plt.figure() 
ax = fig.gca(projection='3d')
ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles = triangles, color='b')  
ax.quiver(verts[:, 0], verts[:, 1], verts[:, 2], scipy_normals[:, 0], scipy_normals[:, 1], 
          scipy_normals[:, 2], 0.5, color = 'red', length = 3, alpha = 0.1, normalize = True)
#save 
fig.savefig('Scipy_vertex_normals.png') 


#VISUALISE THE FACE NORMALS: 
#initialise an empty array to store the centres of triangles 
centre = np.ones((triangles.shape[0], triangles.shape[1]))

#calculate triangle centres 
for idx, i in enumerate(triangles):
    centre[idx, 0]  = (verts[i[0], 0] + verts[i[1], 0] + verts[i[2], 0] ) / 3 
    centre[idx, 1] = (verts[i[0], 1] + verts[i[1], 1] + verts[i[2], 1]) / 3 
    centre[idx, 2] = (verts[i[0], 2] + verts[i[1], 2] + verts[i[2], 2] ) / 3   

#plot the face normals 
fig = plt.figure() 
ax = fig.gca(projection='3d')
ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles = triangles, color='b')  
ax.quiver(centre[:, 0], centre[:, 1], centre[:, 2], numpy_Fnormls[:, 0], numpy_Fnormls[:, 1], 
          numpy_Fnormls[:, 2], color = 'red', length = 2, alpha = 0.1)
#save 
fig.savefig('Face_normals.png') 