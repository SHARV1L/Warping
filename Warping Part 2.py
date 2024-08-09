#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install opencv-python


# In[96]:


import numpy as np
import pickle
import time
import sys
import cv2
from scipy.signal import medfilt
from scipy.ndimage.filters import maximum_filter as maxfilt

def PointCloud2Image(M,Sets3DRGB,viewport,filter_size):

    # setting yp output image
    print("...Initializing 2D image...")
    top = viewport[0]
    left = viewport[1]
    h = viewport[2]
    w = viewport[3]
    bot = top  + h + 1
    right = left + w +1;
    output_image = np.zeros((h+1,w+1,3));    

    for counter in range(len(Sets3DRGB)):
        print("...Projecting point cloud into image plane...")

        # clear drawing area of current layer
        canvas = np.zeros((bot,right,3))

        # segregate 3D points from color
        dataset = Sets3DRGB[counter]
        P3D = dataset[:3,:]
        color = (dataset[3:6,:]).T
        
        # form homogeneous 3D points (4xN)
        len_P = len(P3D[1])
        ones = np.ones((1,len_P))
        X = np.concatenate((P3D, ones))

        # apply (3x4) projection matrix
        x = np.matmul(M,X)

        # normalize by 3rd homogeneous coordinate
        x = np.around(np.divide(x, np.array([x[2,:],x[2,:],x[2,:]])))

        # truncate image coordinates
        x[:2,:] = np.floor(x[:2,:])

        # determine indices to image points within crop area
        i1 = x[1,:] > top
        i2 = x[0,:] > left
        i3 = x[1,:] < bot
        i4 = x[0,:] < right
        ix = np.logical_and(i1, np.logical_and(i2, np.logical_and(i3, i4)))

        # make reduced copies of image points and cooresponding color
        rx = x[:,ix]
        rcolor = color[ix,:]

        for i in range(len(rx[0])):
            canvas[int(rx[1,i]),int(rx[0,i]),:] = rcolor[i,:]

        # crop canvas to desired output size
        cropped_canvas = canvas[top:top+h+1,left:left+w+1]

        # filter individual color channels
        shape = cropped_canvas.shape
        filtered_cropped_canvas = np.zeros(shape)
        print("...Running 2D filters...")
        for i in range(3):
            # max filter
            filtered_cropped_canvas[:,:,i] = maxfilt(cropped_canvas[:,:,i],5)

        
        # get indices of pixel drawn in the current canvas
        drawn_pixels = np.sum(filtered_cropped_canvas,2)
        idx = drawn_pixels != 0
        shape = idx.shape
        shape = (shape[0],shape[1],3)
        idxx = np.zeros(shape,dtype=bool)

        # make a 3-channel copy of the indices
        idxx[:,:,0] = idx
        idxx[:,:,1] = idx
        idxx[:,:,2] = idx

        # erase canvas drawn pixels from the output image
        output_image[idxx] = 0

        #sum current canvas on top of output image
        output_image = output_image + filtered_cropped_canvas

    print("Done")
    return output_image



# Sample use of PointCloud2Image(...)
# The following variables are contained in the provided data file:
#       BackgroundPointCloudRGB,ForegroundPointCloudRGB,K,crop_region,filter_size
# None of these variables needs to be modified



# In[139]:


# load variables: BackgroundPointCloudRGB,ForegroundPointCloudRGB,K,crop_region,filter_size)
def SampleCameraPath():
    # load object file to retrieve data
    file_p = open("data.obj",'rb')
    camera_objs = pickle.load(file_p)

#######  
    # radius of the semicircle
    r = 2

    # distance from the foreground object
    d = 2

    # number of steps in the path
    n_steps = 50

    # create camera positions
    angles = np.linspace(0, np.pi, n_steps+1)[:-1]
    x = d + r*np.cos(angles)
    y = r*np.sin(angles)
    z = np.zeros(n_steps)
    camera_positions = np.vstack((x, y, z)).T

    # create camera orientations
    # initial orientation is looking towards the object
    initial_direction = np.array([0, 0, 1])
    initial_up = np.array([0, 1, 0])
    R = np.vstack((initial_direction, initial_up, np.zeros(3))).T
    camera_orientations = np.zeros((n_steps, 3, 3))
    for i in range(n_steps):
        # rotate the direction vector by an angle of pi/n_steps
        angle = np.pi/n_steps
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]]) @ R
        camera_orientations[i] = R
        print(camera_orientations)

    projection_matrices = np.zeros((n_steps, 3, 4))
######    

    # extract objects from object array
    crop_region = camera_objs[0].flatten()
    filter_size = camera_objs[1].flatten()
    #K = camera_objs[2]
######
    # create projection matrices for each camera position and orientation
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # original camera matrix provided in data.mat
    
    
    ForegroundPointCloudRGB = camera_objs[3]
    BackgroundPointCloudRGB = camera_objs[4]

    # create variables for computation
    data3DC = (BackgroundPointCloudRGB,ForegroundPointCloudRGB)

######   
    for i in range(n_steps):
        R = camera_orientations[i]
#     R = np.identity(3)
    move = np.array([0, 0, -0.25]).reshape((3,1))

    for step in range(8):
        tic = time.time()
        
        fname = "SampleOutput{}.jpg".format(step)
        print("\nGenerating {}".format(fname))
        
#######        
    for i in range(n_steps):
        t = -R @ camera_positions[i]
        projection_matrices[i] = K @ np.hstack((R, t.reshape(3, 1)))  
#         t = step*move

        img = PointCloud2Image(M,data3DC,crop_region,filter_size)

        # Convert image values form (0-1) to (0-255) and cahnge type from float64 to float32
        img = 255*(np.array(img, dtype=np.float32))

        # convert image from RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # write image to file 'fname'
        cv2.imwrite(fname,img_bgr)

        toc = time.time()
        toc = toc-tic
        print("{0:.4g} s".format(toc))


# In[ ]:





# In[154]:


def main():
    SampleCameraPath()
    
######    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30, (640, 400))

    # load the foreground image
    foreground_image = cv2.imread('0004.jpg')
    # print(foreground_image)

    # loop over the projection matrices
    for i in range(n_steps):
    # project the foreground image onto the background
        M = projection_matrices[i]


        # Define output image size
    image_size = (480, 640)
    viewport = (0, 0, *image_size)
    #filter_size = (3,4)

    # Loop over camera positions
    for i in range(n_frames):
        # Define camera pose (combination of intrinsics and extrinsics)
        Rt = np.column_stack((R, t + positions[i].reshape(3, 1)))
        P = K @ Rt

        # Generate image from point cloud
    #     image = PointCloud2Image(foreground_image, P, image_size, viewport)
        output_image = PointCloud2Image(foreground_image, P, viewport, camera_objs[0])

        # Add background to output image
    #     output_image = cv2.addWeighted(image, 1, background_image, 1, 0)

    #     # Display image
    #     plt.imshow(output_image)
    #     plt.axis('off')
    #     plt.show()

    # write the output image to the video file
        out.write(output_image)

    # release the VideoWriter object
    out.release()
########

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[43]:





# In[ ]:





# In[ ]:





    

