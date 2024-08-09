#!/usr/bin/env python
# coding: utf-8

# In[41]:


import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image


# In[42]:


# Reading the .ppm image using 'imagio' library
img = imageio.imread("basketball-court.ppm")


# In[43]:


# Displaying the image using matplotlib
plt.imshow(img)
plt.axis("on")
plt.show()


# In[44]:


# Define source points
src_pts = np.array([[246, 45], [419, 70], [4, 201],[285, 302]])


# In[45]:


# Define destination points
dst_pts = np.array([[0, 0], [0, 500], [940, 0], [940, 500]])


# In[12]:


# Calculate homography matrix using DLT algorithm
H, _ = cv2.findHomography(src_pts, dst_pts)


# In[46]:


def calculate_homography_matrix(src_points, dst_points):
    # construct the coefficient matrix A
    A = []
    for i in range(len(src_points)):
        x, y = src_points[i]
        u, v = dst_points[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)

    # calculate the homography matrix using SVD
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)

    # normalize the homography matrix
    H = H / H[2, 2]


# In[47]:


# Generate a blank 940x500 image
dst_img = np.zeros((500, 940, 3), dtype=np.uint8)


# In[48]:


# Warp basketball court from source image to the blank image
dst_img = cv2.warpPerspective(img, H, (940, 500))


# In[49]:


# Display output image
cv2.imshow("Output Image", dst_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[101]:


#Performing Bilinear Internpolation

#dst_size = [940, 400, 3]
def bilinear_interpolation(src_image, dst_size):
    # get the size of the source image
    src_height, src_width, _ = src_image.shape

    # create a new image with the desired size
    dst_height, dst_width = dst_size
    dst_image = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)

    # calculate the scaling factor
    scale_x = float(src_width) / dst_width
    scale_y = float(src_height) / dst_height

    # looping over each pixel in the output image
    for dst_y in range(dst_height):
        for dst_x in range(dst_width):
            # calculating the coordinates of the source image
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5

            # calculating the four nearest pixels of the source image
            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))
            x1 = x0 + 1
            y1 = y0 + 1

            # clipping the coordinates to the size of the source image
            x0 = max(0, min(x0, src_width - 1))
            x1 = max(0, min(x1, src_width - 1))
            y0 = max(0, min(y0, src_height - 1))
            y1 = max(0, min(y1, src_height - 1))

            # calculating the weights for each pixel
            wx1 = src_x - x0
            wx0 = 1.0 - wx1
            wy1 = src_y - y0
            wy0 = 1.0 - wy1

            # interpolation of the pixel values using the four nearest pixels
            dst_image[dst_y, dst_x, :] = (
                wx0 * wy0 * src_image[y0, x0, :] +
                wx1 * wy0 * src_image[y0, x1, :] +
                wx0 * wy1 * src_image[y1, x0, :] +
                wx1 * wy1 * src_image[y1, x1, :]
            )

    return dst_image


# In[ ]:





# In[110]:


# calculating homography using line features in a DLT algorithm

def estimate_homography_from_lines(src_lines, dst_lines):

#     assert len(src_lines) == len(dst_lines), 'The number of source lines and destination lines must be equal'

    # Constructing matrix A
    A = []
    for src_line, dst_line in zip(src_lines, dst_lines):
#         if len(src_line) < 4 or len(dst_line) < 4:
#             continue
#         a, b, c = src_line
#         u, v, w = dst_line

        x1, y1, x2, y2 = src_line
        u1, v1, u2, v2 = dst_line

#         A.append([a, b, c, 0, 0, 0, -u*a, -u*b, -u*c, v*a, v*b, v*c, 0, 0, 0, -w*a, -w*b, -w*c])
#         A.append([0, 0, 0, a, b, c, -v*a, -v*b, -v*c, 0, 0, 0, w*a, w*b, w*c, 0, 0, 0])

        A.append([-x1, -y1, -1, 0, 0, 0, u1*x1, u1*y1, u1])
        A.append([0, 0, 0, -x1, -y1, -1, v1*x1, v1*y1, v1])
        A.append([-x2, -y2, -1, 0, 0, 0, u2*x2, u2*y2, u2])
        A.append([0, 0, 0, -x2, -y2, -1, v2*x2, v2*y2, v2])
        
#         x1, y1, x2, y2 = src_line[:4]
#         u1, v1, u2, v2 = dst_line[:4]
        
#         A.append([x1, y1, 1, 0, 0, 0, -u1*x1, -u1*y1, -u1])
#         A.append([0, 0, 0, x1, y1, 1, -v1*x1, -v1*y1, -v1])
#         A.append([x2, y2, 1, 0, 0, 0, -u2*x2, -u2*y2, -u2])
#         A.append([0, 0, 0, x2, y2, 1, -v2*x2, -v2*y2, -v2])
        
        
    A = np.asarray(A)

    # Solve for the homography matrix using SVD
    U, S, Vt = np.linalg.svd(A)
    L = Vt[-1,:]/Vt[-1,-1]
    H = L.reshape(3, 3)

    # Normalize the homography matrix
    H = H / H[2, 2]

    return H


# In[ ]:





# In[114]:


def warp_image(dst_img, H):
    """
    Warps an input image using a homography matrix H.
    """
    height, width = dst_img.shape[:2]
    warped_img = cv2.warpPerspective(dst_img, H, (width, height))
    return warped_img


# In[ ]:





# In[ ]:


import cv2

# Loading the input image again
# img = cv2.imread('basketball-court.ppm')

# Converting image to grayscale
gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

# Applying 'Canny edge' detection on gray image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Applying Hough line detection technique on the edges
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Extract line segment coordinates
src_lines = []
dst_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    src_lines.append([x1, y1, x2, y2])
    dst_lines.append([x1, y1, x2, y2])
#     src_lines.append((y2-y1, x1-x2, x2*y1-x1*y2))
#     dst_lines.append([x1, y1, x2, y2])

# Estimate homography matrix
H = estimate_homography_from_lines(src_lines, dst_lines)

# Warp input image
warped_img = warp_image(dst_img, H)

# Display warped image
cv2.imshow('Warped Image', warped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




