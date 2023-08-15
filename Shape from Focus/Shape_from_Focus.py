import numpy as np
from PIL import Image
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Convolves the input image with the input kernel and returns the convolved image of same dimension as input image
def do_convolution(input_kernel,input_img):
    
    k_size=input_kernel.shape[0] # Gets the kernel_size from input kernel
    half_width=k_size//2 # Calculates the number of elements from center of kernel to its edge excluding the center element
    padded_input_img=np.pad(input_img,((half_width,half_width),(half_width,half_width)),'edge') # Pads the input image with edge values along all edges with half_width number of elements calcuate earlier so input data is available while convolving near-border elements of input image
    
    new_img=np.zeros(input_img.shape) # Stores the convolved image which is of same dimension as input image
    
    #flipped_kernel=np.flip(input_kernel) # Flips the input kernel along both axes as the operation we are performing is a convolution

    m,n=input_img.shape

    #Calculates the intensity for each pixel of convoluted image one by one by sliding the flipped input kernel over the padded input image
    for x in range(m):
        for y in range(n):
            val=0.0
            #Centers the input kernel's center at original input image's (x,y) pixel and calculates corresponding new_img(x,y), Alternatively aligns (0,0) of kernel with (x,y) of padded input image
            for i in range(k_size):
                for j in range(k_size):
                    val+=(input_kernel[i,j]*padded_input_img[x+i,y+j])
            new_img[x,y]=val
    
    return new_img


def do_sum_modified_laplacian(img_arr,q): #Calculates and returns the Sum Modified Laplacian Array (SML) for given img_arr and q
    
    add_kernel=np.ones((2*q+1,2*q+1)) #Add kernel when convolved stores the summation over (2*q+1 x 2*q+1) patch for each pixel
    
    x,y=img_arr.shape

    dx=np.array([[0,0,0], #Kernel for double derivative along x-direction
                [1,-2,1],
                [0,0,0]])

    dy=np.array([[0,1,0], #Kernel for double derivative along y-direction
                [0,-2,0],
                [0,1,0]])

    padded_img_arr=np.pad(img_arr,((q,q),(q,q)),'constant',constant_values=0) # Pads the image along each edge with q elements

    I_xx_abs=np.abs(do_convolution(dx,padded_img_arr)) #Generates |Ixx|
    I_yy_abs=np.abs(do_convolution(dy,padded_img_arr)) #Generates |Iyy|
    
    modified_lap_arr=I_xx_abs+I_yy_abs #Generates Modified Laplacian of the given image
    sum_modified_lap_arr=do_convolution(add_kernel,modified_lap_arr) #Generates Sum Modified Laplacian of the given image by convolving it with add kernel
    return sum_modified_lap_arr[q:x+q,q:y+q] #Trims the boundary values and returns SML of same size as input image back


delta_d=50.5 #Provided shift of camera between each consecutive frame

imgstack=loadmat("stack.mat") #Loads the data from provided .mat files

nframes=imgstack['numframes'][0,0] #Extracts the number of frames/images in the data

img_frame_arr=np.empty((nframes,imgstack['frame001'].shape[0],imgstack['frame001'].shape[1]),dtype=np.uint8) # Will store all the frames/images in a 3D numpy array

for i in range(nframes): #Parses through the data to store each image in img_frame_arr
    key="frame"+f"{i+1:03}" #Key for the corresonding frame
    img_frame_arr[i]=imgstack[key]


qvar=[0,1,2] # Different values of q for which to perform SML

perpix_sharp_frameid=np.empty((len(qvar),img_frame_arr.shape[1],img_frame_arr.shape[2])) #For each pixel stores the frameid for which it is sharpest/ its SML value is maximum
perpix_depth=np.empty((len(qvar),img_frame_arr.shape[1],img_frame_arr.shape[2])) #For each pixel stores its corresponding scene depth
qid=0

for q in qvar: # Calculates and plots 3D structure corresponding to each value of q in qvar
    sml_frame_arr=np.empty(img_frame_arr.shape) #Stores the SML for each frame/image
    
    for fid in range(nframes): #Iterates through all frames/images and calculates their SML
        sml_frame_arr[fid]=do_sum_modified_laplacian(img_frame_arr[fid],q)
    
    perpix_sharp_frameid[qid]=np.argmax(sml_frame_arr,axis=0) #For each pixel finds the frameid for which it is sharpest/ its SML value is maximum
    perpix_depth[qid]=perpix_sharp_frameid[qid]*delta_d #For each pixel using perpix_sharp_frameid calculates the corresponding scene depth
    
    # Below section is to plot the 3D structure using the Depth map generated above
    x_arr = range(img_frame_arr.shape[1])
    y_arr = range(img_frame_arr.shape[2])
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_arr, y_arr)
    ha.plot_surface(X, Y, perpix_depth[qid])
    plt.title(f"3D Structure for q={q}")
    plt.show()
    
    qid+=1

