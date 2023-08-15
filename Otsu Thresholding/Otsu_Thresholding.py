import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_otsu_threshold(img):
    m,n = img.shape
    numpixels = m*n #Get total number of pixels

    minval = np.min(img) #Find minimum intensity
    maxval = np.max(img) #Find maximum intensity

    freq = np.zeros(maxval-minval+1,dtype=np.float64) # Stores frequncy of each intensity in image freq[0] stores number of pixels with intensity 'minval' and son on

    #Fills the freq array
    for i in range(m):
        for j in range(n):
            freq[img[i,j]-minval]+=1

    probdist = freq/numpixels #Probability p(i) of each intensity i

    i_mat = np.arange(minval,maxval+1,dtype=np.float64) # 'i' matrix used for mulitplying with p(i)

    w1 = np.cumsum(probdist,dtype=np.float64) # Array storing w for class 1 corresponding to each threshold value between minval and maxval
    w2 = 1-w1 # For class 1 it will be 1-w0

    i_x_pi_1 = np.cumsum(i_mat*probdist) # Summation (i*p(i)) for class 1 corresponding to each possible threshold value

    totsum=i_x_pi_1[-1]

    i_x_pi_2 = totsum - i_x_pi_1 #For class 2 Summation (i*p(i)) will be totsum(which remains constant) - corresponding value for class 0

    mean1 = i_x_pi_1/w1 # Mean of class 1 corresponding to each threshold value between minval and maxval
    mean2 = i_x_pi_2/w2 # Mean of class 2 corresponding to each threshold value between minval and maxval

    inter_class_var = w1*w2*np.square(mean1-mean2,dtype=np.float64) # Inter class variance corresponding to each threshold value between minval and maxval

    threshold = minval + np.argmax(inter_class_var) # Finds the threshold for which Inter class variance is maximum (Otsu's Threshold)

    return threshold   



img1 = Image.open("palmleaf1.png") #Imports the input image using PILs image module
img_arr1 = np.array(img1) #Converts the image into a numpy array

img2 = Image.open("palmleaf2.png") #Imports the input image using PILs image module
img_arr2 = np.array(img2) #Converts the image into a numpy array

threshold1 = get_otsu_threshold(img_arr1) #Get otsu threshold for img1
threshold2 = get_otsu_threshold(img_arr2) #Get otsu threshold for img2

print(threshold1)
print(threshold2)

thresholded_img_arr1 = np.zeros(img_arr1.shape) # Otsu Thresholded image corresponding to img1
thresholded_img_arr2 = np.zeros(img_arr2.shape) # Otsu Thresholded image corresponding to img2

thresholded_img_arr1[img_arr1 > threshold1] = 255 # Pixels in original image with intnesities greater thean otsu threshold are made '255' and with intensities <= threshold remain 0 so we get a binary image
thresholded_img_arr2[img_arr2 > threshold2] = 255 # Pixels in original image with intnesities greater thean otsu threshold are made '255' and with intensities <= threshold remain 0 so we get a binary image

plt.imshow(thresholded_img_arr1,cmap='gray')
plt.show()
plt.imshow(thresholded_img_arr2,cmap='gray')
plt.show()

