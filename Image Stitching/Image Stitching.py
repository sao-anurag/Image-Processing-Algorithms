import numpy as np
from PIL import Image, ImageOps
import math
from sift import sift as sift_corresp
import scipy.linalg

def do_ransac(corresp1,corresp2): ##erforms RANSAC and generates Homography matrix(H) corresponding to transforming points in corresp1 to their corresponding points in corresp2
    
    tot_points=corresp1.shape[0]  #Total no. of points(|M|) in corresp1
    all_point_ids=np.arange(corresp1.shape[0]) #Set of all points(M) in corresp1
    consensus_size=0 # Consensus size of RANSAC initialized to zero
    req_consensus_size=0.9*(tot_points-4) # Required consensus size= 0.8 * |P| where, |P| no. of remaining points which is |M|-4 since we take 4 points in each iteration for estimating H
    epsilon=5 # Value compared to check if error is within bound
    
    while (consensus_size<=req_consensus_size): #Loop is run till we get H correponding to 4 points such that consensus is of atleast the required size
        chosen_point_ids=np.random.choice(corresp1.shape[0], size=4, replace=False) #In each iteration choose 4 random points from available points whose set we call (R)
        A=np.zeros((8,9)) #A matrix initialized to 0, generated from 4 equations formed corresponding to chosen 4 points

        for i in range(4): # Generate the A matrix corresponding to 4 points as per equation Ah=0
            A[2*i][0]=corresp1[chosen_point_ids[i]][0]
            A[2*i][1]=corresp1[chosen_point_ids[i]][1]
            A[2*i][2]=1
            A[2*i][6]=-corresp2[chosen_point_ids[i]][0]*corresp1[chosen_point_ids[i]][0]
            A[2*i][7]=-corresp2[chosen_point_ids[i]][0]*corresp1[chosen_point_ids[i]][1]
            A[2*i][8]=-corresp2[chosen_point_ids[i]][0]

            A[(2*i)+1][3]=corresp1[chosen_point_ids[i]][0]
            A[(2*i)+1][4]=corresp1[chosen_point_ids[i]][1]
            A[(2*i)+1][5]=1
            A[(2*i)+1][6]=-corresp2[chosen_point_ids[i]][1]*corresp1[chosen_point_ids[i]][0]
            A[(2*i)+1][7]=-corresp2[chosen_point_ids[i]][1]*corresp1[chosen_point_ids[i]][1]
            A[(2*i)+1][8]=-corresp2[chosen_point_ids[i]][1]
        
        h=scipy.linalg.null_space(A) #Once a matrix is generated h will be the null space of A

        H=h.reshape((3,3)) #Since h is a 9*1 matrix we reshape it into a 3*3 matrix to get H

        rem_point_ids=np.delete(all_point_ids,chosen_point_ids) # We genmerate set of remaing points(P) other than chosen 4 points from M i.e. we generate P=M/R
        consensus_size=0 # Before calculating consensus size for this 4 points we initialize it to 0

        P=np.zeros((3,1)) #Used to calculate xi" and yi" from xi and yi using H calculated earlier
        P[2]=1

        for pid in rem_point_ids: # Calculates the Consensus set corresponding to H calculated earlier
            P[0]=corresp1[pid][0]
            P[1]=corresp1[pid][1]

            P_calc=np.matmul(H,P) # Generates xi" and yi" which are scaled by zi"

            x_calc=P_calc[0]/P_calc[2] #Scales down by zi" to calculate xi"
            y_calc=P_calc[1]/P_calc[2] #Scales down by zi" to calculate yi"

            error_square=((x_calc-corresp2[pid][0])*(x_calc-corresp2[pid][0]))+((y_calc-corresp2[pid][1])*(y_calc-corresp2[pid][1])) #Calculates error squared w.r.t to corresponding point in corresp2

            if(error_square<(epsilon*epsilon)): #If error is within bound increases the consensus set size
                consensus_size+=1
    
    return H #When consensus set size meets the requirement return the corresponding H

def do_blinterp(img_arr,src_x,src_y): #Performs Bilinear interpolation on supplied point obtained from target to source mapping for img_arr
    
    intensity=0
    m,n=img_arr.shape

    if (src_x>=0) and (src_y>=0) and (src_x<m-1) and (src_y<n-1): #Does the bilinear interpolation only if Target to Source mapping yields source x & y coordinates within bounds of original image
        #Bilinear Interpolation is carried as per formula derived in class
        tl_x=math.floor(src_x) #Finds the the x-coordinate of the top-left corner point corresponding to source pixel
        tl_y=math.floor(src_y) #Finds the the y-coordinate of the top-left corner point corresponding to source pixel
        a=src_x-tl_x #Distance along x-direction of corrsponding source pixel from top-left point
        b=src_y-tl_y #Distance along y-direction of corrsponding source pixel from top-left point
        val=((1-a)*(1-b)*img_arr[tl_x,tl_y])+((a)*(1-b)*img_arr[tl_x+1,tl_y])+((1-a)*(b)*img_arr[tl_x,tl_y+1])+((a)*(b)*img_arr[tl_x+1,tl_y+1]) # Calculates the intensity of target pixel using intensitites of neighboring pixels of mapped source pixel using bilinear interpolation formula
        intensity=int(val) #Since we need intensity to be integer(0-255) while intensity obtained above maybe decimal we round the the value to obtain a valid pixel intensity

    return intensity


img1 = Image.open("img1.png") #Imports the input image1 using PILs image module
img1 = ImageOps.grayscale(img1)
img_arr1 = np.array(img1) #Converts the image into a numpy array
m1,n1=img_arr1.shape #Stores the dimensions of the image1 in m1(for X-axis) and n1(Y-axis)

img2= Image.open("img2.png") #Imports the input image2 using PILs image module
img2 = ImageOps.grayscale(img2)
img_arr2 = np.array(img2) #Converts the image into a numpy array
m2,n2=img_arr2.shape #Stores the dimensions of the image2 in m2(for X-axis) and n2(Y-axis)

img3= Image.open("img3.png") #Imports the input image3 using PILs image module
img3 = ImageOps.grayscale(img3)
img_arr3 = np.array(img3) #Converts the image into a numpy array
m3,n3=img_arr3.shape #Stores the dimensions of the image3 in m3(for X-axis) and n3(Y-axis)

[corresp_im12_1, corresp_im12_2] = sift_corresp(img_arr2,img_arr1) #Gets point correspondences set between Img2 and Img1 using SIFT

H21=do_ransac(corresp_im12_1,corresp_im12_2) #Performs ransac to obtain H21 such that I1 = H21 * I2

[corresp_im32_1, corresp_im32_2] = sift_corresp(img_arr2,img_arr3) #Gets point correspondences set between Img2 and Img3 using SIFT

H23=do_ransac(corresp_im32_1,corresp_im32_2) ##Performs ransac to obtain H23 such that I3 = H23 * I2

numCanvasRows=m1*2 #Height of new canvas is tow times original image height
numCanvasCols=n1+n2+n3 #Width of new canvas is sum of width of original images (assuming it is a horizontal panorama)
offsetRow=(m1//2) #Corresponding height offset to center image is floor(OriginalHeight/2)
offsetCol=n2 #Corresponding Width offset to center w.r.t Img2 is n2

canvas=np.zeros((numCanvasRows,numCanvasCols),dtype=np.uint8) #Generates a blank canvas with values initialized to zero of required dimensions

for ii in range(numCanvasRows): #Fills in the values for new canvas pixel by pixel as per algorithm template given in question
    for jj in range(numCanvasCols):

        i=ii-offsetRow #Correponding coordinates in img2
        j=jj-offsetCol

        P=np.zeros((3,1)) #Used to calculate corresponding coordinates for Img1 & Img3 via target to source mapping using correponding H
        P[0]=i
        P[1]=j
        P[2]=1
        
        P1_calc=np.matmul(H21,P) #Obtains target to source mapped Img1 coordinates
        
        i1=P1_calc[0]/P1_calc[2] #Scales the coordinates w.r.t z value
        j1=P1_calc[1]/P1_calc[2]

        P3_calc=np.matmul(H23,P) #Obtains target to source mapped Img3 coordinates
        
        i3=P3_calc[0]/P3_calc[2] #Scales the coordinates w.r.t z value
        j3=P3_calc[1]/P3_calc[2]

        v1=do_blinterp(img_arr1,i1,j1) #Performs bilinear interpolation to obtain intensity w.r.t Img1
        v2=do_blinterp(img_arr2,i,j) #Performs bilinear interpolation to obtain intensity w.r.t Img2
        v3=do_blinterp(img_arr3,i3,j3) #Performs bilinear interpolation to obtain intensity w.r.t Img3

        #Performs blend as following in order to obtain decent image

        if(v2>0):
            canvas[ii,jj]=v2
        elif(v1>0):
            canvas[ii,jj]=v1
        else:
            canvas[ii,jj]=v3

img_mosaic=Image.fromarray(canvas) #Converts the mosaicedimage which is a numpy array to PIL Image format
img_mosaic.save("IMG_mosaic.png") #Saves the mosaiced image