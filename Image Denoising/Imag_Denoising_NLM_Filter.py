import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Computes and returns the Gaussian Kernel w.r.t provided Standard deviation, Kernel Size and whether kernel needs to be normalized
def compute_gauss_kernel(std_dev,k_size,normalize=True):
    
    kernel=np.zeros((k_size,k_size)) # Creates an empty kernel of required size
    
    if(std_dev==0): # If given Standard_Deviation is 0 Gaussian kernel will be a 1x1 matrix with value 1 as we cannot use original formula which will result in division by zero
        kernel[0,0]=1
        return kernel

    denom=1.0/(math.pi*2*std_dev*std_dev) # Denominator in the formula for Gaussian Kernel is common for all (x,y) locations

    center=k_size//2 # Calculates the center coordinates (center,center) of the kernel

    for i in range(k_size):
        for j in range(k_size):
            dist_x=i-center #kernel at location (i,j) is at distance dist_x = i-center from center of kernel along x-axis
            dist_y=j-center #kernel at location (i,j) is at distance dist_y = j-center from center of kernel along y-axis
            kernel[i,j]=math.exp(-1*((dist_x*dist_x)+(dist_y*dist_y))/(2*std_dev*std_dev))/denom #Computes kernel value at (i,j) as per formula of Gaussian kernel 
    
    # If Normalize flag is true then it normalizes the kernel values such that they sum to 1 by dividing each element with sum of all elements in the kernel
    if(normalize==True):
        tsum=np.sum(kernel)
        kernel=kernel/tsum

    return kernel

# Convolves the input image with the input kernel and returns the convolved image of same dimension as input image
def do_convolution(input_kernel,input_img):
    
    k_size=input_kernel.shape[0] # Gets the kernel_size from input kernel
    half_width=k_size//2 # Calculates the number of elements from center of kernel to its edge excluding the center element
    padded_input_img=np.pad(input_img,((half_width,half_width),(half_width,half_width),(0,0)),'edge') # Pads the input image with edge values along all edges with half_width number of elements calcuate earlier so input data is available while convolving near-border elements of input image
    
    new_img=np.zeros(input_img.shape,dtype=np.float64) # Stores the convolved image which is of same dimension as input image
    
    flipped_kernel=np.flip(input_kernel) # Flips the input kernel along both axes as the operation we are performing is a convolution

    m,n,__=input_img.shape

    #Calculates the intensity for each pixel of convoluted image one by one by sliding the flipped input kernel over the padded input image
    for x in range(m):
        for y in range(n):
            for z in range(3):
                val=0.0
                #Centers the input kernel's center at original input image's (x,y) pixel and calculates corresponding new_img(x,y), Alternatively aligns (0,0) of kernel with (x,y) of padded input image
                for i in range(k_size):
                    for j in range(k_size):
                        val+=(flipped_kernel[i,j]*padded_input_img[x+i,y+j,z])
                if(val>1.0): #To essentially threshold and keep the image intensity within valid range
                    val=1.0
                new_img[x,y,z]=val
    
    return new_img

def get_NLM_Filter(input_img,W,W_sim,sigma,x,y):
    patch_p = input_img[x-W_sim:x+W_sim+1,y-W_sim:y+W_sim+1,0:3]
    V_p = np.transpose(patch_p.flatten())
    w_p = np.zeros((2*W+1,2*W+1),dtype=np.float64)
    
    for i in range(x-W,x+W+1):
        for j in range(y-W,y+W+1):
            patch_q = input_img[i-W_sim:i+W_sim+1,j-W_sim:j+W_sim+1,0:3]
            V_q = np.transpose(patch_q.flatten())
            diff_vector = V_p-V_q
            w_p[i-(x-W),j-(y-W)] = np.exp(-1*np.matmul(np.transpose(diff_vector),(diff_vector))/(sigma*sigma))

    w_p = w_p/np.sum(w_p)

    return w_p


def do_NLM_Filter_img(input_img,W,W_sim,sigma):

    m,n,__ = input_img.shape
    width = W+W_sim
    padded_input_img=np.pad(input_img,((width,width),(width,width),(0,0)),'edge')

    new_img=np.zeros(input_img.shape,dtype=np.float64)

    for x in range(width,m+width):
        for y in range(width,n+width):
            patch_p = padded_input_img[x-W_sim:x+W_sim+1,y-W_sim:y+W_sim+1,0:3]
            V_p = np.transpose(patch_p.flatten())
            w_p = np.zeros((2*W+1,2*W+1),dtype=np.float64)
            
            for i in range(x-W,x+W+1):
                for j in range(y-W,y+W+1):
                    patch_q = padded_input_img[i-W_sim:i+W_sim+1,j-W_sim:j+W_sim+1,0:3]
                    V_q = np.transpose(patch_q.flatten())
                    diff_vector = V_p-V_q
                    w_p[i-(x-W),j-(y-W)] = np.exp(-1*np.matmul(np.transpose(diff_vector),(diff_vector))/(sigma*sigma))

            w_p = w_p/np.sum(w_p)
            w_p = w_p.flatten()
            
            for k in range(3):
                patch_color = padded_input_img[x-W:x+W+1,y-W:y+W+1,k]
                V_p_k = np.transpose(patch_color.flatten())
                new_img[x-width,y-width,k] = min(np.matmul(np.transpose(V_p_k),w_p),1.0)
    
    return new_img

def calc_PSNR(reference_img,compared_img):
    diff_img = np.transpose(reference_img.flatten() - compared_img.flatten())
    #mse = np.mean(np.square(compared_img-reference_img))
    mse = np.matmul(np.transpose(diff_img),diff_img)/reference_img.size
    return 10*np.log10(1.0/mse)


orig_img = Image.open("krishna.png") #Imports the input image using PILs image module
orig_img_arr = np.array(orig_img,dtype=np.float64)/255 #Converts the image into a numpy array

noisy_img = Image.open("krishna_0_001.png") #Imports the input image using PILs image module
noisy_img_arr = np.array(noisy_img,dtype=np.float64)/255 #Converts the image into a numpy array

PSNR_noisy = calc_PSNR(orig_img_arr,noisy_img_arr)

print(f"PSNR of Noisy Image is: {PSNR_noisy}")

w_sim = 3
w_vals = [3,5]
sigma_vals=[0.1,0.2,0.3,0.4,0.5]

NLM_denoised_imgs = {}
PSNR_NLM_denoised_imgs = {}

for w in w_vals:
    NLM_denoised_imgs[w]={}
    PSNR_NLM_denoised_imgs[w]={}
    for sigma in sigma_vals:
        NLM_denoised_imgs[w][sigma] = do_NLM_Filter_img(noisy_img_arr,w,w_sim,sigma)
        PSNR_NLM_denoised_imgs[w][sigma] = calc_PSNR(orig_img_arr,NLM_denoised_imgs[w][sigma])
        print(f"Using W = {w} W_sim = {w_sim} and sigma = {sigma} PNR of Denoised image using NLM Filter is: {PSNR_NLM_denoised_imgs[w][sigma]}")
        #denoised_img=Image.fromarray(NLM_denoised_imgs[w][sigma])
        #denoised_img.save(f"NLM_Denoised_Img_W_{w}_sigma_0_{sigma*10}.png")
        mpimg.imsave(f"NLM_Denoised_Img_W_{w}_sigma_0_{sigma*10}.png",NLM_denoised_imgs[w][sigma])


std_dev_vals=[0.1,0.2,0.3,0.4,0.5] #Standard Deviation or Sigma to be used as input for deriving Gaussian Kernel
gauss_kernel_size=7 # The Gaussian kernel will be a matrix of size (kerne_size x kernel_size) where kernel_size varies w.r.t Standard Deviation

gauss_denoised_imgs = {}
PSNR_gauss_denoised_imgs = {}

for std_dev in std_dev_vals:
    gauss_kernel=compute_gauss_kernel(std_dev,gauss_kernel_size) # Gets the corresponding Gaussian Kernel
    gauss_denoised_imgs[std_dev]=do_convolution(gauss_kernel,noisy_img_arr) # Convolves the input image in img_arr with Gaussian kernel to obtain the blurred image
    PSNR_gauss_denoised_imgs[std_dev] = calc_PSNR(orig_img_arr,gauss_denoised_imgs[std_dev])
    print(f"Using Standard Deviation = {std_dev} PNR of Denoised image using Gaussin Filter is: {PSNR_gauss_denoised_imgs[std_dev]}")
    #denoised_img=Image.fromarray(gauss_denoised_imgs[std_dev])
    #denoised_img.save(f"Gauss_Denoised_Img_sigma_0_{std_dev*10}.png")
    mpimg.imsave(f"Gauss_Denoised_Img_sigma_0_{std_dev*10}.png",gauss_denoised_imgs[std_dev])

plt.figure(figsize=(10, 8))
plt.plot(sigma_vals, [PSNR_noisy for sigma in sigma_vals], 'y--')
plt.plot(sigma_vals, [PSNR_NLM_denoised_imgs[3][sigma] for sigma in sigma_vals])
plt.plot(sigma_vals, [PSNR_NLM_denoised_imgs[5][sigma] for sigma in sigma_vals])
plt.plot(sigma_vals, [PSNR_gauss_denoised_imgs[sigma] for sigma in sigma_vals])
plt.xlabel(r"${\sigma}$")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend(["Baseline", "W = 3, W_sim=3", "W = 5, W_sim=3", "Gaussian"])
plt.xticks(sigma_vals)
plt.title(r"PSNR vs $\sigma$")
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(get_NLM_Filter(noisy_img_arr,5,3,0.5,31,46))
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.title("NLM Kernel for x=31 y=46")
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(get_NLM_Filter(noisy_img_arr,5,3,0.5,38,58))
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.title("NLM Kernel for x=38 y=58")
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(compute_gauss_kernel(1.0,11), 'gray')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.title("Gaussian Kernel for both coordinates")
plt.show()

fig, ((im1, im2, im3), (im4, im5, im6)) = plt.subplots(2, 3, figsize=(12, 12), constrained_layout=True)
im1.set_xticks([])
im1.set_yticks([])
im2.set_xticks([])
im2.set_yticks([])
im3.set_xticks([])
im3.set_yticks([])
im4.set_xticks([])
im4.set_yticks([])
im5.set_xticks([])
im5.set_yticks([])
im6.set_xticks([])
im6.set_yticks([])
im1.axis('off')
im2.axis('off')
im3.axis('off')
im4.axis('off')
im5.axis('off')
im6.axis('off')
im1.imshow(noisy_img_arr[31-5:31+6,46-5:46+6,:])
im1.title.set_text(rf"Noisy Image Patch at (31,46)")
im2.imshow(NLM_denoised_imgs[5][0.5][31-5:31+6,46-5:46+6,:])
im2.title.set_text(rf"NLM Denoised Image Patch at (31,46)")
im3.imshow(gauss_denoised_imgs[0.5][31-5:31+6,46-5:46+6,:])
im3.title.set_text(rf"Gaussian Denoised Image Patch at (31,46)")
im4.imshow(noisy_img_arr[38-5:38+6,58-5:58+6,:])
im4.title.set_text(rf"Noisy Image Patch at (38,58)")
im5.imshow(NLM_denoised_imgs[5][0.5][38-5:38+6,58-5:58+6,:])
im5.title.set_text(rf"NLM Denoised Image Patch at (38,58)")
im6.imshow(gauss_denoised_imgs[0.5][38-5:38+6,58-5:58+6,:])
im6.title.set_text(rf"Gaussian Denoised Image Patch at (38,58)")
plt.show()
