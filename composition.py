
import numpy as np
import cv2 as cv


rgb= [ 61,61,88]
read='val/airforce1white.jpg'
write='img/RGB_test.jpg'

def RGB(rgb,read,write ):
 
    r,g,b= np.array(rgb)
    
    src1 = cv.imread(read)  
    src1[:,:,0] = src1[:,:,0] -r 
    src1[:,:,1] = src1[:,:,1] -g 
    src1[:,:,2] = src1[:,:,2] -b 
    src1 = cv.cvtColor(src1, cv.COLOR_RGB2BGR)

    plt.imshow(src1)
    cv.imwrite(write,src1) 


foreground ='img/RGB_test.jpg'  
background = 'val/airforce1white.jpg' 

alpha ='mask1.jpg'
output='img/linear_blend_output.jpg'

def linear_blend(foreground, background, alpha ,output): 
    foreground =cv.imread(foreground).astype(float)
    background = cv.imread(background).astype(float) 
    alpha =cv.imread(alpha)    

    foreground= cv.resize(foreground, dsize=(640, 640), interpolation=cv2.INTER_AREA)
    background = cv.resize(background, dsize=(640, 640), interpolation=cv2.INTER_AREA)
    alpha = cv.resize(alpha, dsize=(640, 640), interpolation=cv2.INTER_AREA)

    alpha = alpha.astype(float)/255

    foreground = cv.multiply(alpha, foreground)
    background = cv.multiply(1.0 - alpha, background)
    outImage = cv.add(foreground, background)

    plt.imshow(outImage)
    cv.imwrite(output,outImage)

