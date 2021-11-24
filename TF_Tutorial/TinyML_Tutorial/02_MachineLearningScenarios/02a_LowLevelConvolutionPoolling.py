import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

#####
#	#
# 1	#	Get an image from the dataset
#	#
#####
origImg 	= misc.ascent( )
plt.subplot(1, 3, 1)
plt.grid(False)
plt.gray()
#plt.axis('off')
plt.title("Original Image")
plt.imshow( origImg )  

#####
#	#
# 2	#	Create a filter. 3x3
#	#
#####
convFilter = [ [ -1, -2, -1 ], [ 0, 0, 0 ], [ 1, 2, 1 ] ]
weight  = 1

#####
#	#
# 3	#	Convolute
#	#
#####
copyImg		= np.copy( origImg )
# Get some image info.
size_x = copyImg.shape[0]
size_y = copyImg.shape[1]
for x in range( 1, size_x - 1 ):
  for y in range( 1, size_y - 1 ):
      convolution = 0.0
      convolution = convolution + ( origImg[x - 1, y-1] 	* convFilter[0][0] )
      convolution = convolution + ( origImg[x, y-1] 		* convFilter[1][0] )
      convolution = convolution + ( origImg[x + 1, y-1] 	* convFilter[2][0] )
      convolution = convolution + ( origImg[x-1, y] 		* convFilter[0][1] )
      convolution = convolution + ( origImg[x, y] 			* convFilter[1][1] )
      convolution = convolution + ( origImg[x+1, y] 		* convFilter[2][1] )
      convolution = convolution + ( origImg[x-1, y+1] 		* convFilter[0][2] )
      convolution = convolution + ( origImg[x, y+1] 		* convFilter[1][2] )
      convolution = convolution + ( origImg[x+1, y+1] 		* convFilter[2][2] )
      convolution = convolution * weight
      if(convolution<0):
        convolution=0
      if(convolution>255):
        convolution=255
      copyImg[ x, y ] = convolution
plt.subplot(1, 3, 2)
plt.grid(False)
plt.gray()
#plt.axis('off')
plt.title("Convoluted Image")
plt.imshow( copyImg )      

#####
#	#
# 4	#	Pool
#	#
#####
new_x = int(size_x/4)
new_y = int(size_y/4)
zippImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 4):
  for y in range(0, size_y, 4):
    pixels = []
    pixels.append(copyImg[x, y])
    pixels.append(copyImg[x+1, y])
    pixels.append(copyImg[x+2, y])
    pixels.append(copyImg[x+3, y])
    pixels.append(copyImg[x, y+1])
    pixels.append(copyImg[x+1, y+1])
    pixels.append(copyImg[x+2, y+1])
    pixels.append(copyImg[x+3, y+1])
    pixels.append(copyImg[x, y+2])
    pixels.append(copyImg[x+1, y+2])
    pixels.append(copyImg[x+2, y+2])
    pixels.append(copyImg[x+3, y+2])
    pixels.append(copyImg[x, y+3])
    pixels.append(copyImg[x+1, y+3])
    pixels.append(copyImg[x+2, y+3])
    pixels.append(copyImg[x+3, y+3])
    pixels.sort(reverse=True)
    zippImage[int(x/4),int(y/4)] = pixels[0]
plt.subplot(1, 3, 3)
plt.grid(False)
plt.gray()
#plt.axis('off')
plt.title("Polled Image")
plt.imshow( zippImage )

plt.show()

#### end of file ####
