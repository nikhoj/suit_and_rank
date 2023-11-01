import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('captured_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot the histogram
plt.hist(image.ravel(), 256, [0, 256])
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

threshold = 140

image[image <= threshold] = 0
image[image > threshold] = 255
image = 255 - image

#now lets remove the background
L, W = image.shape

for i in range(W):
    if np.all(image[int(L/2), i:i+4]) == 0:
        print(i)
        image = image[:,i:]
        break

for i in range(W-1,-1,-1):
    if np.all(image[int(L/2), i-4:i]) == 0:
        print(i)
        image = image[:,0:i]
        break

for i in range(L):
    if np.all(image[i:i+4, int(W/2)]) == 0:
        print(i)
        image = image[i:,:]
        break

for i in range(L-1,-1,-1):
    if np.all(image[i-4:i, int(W/2)]) == 0:
        print(i)
        image = image[0:i,:]
        break

image = 255- image
# plt.title('card')
# plt.imshow(image, cmap="gray")
# plt.waitforbuttonpress()
# plt.close()

cv2.imwrite('reconstructed_image.jpg', image)