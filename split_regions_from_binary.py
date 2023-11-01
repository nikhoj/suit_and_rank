#####################################################################################
# Author        : Md Mahabub Uz Zaman
# Program for   : 1. from a card show what is the suit and number
#                 # Date          : Nov 2nd, 2023
# input         : image of card taken by self
# Requirement   : cv2. numpy, matplotlib
#####################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt


def take_image(output_name = 'captured_image.jpg'):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (usually the built-in webcam)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
    else:
        # Capture a single frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not capture a frame.")
        else:
            # Save the captured frame as an image file
            cv2.imwrite(output_name, frame)
            print("Image saved as 'captured_image.jpg'")

        # Release the webcam
        cap.release()

    # Close any OpenCV windows that may have opened during the process
    cv2.destroyAllWindows()


folder = 'cards'

# image = cv2.imread(folder + '/' + '10_of_clubs.png')
image = cv2.imread('reconstructed_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image[image <= 125] = 0
image[image > 125] = 255
image = 255 - image
L, W = image.shape


def find_block_center(img, padding_start=15, cube_len=5):  # padding is used to ignore the boundary edge of the card
    L, W = img.shape
    for i in range(padding_start, L - padding_start):
        for j in range(padding_start, W - padding_start):
            if np.all(img[i:i + cube_len, j:j + cube_len] == 255):
                return i + 1, j + 1


# Finding number start from here
y1, x1 = find_block_center(image)
print(y1, x1)

image_B = np.zeros(image.shape, dtype=np.uint8)  # create a same size image like the card
image_B[y1, x1] = 255  # now create a dot in the canvas to start my dilation
# Define the kernel to connect points (8-connectivity)
kernel = np.array([[255, 255, 255],
                   [255, 255, 255],
                   [255, 255, 255]], dtype=np.uint8)

while True:
    image_B_new = cv2.dilate(image_B, kernel, iterations=1)
    image_B_new = np.logical_and(image, image_B_new).astype(np.uint8) * 255
    if np.array_equal(image_B, image_B_new):
        break
    else:
        image_B = image_B_new

# The below part will only be activated when card number is 10
##################################################################
# since in 10 card 0 could be 8-10 pixel right of 1 ahead of
image_B_new = cv2.dilate(image_B, kernel, iterations=6)
image_B_new = np.logical_and(image, image_B_new).astype(np.uint8) * 255

# if they are not equal it means it found 0 close by 1
if not np.array_equal(image_B, image_B_new):
    image_B = image_B_new

    while True:
        image_B_new = cv2.dilate(image_B, kernel, iterations=1)
        image_B_new = np.logical_and(image, image_B_new).astype(np.uint8) * 255
        if np.array_equal(image_B, image_B_new):
            break
        else:
            image_B = image_B_new

########################################################################

# now lets remove the number from the actual card
image_S = image - image_B

# Finding suite start from here
y1, x1 = find_block_center(image_S)
print(y1, x1)

image_C = np.zeros(image_S.shape, dtype=np.uint8)  # create a same size image like the card
image_C[y1, x1] = 255  # now create a dot in the canvas to start my dilation

while True:
    image_C_new = cv2.dilate(image_C, kernel, iterations=1)
    image_C_new = np.logical_and(image_S, image_C_new).astype(np.uint8) * 255
    if np.array_equal(image_C, image_C_new):
        break
    else:
        image_C = image_C_new

plt.title('card')
plt.imshow(image, cmap="gray")
plt.waitforbuttonpress()
plt.close()
plt.imshow(image_B, cmap="gray")
plt.waitforbuttonpress()
plt.close()
plt.imshow(image_S, cmap="gray")
plt.waitforbuttonpress()
plt.close()
plt.imshow(image_C, cmap="gray")
plt.waitforbuttonpress()
plt.close()
