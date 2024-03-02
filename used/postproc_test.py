import cv2
import numpy as np

# Load the image
image = cv2.imread('temp.png')  # Replace 'your_image_path.jpg' with the actual path to your image

# Convert the image to greyscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use adaptive thresholding to make background black and lines white
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Display the original and processed images
cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Apply erosion to thin the lines  # not looking good
kernel = np.ones((3, 3), np.uint8)
eroded_image = cv2.erode(thresholded_image, kernel, iterations=1)

# Apply dilation to thicken the lines
kernel = np.ones((2, 2), np.uint8)
dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

# Display the eroded and dilated images
cv2.imshow('Eroded Image', eroded_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 