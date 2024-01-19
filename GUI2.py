import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load the two images
img1 = cv2.imread("sample_images/img_1_2.jpg")
img2 = cv2.imread("sample_images/img_1_1.jpg")

# Resize images if necessary
img1 = cv2.resize(img1, (700, 480))
img2 = cv2.resize(img2, (700, 480))

img_height = img1.shape[0]

# Grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(similar, diff) = ssim(gray1, gray2, full=True)
print("Level of similarity : {}".format(similar))

diff = (diff * 255).astype("uint8")
cv2.imshow("Difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()