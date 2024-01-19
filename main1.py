import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity

# Load the two images
img1 = cv2.imread("sample_images/img_4_1.jpg")
img2 = cv2.imread("sample_images/img_4_2.jpg")

# Resize images if necessary
img1 = cv2.resize(img1, (700,480))
img2 = cv2.resize(img2, (700,480))

img_height = img1.shape[0]

# Grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(similar, diff) = structural_similarity(gray1, gray2, full=True)
print("Level of similarity : {}".format(similar))

diff = (diff*255).astype("uint8")
cv2.imshow("Difference", diff)
# Apply threshold. Apply both THRESH_BINARY_INV and THRESH_OTSU
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("Threshold", thresh)

cv2.imshow("Threshold", thresh)


# Calculate contours
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

"""for contour in contours:
    # Calculate bounding box around contour
    if cv2.contourArea(contour) > 5:
        x, y, w, h = cv2.boundingRect(contour)
        # Apply Gaussian blur to the differing region
        blurred_region = cv2.GaussianBlur(img1[y:y+h, x:x+w], (25, 25), 0)
        img1[y:y + h, x:x + w] = blurred_region"""

# Create a copy of the original image to draw contours on
contour_image = img1.copy()

# Draw contours on the image
cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)  # You can adjust the color and thickness

# Convert the similarity score to a formatted string
score_text = f"Similarity Score: {similar:.2f}"

# Add the similarity score as text on the image
cv2.putText(contour_image, score_text, (10, img_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show the modified image with blurred differences
cv2.imshow("Blurred Differences", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




