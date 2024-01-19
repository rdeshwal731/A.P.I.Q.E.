import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity as ssim

diff = None

def update_threshold(x):
    global threshold
    threshold = x
    update_display(threshold)


def update_display(threshold):
    global diff  # Access the global diff variable
    # Apply the threshold with the updated value
    difference = (diff * 255).astype("uint8")
    _, thresh = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY_INV)

    # Create an image with saturated blue color
    displayed_image = np.zeros_like(img1)
    displayed_image[:, :, 0] = 255  # Set blue channel to 255 (full blue)

    # Create a binary mask for the differences
    diff_mask = np.zeros_like(thresh)
    diff_mask[thresh > 0] = 1

    # Create an image with the differences in red
    displayed_image[diff_mask == 1] = [0, 0, 255]

    # Blend the displayed image with the result image (original image with green edge contours)
    alpha = 0.2
    result_image = img1.copy()
    cv2.addWeighted(result_image, alpha, displayed_image, 1 - alpha, 0, result_image)

    # Draw green edge contours on the result image
    #cv2.drawContours(result_image, edge_contours, -1, (0, 255, 0), 1)

    # Add the similarity score as text on the image
    cv2.putText(result_image, f"Similarity Score: {similar:.2f}", (10, img_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 4,
                (0, 255, 0), 5)

    # Show the result image with real-time threshold adjustment
    result_image = cv2.resize(result_image, (560,700))
    cv2.imshow("Result Image with Overlaid Differences and Edge Contours", result_image)


# Load the two images
img1 = cv2.imread("/home/dell/Pictures/6.jpg")
img2 = cv2.imread("/home/dell/Pictures/portrait/6.jpg")

# Resize images if necessary
#img1 = cv2.resize(img1, (700, 480))
#img2 = cv2.resize(img2, (700, 480))

img_height = img1.shape[0]

# Grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(similar, diff) = ssim(gray1, gray2, full=True)
difference = cv2.absdiff(gray1, gray2)
# Initialize the threshold value
threshold = 100  # You can set an initial value


# Apply Canny edge detection
#edges = cv2.Canny(gray1, 100, 200)

# Create a mask to focus on edges near the differences
#masked_edges = cv2.bitwise_and(edges, threshold)

# Calculate contours for masked edges
#edge_contours = cv2.findContours(masked_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#edge_contours = imutils.grab_contours(edge_contours)

# Create a window with a trackbar for threshold adjustment
cv2.namedWindow("Result Image with Overlaid Differences and Edge Contours")
cv2.createTrackbar("Threshold", "Result Image with Overlaid Differences and Edge Contours", threshold, 255,update_threshold)

update_display(threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
