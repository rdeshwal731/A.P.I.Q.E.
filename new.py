import cv2
import numpy as np

def image_difference(image1, image2):
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("not loaded")
        return None

    diff = cv2.absdiff(img1, img2)

    # Threshold the difference image
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw contours on the original image
    result = cv2.drawContours(cv2.imread(image1), contours, -1, (0, 0, 255), 2)

    cv2.imshow("Difference Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return diff


image_path1 = "sample_images/img_4_1.jpg"
image_path2 = "sample_images/img_4_2.jpg"

difference_image = image_difference(image_path1, image_path2)

if difference_image is not None:
    cv2.imwrite("difference_image4.jpg", difference_image)
    print("Difference image saved as difference_image.jpg")
