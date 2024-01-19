'''import cv2
import numpy as np
def align_images(image1, image2):
  # Resize images if necessary
  image1 = cv2.resize(image1, (700, 480))
  image2 = cv2.resize(image2, (700, 480))

  # Convert images to grayscale
  gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
  gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

  # Perform feature detection and matching
  orb = cv2.ORB_create()
  keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(descriptors1, descriptors2)
  matches = sorted(matches, key=lambda x: x.distance)

  # Estimate homography matrix
  src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
  dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

  M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

  # Warp one image to align with the other
  aligned_image2 = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

  return aligned_image2

# Load the images.
img1 = cv2.imread("/home/dell/Pictures/grnd_trth/18.jpg")
img2 = cv2.imread("/home/dell/Pictures/portrait/18.jpg")

# Align the images.
aligned_img = align_images(img1, img2)

# Display the aligned images.
cv2.imshow('Aligned image', aligned_img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
import numpy as np
import cv2  # You may need OpenCV to load and display images

# Load your two images
image1 = cv2.imread("/home/dell/Pictures/5.jpg")
image2 = cv2.imread("/home/dell/Pictures/portrait/5.jpg")

image1 = cv2.resize(image1, (700, 480))
image2 = cv2.resize(image2, (700, 480))

# Ensure that the images have the same dimensions
if image1.shape == image2.shape:
    # Compute the difference image
    difference = cv2.absdiff(image1, image2)

    # Display or save the difference image
    cv2.imshow('Difference Image', difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("The images must have the same dimensions for subtraction.")
