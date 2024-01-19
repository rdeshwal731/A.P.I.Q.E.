import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image):
    # Resize the image to the required input dimensions of the ResNet model
    image = cv2.resize(image, (224, 224))

    # Convert the image to an RGB image
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Convert the image to an array
    image = img_to_array(image)

    # Expand the dimensions of the image to match the expected input shape
    image = np.expand_dims(image, axis=0)

    # Preprocess the image
    image = preprocess_input(image)

    return image


def get_feature_vector(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Extract the feature vectors using the ResNet model
    features = model.predict(image)
    # Flatten the feature vectors
    features = features.flatten()
    return features

# Load the pre-trained ResNet model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Load the images
image_path1 = "C:/Users/Dell/PycharmProjects/portrait_image_compare/sample_images/img_1_1.jpg"
image_path2 = "C:/Users/Dell/PycharmProjects/portrait_image_compare/sample_images/img_1_2.jpg"

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# Apply artificial blur to one of the images
blurred_image = cv2.GaussianBlur(image2, (15, 15), 0)

# Calculate the feature vectors for the original and blurred images
features1 = get_feature_vector(image1)
features2 = get_feature_vector(blurred_image)

# Calculate the difference between the feature vectors
difference = np.abs(features1 - features2)

# Reshape the difference array to match the image dimensions
difference_image = np.reshape(difference, (224, 224, -1))

# Convert the difference image to a single-channel grayscale image
grayscale_difference = cv2.cvtColor(difference_image, cv2.COLOR_BGR2GRAY)

# Normalize the difference image to the range [0, 255]
normalized_difference = cv2.normalize(grayscale_difference, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the difference image
cv2.imshow("Difference Image", normalized_difference)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the difference image
cv2.imwrite("difference_image.jpg", normalized_difference)
print("Difference image saved as difference_image.jpg")





