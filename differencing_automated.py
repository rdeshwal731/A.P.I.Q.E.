import cv2
import os
import imageio
import csv
import imutils
import numpy as np
from skimage.metrics import structural_similarity as ssim


def image_comparison_with_visualization(image_1, image_2, output_folder):

    file_name = os.path.splitext(os.path.basename(image_1))[0]

    # Load the two images
    image_1 = cv2.imread(image_1)
    image_2 = cv2.imread(image_2)

    # Resize images if necessary
    image_1 = cv2.resize(image_1, (2000, 3000))
    image_2 = cv2.resize(image_2, (2000, 3000))

    img_height = image_1.shape[0]

    # Grayscale
    gray1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    (similar, diff) = ssim(gray1, gray2, full=True)
    #print("Level of similarity : {}".format(similar))

    diff = (diff * 255).astype("uint8")
    diff = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #Create an image with saturated blue color
    saturated_blue_image = np.zeros_like(image_1)
    saturated_blue_image[:, :, 0] = 255  # Set blue channel to 255 (full blue)
    saturated_blue_image[:, :, 1] = 0    # Set green channel to 0
    saturated_blue_image[:, :, 2] = 0    # Set red channel to 0

    # Keep only the differences as red
    saturated_blue_image[diff > 0] = [0, 0, 255]  # Set red channel to 255 where differences are present

    # Calculate contours
    contours = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Create a copy of the saturated blue image to draw contours on
    contour_image = saturated_blue_image.copy()

    # Draw contours on the image
    cv2.drawContours(contour_image, contours, -1, (0, 0, 0), 1)  # Draw contours in black color

    result_image = image_1.copy()
    # Blending the saturated blue image with the result image
    alpha = 0.0  # Adjust the alpha (transparency) value as needed
    cv2.addWeighted(result_image, alpha, saturated_blue_image, 1 - alpha, 0, result_image)


    # Show the modified image with saturated blue color and red differences
    #cv2.imshow("Saturated Blue with Red Differences", result_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Extract file name without extension
    #file_name = os.path.splitext(os.path.basename(image_1))[0]

    # Save the modified image with saturated blue color and red differences
    output_path = os.path.join(output_folder, f"{file_name}_output.jpg")
    cv2.imwrite(output_path, result_image)

def calculate_ssim(image1, image2):
    # Load the images
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    # Resize images if necessary
    image1 = cv2.resize(image1, (1500, 2000))
    image2 = cv2.resize(image2, (1500, 2000))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM score
    ssim_score, _ = ssim(gray1, gray2, full=True)

    return ssim_score

def calculate_psnr(image1, image2):

    # Load the images
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    # Resize images if necessary
    image1 = cv2.resize(image1, (1500, 2000))
    image2 = cv2.resize(image2, (1500, 2000))

    # Convert images to float32
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Calculate mean squared error
    mse = np.mean((image1 - image2) ** 2)

    # Calculate PSNR
    if mse == 0:
        # PSNR is infinite if images are identical
        psnr_score = float('inf')
    else:
        max_pixel = 255.0
        psnr_score = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr_score

def process_image_pairs(folder1, folder2, output_csv):
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ['Image1', 'Image2', 'SimilarityScore','PSNR Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(folder1):
            if filename.endswith(".jpg"):  # Adjust the file extension as needed
                image1_path = os.path.join(folder1, filename)
                image2_path = os.path.join(folder2, filename)

                if os.path.exists(image2_path):
                    similarity_score = calculate_ssim(image1_path, image2_path)
                    psnr_score = calculate_psnr(image1_path,image2_path)
                    # Visualize differences and save the image
                    image_comparison_with_visualization(image1_path, image2_path, output_visualization)

                    # Write results to CSV
                    writer.writerow({'Image1': filename, 'Image2': filename, 'SimilarityScore': similarity_score, 'PSNR Score' : psnr_score})


if __name__ == "__main__":
    folder1_path = "sample_images/image 1"
    folder2_path = "sample_images/image 2"
    output_csv_path = "sample_images/output.csv"
    output_visualization = "sample_images/Output"

    process_image_pairs(folder1_path, folder2_path, output_csv_path)
