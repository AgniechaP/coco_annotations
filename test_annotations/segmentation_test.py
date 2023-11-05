import json
import cv2
import numpy as np
import os

# Load the COCO JSON file
# mask json (mask from the Internet)
# coco_data = json.load(open('mask_test_annotation.json'))

# kefir, gum, tissue
coco_data = json.load(open('test_trash-1.json'))

# UAVVaste dataset
# coco_data = json.load(open('/home/agnieszka/Documents/UAVVaste/annotations/annotations.json'))


# Directory where the images are stored
image_directory = '/home/agnieszka/Documents/coco-annotator/datasets/test_trash/'

# image_directory = '/home/agnieszka/Documents/UAVVaste/images/'


current_index = 0
cv2.namedWindow('Image with Segmentation', cv2.WINDOW_NORMAL)
# Loop through images and their corresponding annotations
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)

    if image_info:
        image_path = os.path.join(image_directory, image_info['file_name'])

        # Loading the image using OpenCV
        image = cv2.imread(image_path)

        # Getting the segmentation points for the annotation
        segmentation = annotation['segmentation'][0]

        segmentation = np.array(segmentation, np.int32)
        segmentation = segmentation.reshape((-1, 1, 2))

        # Drawing the segmentation mask on the image
        color = (0, 0, 255)
        image = cv2.polylines(image, [segmentation], isClosed=True, color=color, thickness=2)

        desired_width = 800
        aspect_ratio = image.shape[1] / image.shape[0]
        desired_height = int(desired_width / aspect_ratio)

        cv2.resizeWindow('Image with Segmentation', desired_width, desired_height)

        cv2.imshow('Image with Segmentation', image)

        # Wait for a key event and check if the Esc key (key code 27) is pressed
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('n'):  # If 'N' key is pressed, go to the next image
            current_index += 1

cv2.destroyAllWindows()
