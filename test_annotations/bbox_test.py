import json
import cv2
import numpy as np
import os

# Load the COCO JSON file
# Photos of: kefir, gum, tissue
coco_data = json.load(open('test_trash-1.json'))
# Photo of mask from the Internet
# coco_data = json.load(open('mask_test_annotation.json'))

# Directory where the images are stored
image_directory = '/home/agnieszka/Documents/coco-annotator/datasets/test_trash/'

current_index = 0
window_width = 800
window_height = 600

# Named window for displaying images
cv2.namedWindow('Image with Bounding Box', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image with Bounding Box', window_width, window_height)

# Loop through images and their corresponding annotations
for annotation in coco_data['annotations']:
    image_id = annotation['image_id']
    image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)

    if image_info:
        image_path = os.path.join(image_directory, image_info['file_name'])

        # Loading the image using OpenCV
        image = cv2.imread(image_path)

        # Get bounding box coordinates
        x, y, width, height = map(int, annotation['bbox'])
        original_height, original_width, _ = image.shape
        scale_x = window_width / original_width
        scale_y = window_height / original_height

        image = cv2.resize(image, (window_width, window_height))
        x = int(x * scale_x)
        y = int(y * scale_y)
        width = int(width * scale_x)
        height = int(height * scale_y)

        x2, y2 = x + width, y + height

        # h, w, _ = image.shape
        # if h > 600 or w > 600:
        #     scale = min(600 / w, 600 / h)
        #     new_width = int(w * scale)
        #     new_height = int(h * scale)
        #     image = cv2.resize(image, (new_width, new_height))

        # Drawing the bounding box on the image
        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(image, (x, y), (x2, y2), color, thickness)

        # Display the image with the bounding box
        cv2.imshow('Image with Bounding Box', image)

        key = cv2.waitKey(0)
        if key == 27:  # Check if the Esc key (key code 27) is pressed
            break
        elif key == ord('n'):  # If 'N' key is pressed, go to the next image
            current_index += 1

    # Release the OpenCV window and close it
    cv2.destroyAllWindows()
