import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import requests
import base64
import re
from config import VISION_KEY

API_ENDPOINT = 'https://vision.googleapis.com/v1/images:annotate?key=' + VISION_KEY

def read_plate(img_path):
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray, cmap='gray') 
    plt.title('Grayscale Image')
    plt.axis('off')

    # Apply bilateral filter to reduce noise while preserving edges
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Parameters: (source, diameter, sigmaColor, sigmaSpace)

    edged = cv2.Canny(bfilter, 30, 200)  # Parameters: (source, threshold1, threshold2)

    plt.imshow(edged, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')

    # Find contours in the edge-detected image
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)  # Extract contours from keypoints

    # Sort contours by area in descending orde
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None

    for contour in contours:
        # Approximate the contour to a polygon with fewer vertices
        approx = cv2.approxPolyDP(contour, 10, True)  # Parameters: (contour, epsilon, closed)

        # Check if the approximated polygon has 4 vertices
        if len(approx) == 4:
            location = approx  # Store the contour as the detected location
            break  # Exit the loop once the first quadrilateral is found

    print(location)

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Contour')
    plt.axis('off')

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    cropped_image_path = 'cropped_image.jpg'
    cv2.imwrite(cropped_image_path, cropped_image)

    plt.imshow(cropped_image, cmap='gray')
    plt.title('Cropped Image')
    plt.axis('off')

    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image_to_base64(cropped_image_path)

    request_payload = {
        "requests": [
            {
                "image": {
                    "content": base64_image
                },
                "features": [
                    {
                        "type": "TEXT_DETECTION"
                    }
                ]
            }
        ]
    }
    response = requests.post(API_ENDPOINT, json=request_payload)
    response_data = response.json()

    if 'responses' in response_data and len(response_data['responses']) > 0:
        texts = response_data['responses'][0].get('textAnnotations', [])
        text = texts[0]['description'] if texts else 'No text found'
    else:
        text = None

    if text:
        split_text=text.split("\n")
        PLATE_NUMBER=split_text[-1]
        print(f'Plate DETECTED: {PLATE_NUMBER}')
        return PLATE_NUMBER.replace("-",'')
    return ""

read_plate('image4.jpg')