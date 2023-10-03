from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob
import io
import cv2
import base64
import requests
from collections import namedtuple

# URL of the remote API server for mask prediction.
SERVER_URL = "https://openclassrooms-p8-api-3ef27b2812fb.herokuapp.com/predict_mask"

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 128
NUMBER_CLASSES = 8

img_names = glob.glob('./data/image/*.png')
mask_names = glob.glob('./data/mask/*_labelIds.png')

img_names.sort()
mask_names.sort()
df = pd.DataFrame({'img': img_names, 'mask': mask_names})

Label = namedtuple('Label', ['name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color',])

labels = [
	#       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
	Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
	Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
	Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
	Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
	Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
	Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
	Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
	Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
	Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
	Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
	Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
	Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
	Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
	Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
	Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
	Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
	Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
	Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
	Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
	Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
	Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
	Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
	Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
	Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
	Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
	Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
	Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
	Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
	Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
	Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
	Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
	Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
	Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
	Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
	Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

category_label_mapping = {l.id : l.categoryId for l in labels}

def preprocess_image(path):
    """
    Load, resize, and normalize an image.

    Args:
        path (str): Path to the image file.

    Returns:
        numpy.ndarray: The preprocessed image as a NumPy array.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.astype(np.float32) / 255.0
    
    return image


def preprocess_mask(path):
    """
    Preprocess and one-hot encode a segmentation mask image.

    Args:
        path (str): Path to the mask image file.

    Returns:
        numpy.ndarray: The preprocessed and one-hot encoded mask as a NumPy array.
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT))
	
    label_mask = np.zeros_like(mask)
    
    for category_id in category_label_mapping:
        label_mask[mask == category_id] = category_label_mapping[category_id]
    
    # One-hot encode the label mask
    label_mask = np.eye(NUMBER_CLASSES)[label_mask]
    
    return label_mask


def get_segmentation_mask(image_path):
    """
    Get a segmentation mask from an image using a remote server.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        numpy.ndarray: The segmentation mask as a NumPy array.
    """
    # Read the input image
    image = cv2.imread(image_path)

    # Encode the image as a base64 string
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes)

    # Send a request to the remote server
    json_content = {"image": img_base64.decode('utf-8')}
    response = requests.post(SERVER_URL, json=json_content)

    # Decode the mask from the response
    mask_base64 = response.json()["mask"]
    mask = base64.b64decode(mask_base64)
    mask = np.frombuffer(mask, dtype=np.float32)
    mask = mask.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 8)

    return mask


def create_segmentation_figure(img, pred_mask, mask):
    """
    Create a Matplotlib figure with subplots for image visualization.

    Args:
        img (numpy.ndarray): Original RGB image.
        pred_mask (numpy.ndarray): Predicted segmentation mask.
        mask (numpy.ndarray): Original segmentation mask.

    Returns:
        str: Base64-encoded image data.
    """
    # Create subplots for visualization
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    # Display original RGB image
    axs[0].imshow(img)
    axs[0].set_title("Original RGB Image")
    axs[0].grid(False)

    # Display predicted mask
    axs[1].imshow(np.argmax(pred_mask, axis=-1), cmap='gist_rainbow')
    axs[1].set_title('Predicted Mask')
    axs[1].grid(False)

    # Display original mask
    axs[2].imshow(np.argmax(mask, axis=-1), cmap='gist_rainbow')
    axs[2].set_title("Original Mask")
    axs[2].grid(False)

    # Convert the Matplotlib figure to a base64-encoded image
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode()

    return img_data


app = Flask(__name__, static_url_path='/')
app.secret_key = 'secret_key'

@app.route('/')
def index():
	print('index')
	return render_template('index.html', img_list=df['img'].values)


@app.route('/predicted', methods=['POST', 'GET'])
def predicted():
    if request.method == 'POST':
        image_path = request.form['option']
        index_df = df[df['img'] == image_path].index[0]
        mask_path = df['mask'][index_df]
		
        img = preprocess_image(image_path)
        mask = preprocess_mask(mask_path)
        pred_mask = get_segmentation_mask(image_path)

        img_data = create_segmentation_figure(img, pred_mask, mask)

        return render_template('index.html', img_list=df['img'].values, img_data=img_data)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5050)