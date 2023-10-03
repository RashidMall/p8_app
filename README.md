# Image Segmentation Web App  

https://opencalssrooms-p8-app-c8f9ec2b822a.herokuapp.com/  

This is a Flask-based web application for performing image segmentation using a deep learning model. The app allows users to upload an image, and it provides predictions for the segmentation mask.

## Features

- User-friendly web interface for image segmentation.
- Utilizes a deep learning model to provide accurate mask predictions.
- Supports multiple image formats.
- Displays visual comparisons between the original image, predicted mask, and actual mask.

## Requirements

- Python 3.7+
- Flask
- NumPy
- pandas
- OpenCV (cv2)
- Matplotlib
- Base64
- Requests

You can find the complete list of requirements in the `requirements.txt` file.

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/image-segmentation-web-app.git
cd image-segmentation-web-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask application locally:
```bash
python app.py
```

4. The app will be accessible at http://127.0.0.1:5050/

## Web Interface  

* Access the web interface by opening a web browser and navigating to http://127.0.0.1:5050/
* It utilizes a deep learning model hosted on a remote API server to predict segmentation masks.
