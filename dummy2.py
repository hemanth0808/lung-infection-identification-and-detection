import io
import base64
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
app = Flask(__name__)
# Load the UNet++ model
# create a new instance of the optimizer

unetpp = load_model('my_model(100).h5', compile=False)

# Define a function to perform lung infection segmentation
def segment_lung_infection(image):
    # Preprocess the image
    img = image.convert('L')
    img = img.resize((224, 224))
    img_array = np.asarray(img) / 255.0
    img_array = np.reshape(img_array, (1, 224, 224, 1))   
    # Make a prediction using the UNet++ model
    prediction = unetpp.predict(img_array)    
    # Postprocess the prediction
    segmented_image = np.squeeze(prediction) > 0.5    
    # Convert the segmented image to a PIL Image object
    segmented_image = Image.fromarray(np.uint8(segmented_image * 255))
    segmented_image = segmented_image.resize((image.width, image.height))   
    return segmented_image

# Define a route to handle the image upload form
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file from the form
        file = request.files['file']
        if file:
            # Open the file as a PIL Image object
            image = Image.open(file)

            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str_input = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Perform lung infection segmentation
            segmented_image = segment_lung_infection(image)
            
            # Convert the segmented image to base64 format
            buffered = io.BytesIO()
            segmented_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Render the result template with the segmented image
            return render_template('dummyindex.html',img_str_input=img_str_input, img_str=img_str)
    # Render the upload form template by default
    return render_template('dummyindex.html')

if __name__ == '__main__':
    app.run(debug=True)
