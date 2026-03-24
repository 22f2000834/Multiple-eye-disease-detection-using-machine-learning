from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image

app = Flask(__name__)

model_path = r'C:\Users\HP\Desktop\eye_detection\model.h5'
loaded_model = load_model(model_path)

disease_descriptions = {
    'Cataract': "Cataract is a clouding of the eye's lens, which leads to blurred vision. It's often age-related but can also result from injury or other medical conditions. Symptoms include blurry or foggy vision, difficulty seeing at night, sensitivity to light, and seeing halos around lights.",
    'Diabetic Retinopathy': "Diabetic retinopathy is a complication of diabetes that affects the eyes. It occurs when high blood sugar levels damage the blood vessels in the retina, leading to vision problems. Symptoms include blurred vision, floaters, difficulty seeing at night, and vision loss.",
    'Glaucoma': "Glaucoma is a group of eye conditions that damage the optic nerve, often due to increased pressure in the eye. It can lead to vision loss or blindness if left untreated. Symptoms may not appear until the condition is advanced, but they can include blurred vision, eye pain, headache, and seeing halos around lights.",
    'Normal': "In the context of your model, 'Normal' indicates that no signs of eye diseases were detected in the uploaded image. This result suggests that the eyes appear healthy without any abnormalities or conditions requiring immediate attention.",
    'Other': "'Other' typically refers to any eye condition not specifically categorized by the model. It could include less common eye diseases or conditions that the model hasn't been trained to identify. Further evaluation by an eye care professional may be necessary to determine the specific issue."
}

@app.route("/", methods=["GET"])
def home():
    return render_template('home.html')

@app.route("/faq", methods=["GET"])
def faq():
    return render_template('faq.html')

@app.route("/symptoms", methods=["GET"])
def symptoms():
    return render_template('symptoms.html')

@app.route("/diagnosis", methods=["GET"])
def diagnosis():
    predicted_class = request.args.get('predicted_class')
    probability = request.args.get('probability')
    image_path = request.args.get('image_path')
    description = disease_descriptions.get(predicted_class, "Description not available")
    symptoms = {
        'blurredVision': request.args.get('blurredVision'),
        'halosAroundLights': request.args.get('halosAroundLights'),
        'fluctuatingVision': request.args.get('fluctuatingVision'),
        'darkSpotsFloaters': request.args.get('darkSpotsFloaters'),
        'gradualPeripheralVisionLoss': request.args.get('gradualPeripheralVisionLoss'),
        'severeEyePain': request.args.get('severeEyePain')
    }
    return render_template('diagnosis.html', predicted_class=predicted_class, probability=probability,
                           image_path=image_path, disease_description=description, symptoms=symptoms)

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    static_dir = os.path.join(app.root_path, 'static')
    uploaded_image_path = os.path.join(static_dir, 'uploaded_image.jpg')

    img = Image.open(file)
    img = img.resize((256, 256))
    img.save(uploaded_image_path)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = loaded_model.predict(img_array)

    classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal', 'Other']

    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    probability = prediction[0][predicted_class_index] * 100

    symptoms = {
        'blurredVision': request.form.get('blurredVision'),
        'halosAroundLights': request.form.get('halosAroundLights'),
        'fluctuatingVision': request.form.get('fluctuatingVision'),
        'darkSpotsFloaters': request.form.get('darkSpotsFloaters'),
        'gradualPeripheralVisionLoss': request.form.get('gradualPeripheralVisionLoss'),
        'severeEyePain': request.form.get('severeEyePain')
    }

    return redirect(url_for('diagnosis', predicted_class=predicted_class, probability=probability,
                            image_path='uploaded_image.jpg', **symptoms))

if __name__ == "__main__":
    app.run(port="5000")
