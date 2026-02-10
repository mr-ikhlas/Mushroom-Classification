from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# load real model
model = load_model("Mushroom Classification Model.h5", compile=False)



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/Mushroom-classification-predict", methods=["GET", "POST"])
def predict():

    predicted_class = None
    image_name = None

    if request.method == "POST":

        imageFile = request.files["image_file"]

        if imageFile:
            image_name = imageFile.filename
            imagePath = os.path.join(UPLOAD_FOLDER, image_name)
            imageFile.save(imagePath)

            img = image.load_img(imagePath, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            prediction = model.predict(img)
            class_names = ['Boletus', 'Lactarius', 'Russula']

            predicted_class = class_names[np.argmax(prediction)]

    return render_template(
        "input.html",
        prediction=predicted_class,
        image=image_name
    )


if __name__ == "__main__":
    app.run(debug=True)
