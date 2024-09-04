import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img, img_to_array
# from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

app = Flask(__name__)

# Set the upload folder inside the root directory
ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_FOLDER, 'static', 'usr-uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Import the model above (Comment/Uncomment)
# model = ResNet50()
model = VGG16()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Decorator for the home-page route
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file.save(filepath)

            image = load_img(filepath, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)

            yhat = model.predict(image)
            label = decode_predictions(yhat)
            label = label[0][0]
            prediction = label[1]
            probability = "{:.2f}".format(label[2] * 100)

            return render_template('index.html', image_path=url_for('static', filename=f'usr-uploads/{filename}'), prediction=prediction, probability=probability)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
