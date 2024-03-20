from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('iris_model.pkl')
le = joblib.load('label_encoder.pkl')

def predict_output(inputData):
    # import numpy as np
    # import joblib
    # le = joblib.load('label_encoder.pkl')
    # model = joblib.load('iris_model.pkl')
    inputData = np.array(inputData).reshape(1, -1)
    output = model.predict(inputData)
    class_name = le.inverse_transform(output)
    return class_name[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get the input features from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = predict_output(features)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
