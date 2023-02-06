from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data = request.form
    a = float(data['a'])
    b = int(data['b'])
    c = float(data['c'])

    arr = np.array([[a, b, c]])
    pred = model.predict(arr)
    print('hello')
    return render_template('after.html', Charges=np.round(pred[0],2))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)   

