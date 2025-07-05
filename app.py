from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    hours = float(request.form['hours'])
    prediction = model.predict([[hours]])
    return render_template('index.html', prediction=f"Predicted Score: {prediction[0]:.2f}")

if __name__ == "__main__":
    app.run()
