from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
 
data = pd.read_csv("delivery_data.csv")
print(data.head())

app = Flask(__name__)

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        distance = float(data.get('distance_km'))
        product = data.get('product_category')
        priority = data.get('priority')
        shipping = data.get('shipping_mode')
        weather = data.get('weather')
        stock = data.get('stock_available')

        # Fix stock value
        if stock == "In Stock":
            stock = 1
        else:
            stock = 0

        # Encode categorical values
        product = encoders['product_category'].transform([product])[0]
        priority = encoders['priority'].transform([priority])[0]
        shipping = encoders['shipping_mode'].transform([shipping])[0]
        weather = encoders['weather'].transform([weather])[0]

        # Create input array
        input_data = np.array([[distance, product, priority, shipping, weather, stock]])

        # Predict
        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction=round(prediction, 2))

   
    except Exception as e:
        return render_template("index.html", prediction="Error in input values")
if __name__ == "__main__":
    app.run(debug=True)
