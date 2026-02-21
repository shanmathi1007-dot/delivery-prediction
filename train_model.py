import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample dataset
data = pd.DataFrame({
    "distance_km": np.random.randint(10, 1000, 500),
    "product_category": np.random.choice(["Electronics", "Clothing", "Home"], 500),
    "priority": np.random.choice(["Low", "Medium", "High"], 500),
    "shipping_mode": np.random.choice(["Standard", "Express"], 500),
    "weather": np.random.choice(["Clear", "Rain", "Storm"], 500),
    "stock_available": np.random.choice([0, 1], 500),
    "delivery_time_days": np.random.randint(1, 15, 500)
})

encoders = {}

for col in ["product_category", "priority", "shipping_mode", "weather"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data.drop("delivery_time_days", axis=1)
y = data["delivery_time_days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model Ready!")
