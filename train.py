import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from processing import process_data
import joblib
process_data()

model = LinearRegression()
model.fit(process_data.X_train, process_data.y_train)

joblib.dump(model,"model.pkl")
