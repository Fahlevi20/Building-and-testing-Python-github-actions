import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preprocessing import process_data
import joblib
X_train,X_test,y_train,y_test=process_data()

model = LinearRegression()
model.fit(process_data.X_train, process_data.y_train)

joblib.dump(model,"model.pkl")
