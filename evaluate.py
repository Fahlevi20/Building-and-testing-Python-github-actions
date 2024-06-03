from preprocessing import process_data
import joblib
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

X_train,X_test,y_train,y_test = process_data()
model = joblib.load("model.pkl")

y_pred=model.predict(X_test)
print(y_pred)
mse = mean_squared_error(y_test,y_pred)
print(mse)
r_score = r2_score(y_test,y_pred)
print(r_score)
rmse = root_mean_squared_error(y_test,y_pred)
print(rmse)
