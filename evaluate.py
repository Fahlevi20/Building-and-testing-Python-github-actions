from preprocessing import process_data
import joblib
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

X_train,X_test,y_train,y_test = process_data()
model = joblib.load("model.pkl")

y_pred=model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r_score = r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(f'\nR^2={r_score}\nMean Square Error={mse}\nRoot Mean Square Error={rmse}.')

with open('metrics.txt','w') as outfile:
  outfile.write(f'\nR^2={r_score}\nMean Square Error={mse}\nRoot Mean Square Error={rmse}.')
