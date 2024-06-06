import joblib
from sklearn.metrics import mean_squared_error, r2_score


model_new = joblib.load("model_new.pkl")

y_pred_new=model_new.predict(X_test_new)
mse = mean_squared_error(y_test_new,y_pred_new)
r_score = r2_score(y_test_new,y_pred_new)
rmse = np.sqrt(mean_squared_error(y_test_new,y_pred_new))
print(f'\nR^2={r_score}\nMean Square Error={mse}\nRoot Mean Square Error={rmse}.')

with open('metrics.txt','w') as outfile:
  outfile.write(f'\nR^2={r_score}\nMean Square Error={mse}\nRoot Mean Square Error={rmse}.')
