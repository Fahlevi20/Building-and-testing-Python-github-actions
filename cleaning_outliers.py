import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target
import pandas as pd

def identify_outliers_columns(df):
    outliers = pd.DataFrame(columns=df.columns)
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers_in_column = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
        outliers = pd.concat([outliers, outliers_in_column])
    return outliers.drop_duplicates()

# Assuming 'data' DataFrame is defined

df = data.copy()

outliers = identify_outliers_columns(df)
print(f'Number of outliers found: {outliers.shape[0]}')
print(outliers)

df_cleaned = df.drop(outliers.index)
df_cleaned = df_cleaned.dropna()  # Corrected line: Drop NaN values from the cleaned DataFrame
print(df_cleaned)
print(f"Number of outliers found: {outliers.shape[0]}")

X_cleaned = df_cleaned.drop(columns=['PRICE']).values
y_cleaned = df_cleaned['PRICE'].values
#training data outliers cleaned
X_train_new,X_test_new,y_train_new,y_test_new = train_test_split(X_cleaned,y_cleaned, random_state=42)

scaler = StandardScaler()
X_train_new = scaler.fit_transform(X_train_new)
X_test_new=scaler.fit_transform(X_test_new)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

model_new = LinearRegression()
model_new.fit(X_train_new, y_train_new)

joblib.dump(model_new,"model_new.pkl")

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
print("before clean", df.shape[0])
print("after clean", df_cleaned.shape[0])
print("before clean", X.shape[0])
print("after clean", X_cleaned.shape[0])
