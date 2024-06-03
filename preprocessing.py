# import dataset
# Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

X = data.drop("PRICE", axis=1)
y = data["PRICE"]

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=42)

scaler = StandarScaler()
X_train = scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
