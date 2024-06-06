# import dataset
# Import libraries
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
def EDA(dataset):
  
  print("================= Exploratory Data Analysis =================")
  print("\n5 rows dataset:", dataset.head())
  print("=============================================================")
  print("\nSum of total columns:", len(dataset.columns))
  print("=============================================================")
  print("\ndataset info:", dataset.info())
  print("=============================================================")
  print("\nDescribe of data:", dataset.describe())
  print("=============================================================")
  print(sns.heatmap(dataset.corr(), annot=True))

  return dataset.head()

EDA(data)
