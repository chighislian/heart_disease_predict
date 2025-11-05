import  pandas as pd
import  numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing  import  MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder , LabelEncoder

df = pd.read_csv(r"heart_disease.csv")
print(df.head())