import  pandas as pd
import  numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing  import  MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"heart_disease.csv")
print(df.head())

#Check the Target Variable Balance We need to see how many people have Heart Disease Status = Yes vs No. Plot Bar chart (countplot)  If one class is much smaller than the other, the model may become biased.
sns.countplot(x='Heart Disease Status', data=df)
plt.title("Heart Disease Count")
plt.show()