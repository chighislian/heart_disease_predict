import  pandas as pd
import  numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing  import  MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt



heart_dat = pd.read_csv(r"heart_disease.csv")
print(heart_dat.head())

# Convert columns with  yes/no categorical values to 1/0
binary_cols = [
    'Smoking', 'Family Heart Disease', 'Diabetes', 'High Blood Pressure',
    'Low HDL Cholesterol', 'High LDL Cholesterol', 'Heart Disease Status'
]

for col in binary_cols:
    heart_dat[col] = heart_dat[col].map({'Yes': 1, 'No': 0})

heart_dat['Gender'] = heart_dat['Gender'].map({'Male': 1, 'Female': 0})
heart_dat['Exercise Habits'] = heart_dat['Exercise Habits'].map({'Low': 0, 'Medium': 1, 'High': 2})
heart_dat['Alcohol Consumption'] = heart_dat['Alcohol Consumption'].map({'Low': 0, 'Medium': 1, 'High': 2})
heart_dat['Stress Level'] = heart_dat['Stress Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
heart_dat['Sugar Consumption'] = heart_dat['Sugar Consumption'].map({'Low': 0, 'Medium': 1, 'High': 2})
features = heart_dat.drop(columns=['Heart Disease Status'])
features.fillna(features.mean(), inplace=True)
heart_dat.update(features)







# Requirements: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib
# pip install scikit-learn imbalanced-learn matplotlib


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---------- assume heart_dat is already loaded and preprocessed as you did ----------
X = heart_dat.drop(columns=['Heart Disease Status'])
y = heart_dat['Heart Disease Status']

# If not defined, uncomment and adapt path:
# heart_dat = pd.read_csv("heart_disease.csv")
# ... (your mapping / fillna code) ...
# X = heart_dat.drop(columns=['Heart Disease Status'])
# y = heart_dat['Heart Disease Status']

RANDOM_STATE = 42

# 1) Split BEFORE resampling
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    stratify=y,
                                                    random_state=RANDOM_STATE)

print("TRAIN SIZE:", X_train.shape, " TEST SIZE:", X_test.shape)
print("Train class counts (before SMOTE):\n", y_train.value_counts())

# 2) Diagnostics: check for constant cols, nulls, dtype issues
def diagnostics(X_df, label):
    print(f"\nDiagnostics for {label}:")
    print(" - dtypes:\n", X_df.dtypes.value_counts().to_dict())
    nunique = X_df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    print(" - constant or single-valued columns:", const_cols)
    print(" - missing values per column (top 5):\n", X_df.isna().sum().sort_values(ascending=False).head(5))
    print(" - sample min/max for numeric columns:\n", X_df.select_dtypes(include=[np.number]).agg(['min','max']).T.head(10))

diagnostics(X_train, "X_train (raw)")
diagnostics(X_test, "X_test (raw)")

# 3) Scale (optional) and SMOTE only on training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # returns numpy array
X_test_scaled  = scaler.transform(X_test)

# check for non-finite after scaling
print("\nNon-finite values in scaled train/test:", np.isfinite(X_train_scaled).all(), np.isfinite(X_test_scaled).all())

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("\nAfter SMOTE class counts:", pd.Series(y_train_res).value_counts())

# 4) Check variance of features after resample
var = np.var(X_train_res, axis=0)
print("Feature variance (resampled) - min/median/max:", var.min(), np.median(var), var.max())

# 5) Function for evaluation printing
def evaluate_model(model, X_test_in, y_test_in, name="model"):
    y_pred = model.predict(X_test_in)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_in)[:,1]
    cm = confusion_matrix(y_test_in, y_pred)
    acc = accuracy_score(y_test_in, y_pred)
    print(f"\n--- Results for {name} ---")
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", acc)
    if proba is not None:
        try:
            auc = roc_auc_score(y_test_in, proba)
            print("ROC AUC:", auc)
            print("Pred proba stats: min {:.6f}, mean {:.6f}, max {:.6f}, std {:.6f}".format(proba.min(), proba.mean(), proba.max(), proba.std()))
        except Exception as e:
            print("Couldn't compute ROC AUC:", e)
    print("\nClassification Report:\n", classification_report(y_test_in, y_pred, digits=4))
    return y_pred, proba

# 6) Try RandomForest on resampled data (WITHOUT scaling as an experiment)
rf1 = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=RANDOM_STATE)
# Train on resampled scaled data (as you did)
rf1.fit(X_train_res, y_train_res)
evaluate_model(rf1, X_test_scaled, y_test, name="RandomForest (trained on SMOTE + scaled)")

# 7) Try RandomForest but trained on original (unscaled) features with RandomUnderSampler OR SMOTE
# Train without scaling: do SMOTE on original numeric X_train (no scaler)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Try pipeline: SMOTE -> RandomForest on raw (unscaled) numeric features
sm = SMOTE(random_state=RANDOM_STATE)
X_train_raw = X_train.values  # if all numeric
X_test_raw  = X_test.values
X_sm, y_sm = sm.fit_resample(X_train_raw, y_train)

rf2 = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=RANDOM_STATE)
rf2.fit(X_sm, y_sm)
evaluate_model(rf2, X_test_raw, y_test, name="RandomForest (trained on SMOTE + RAW)")

# 8) Try Logistic Regression on scaled resampled data (good baseline)
log = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
log.fit(X_train_res, y_train_res)
y_pred_log, proba_log = evaluate_model(log, X_test_scaled, y_test, name="LogisticRegression (SMOTE + scaled)")

# 9) Check predicted probability distributions for separation (for best model candidate)
if hasattr(rf1, "predict_proba"):
    proba_rf = rf1.predict_proba(X_test_scaled)[:,1]
    print("\nRandomForest predicted proba mean/std:", proba_rf.mean(), proba_rf.std())
    # quick histogram
    plt.figure(figsize=(6,3))
    plt.hist(proba_rf, bins=40)
    plt.title("RandomForest predicted probability distribution (test set)")
    plt.show()

# 10) Cross-validated ROC-AUC (stratified)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("\nCross-validated ROC-AUC (RandomForest on original X,y with class_weight balanced):")
rf_cv = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=RANDOM_STATE)
# we use cross_val_score on original X (unresampled) but with class_weight balanced
try:
    aucs = cross_val_score(rf_cv, X.values, y.values, cv=skf, scoring='roc_auc', n_jobs=-1)
    print("CV AUCs:", aucs, " mean:", aucs.mean())
except Exception as e:
    print("Cross-val failed:", e)

# 11) Feature importances from the best RF (rf2 used on raw)
if hasattr(rf2, "feature_importances_"):
    importances = rf2.feature_importances_
    fi = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(20)
    print("\nTop feature importances (rf2):\n", fi)
    plt.figure(figsize=(6,4))
    fi.plot(kind='bar')
    plt.title("Top feature importances (rf2)")
    plt.show()

# 12) If everything fails: try simple downsampling (random undersample majority)
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=RANDOM_STATE)
X_rus, y_rus = rus.fit_resample(X_train_scaled, y_train)
print("\nAfter RandomUnderSampler class counts:", pd.Series(y_rus).value_counts())
rf3 = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_estimators=200)
rf3.fit(X_rus, y_rus)
evaluate_model(rf3, X_test_scaled, y_test, name="RandomForest (RandomUnderSampler + scaled)")

print("\n--- End of experiments ---")