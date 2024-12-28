import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE 


df = pd.read_csv('.\creditcard.csv')


print(df.info())
print(df.describe())
print(df.head())


print(df.isnull().sum())


X = df.drop('Class', axis=1)
y = df['Class']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)  
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train) 


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)  


y_pred = model.predict(X_test)


print("Classification Report:")
print(classification_report(y_test, y_pred))


print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.show()


import joblib
joblib.dump(model, './fraud_detectionCreditCard_RandomForestfinal_model.pkl')

