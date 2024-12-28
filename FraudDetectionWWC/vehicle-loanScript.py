import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


df = pd.read_csv('./vehicle-loan-default-prediction/train.csv')


def preprocess_vehicle_loan_data(df):
    
    df.dropna(subset=['EMPLOYMENT_TYPE'], inplace=True)
    
    
    df['DATE_OF_BIRTH'] = pd.to_datetime(df['DATE_OF_BIRTH'], dayfirst=True)
    df['age'] = (pd.to_datetime('today') - df['DATE_OF_BIRTH']).dt.days // 365
    df.drop(['DATE_OF_BIRTH'], axis=1, inplace=True)
    
    
    df = pd.get_dummies(df, drop_first=True)
    
    return df


df = preprocess_vehicle_loan_data(df)


X = df.drop('LOAN_DEFAULT', axis=1)
y = df['LOAN_DEFAULT']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)


model = Sequential()


model.add(Dense(64, input_dim=X_train_resampled.shape[1], activation='relu'))


model.add(Dropout(0.5))


model.add(Dense(32, activation='relu'))


model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, validation_split=0.1, verbose=1)


y_pred = (model.predict(X_test) > 0.5).astype('int32')


print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


model.save('vehicle_loan_nn_model.h5')
