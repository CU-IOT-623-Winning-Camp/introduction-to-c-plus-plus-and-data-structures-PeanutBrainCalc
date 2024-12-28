import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('./fraud-ecommerce/Fraud_Data.csv')


def preprocess_ecommerce_fraud_data(df):
    
    df.dropna(inplace=True)
    
    
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['time_diff'] = (df['purchase_time'] - df['signup_time']).dt.seconds
    df.drop(['signup_time', 'purchase_time'], axis=1, inplace=True)
    
    
    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    
    return df


df = preprocess_ecommerce_fraud_data(df)


X = df.drop('class', axis=1)
y = df['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


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


model.save('ecommerce_fraud_nn_model.h5')
