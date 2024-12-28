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


df = pd.read_csv('./creditcard.csv')


def preprocess_credit_card_fraud_data(df):
   
    df.dropna(inplace=True)
    
   
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    
    df.drop(['Time'], axis=1, inplace=True)
    
    return df


df = preprocess_credit_card_fraud_data(df)


X = df.drop('Class', axis=1)
y = df['Class']


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


model.save('credit_card_fraud_nn_model.h5')
