import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('#Dataset_path')

# Separate features and target
x = df.drop(columns=['name', 'status'], axis=1)
y = df['status']

# Split data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Standardize data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Model training
model = svm.SVC()
model.fit(x_train, y_train)

# Model evaluation
x_train_predict = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_predict)
print("Training Accuracy Score:", training_data_accuracy)

x_test_predict = model.predict(x_test)
testing_data_accuracy = accuracy_score(y_test, x_test_predict)
print("Testing Accuracy Score:", testing_data_accuracy)

# Predict for a new input
input_data = [119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,
              0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,
              -4.813031,0.266482,2.301442,0.284654]

input_as_array = np.asarray(input_data)
input_reshaped = input_as_array.reshape(1, -1)
std_data = scaler.transform(input_reshaped)
prediction = model.predict(std_data)

print("Prediction:", prediction)
if prediction[0] == 1:
    print("The person has Parkinson's disease.")
else:
    print("The person does not have Parkinson's disease.")
