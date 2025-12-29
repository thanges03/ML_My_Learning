## Project Linear regression on a random dataset

## import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

##datset generation
np.random.seed(42)
x = np.random.rand(100,1)*10 # 100 random values
y = 3*x+5+np.random.randn(100,1) # y = 3x+5 +noise

##dataframe conversion
data = pd.DataFrame({'Feature':x.flatten(),'Target':y.flatten()})

## train test model
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

##train model
model = LinearRegression()
model.fit(x_train,y_train)

##model parameter
print("Slope (m) : ",model.coef_[0][0])
print("Intercept (c) : ",model.intercept_[0])

## prediction
y_pred = model.predict(x_test)

##Evaluation
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("MSE : ",mse)
print("r2 Score : ",r2)

## visualization
plt.scatter(x_test,y_test)
plt.plot(x_test,y_test)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title("Linear Regression on Random Dataset")
plt.show()
