# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## Aim:

To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Load the dataset Salary.csv using pandas and view the first few rows.

2.Check dataset information and identify any missing values.

3.Encode the categorical column "Position" into numerical values using LabelEncoder.

4.Define feature variables x as "Position" and "Level", and target variable y as "Salary".

5.Split the dataset into training and testing sets using an 80-20 split.

6.Create a DecisionTreeRegressor model instance.

7.Train the model using the training data.

8.Predict the salary values using the test data.

9.Evaluate the model using Mean Squared Error (MSE) and R² Score.

10.Use the trained model to predict salary for a new input [5, 6].

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Amruthavarshini Gopal 
RegisterNumber: 212223230013
*/
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()
df.info()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head()
x=df[['Position','Level']]
y=df['Salary']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_pred)
mse
r2=r2_score(y_test,y_pred)
r2
model.predict([[5,6]])
```

## Output:

### Head of Dataset
![image](https://github.com/user-attachments/assets/eda80df6-b201-4c51-9eac-feb03f424115)

### Dataset Info
![image](https://github.com/user-attachments/assets/fd97a873-4568-4aa5-89aa-8887512ec21d)

### Null Counts
![image](https://github.com/user-attachments/assets/ce4f129d-b3dc-463e-8f4e-867c07d6c896)

### Encoded Data
![Screenshot 2025-04-28 162330](https://github.com/user-attachments/assets/466e046a-8436-4950-902c-b7fc514cdaee)

### MSE Value
![image](https://github.com/user-attachments/assets/9fd29151-1c8a-461b-8309-949bb7ca3d0d)

### R2 Value
![image](https://github.com/user-attachments/assets/2863620e-d3fc-4107-a7af-05d1f3198a62)

### Predicted Value for new data
![image](https://github.com/user-attachments/assets/0ea3d587-7da8-43f2-b5d8-439c8f46827f)

## Result:

Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
