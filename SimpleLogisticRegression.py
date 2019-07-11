import pandas as pd
#col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
time_data = pd.read_csv("D:/time.csv",header=None,names=['Hours','Pass'])
#print(time_data)


x_data = time_data.iloc[:,:-1].values
y_data = time_data.iloc[:,1].values
print(x_data)
print(y_data)

#X = time_data[:,:-1].values # Features
#y = time_data[:,1].values # Target variable
#print("X::",X)
#print("Y::",y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.30,random_state=0)
print("X_train:::::::::::::::",X_train)
print("X_test:::::::::::::::::",X_test)


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict([[4]])
print("predicted value :::::::::::",y_pred)

print("Accuracy::::::::",logreg.score(X_train,y_train))
print("Coefficent::::::::;",logreg.coef_)
print(logreg.intercept_)


