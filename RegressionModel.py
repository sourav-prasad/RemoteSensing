#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

# Importing the dataset
dataset = pd.read_csv('data2csv.csv')

X = dataset.iloc[:,1:10].values
y = dataset.iloc[:, :1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=12)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept=True)
regressor.fit(X_train, y_train)

def calc_ISE(X_train, y_train, model):
    '''returns the in-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return model.score(X_train, y_train), rmse
def calc_OSE(X_test, y_test, model):
    '''returns the out-of-sample R^2 and RMSE; assumes model already fit.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return model.score(X_test, y_test), rmse
is_r2, ise = calc_ISE(X_train, y_train,regressor)
os_r2, ose = calc_OSE(X_test, y_test, regressor)

# show dataset sizes
data_list = (('R^2_in', is_r2), ('R^2_out', os_r2), 
             ('ISE', ise), ('OSE', ose))
for item in data_list:
    print('{:10}: {}'.format(item[0], item[1]))

# Predicting the Test set results
y_pred = regressor.predict(X_test)
from sklearn import metrics
correlation_matrix=dataset.corr()

print("Root Mean squared error: %.2f" %np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))

print('Variance score: %.2f' % regressor.score(X_test, y_test))

import statsmodels.formula.api as sm
X= np.append(arr=X, values=np.ones((119924,1)).astype(int),axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


