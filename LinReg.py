import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score

#----------------------------------------------------------------------------------------
# Select Matplotlib style

plt.style.use('seaborn')

dataset = pd.read_csv('data2csv.csv')

X = dataset.iloc[:,1:10].values
y = dataset.iloc[:, :1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
lm = LinearRegression()
lm.fit(X_train, y_train)
predicted = lm.predict(X_test)
plt.scatter(y_test, predicted)
plt.plot([100,200], [100,200], "g--", lw=2, alpha=0.4)
plt.xlabel("NDVI_Original")
plt.ylabel("NDVI_Predicted")
plt.axis([100,200,100,200])
plt.text(100,180, ' R-squared = {}'.format(round(float(lm.score(X_test,y_test)), 2)))
plt.text(100,160, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted)), 2)))
plt.title('OLS - NDVI_Predicted v/s NDVI_Original')
plt.show()
# Compute and print R^2 and RMSE

rmse = np.sqrt(mean_squared_error(y_test, predicted))
print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(round(float(lm.score(X_test,y_test)), 2)))

def calc_train_error(X_train, y_train, model):
    '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return rmse
    
def calc_validation_error(X_test, y_test, model):
    '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
    '''fits model and returns the RMSE for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error

# intermediate/test split (gives us test set)
X_intermediate, X_test, y_intermediate, y_test = train_test_split(X, 
                                                                  y, 
                                                                  shuffle=True,
                                                                  test_size=0.2, 
                                                                  random_state=42)

# train/validation split (gives us train and validation sets)
X_train, X_validation, y_train, y_validation = train_test_split(X_intermediate,
                                                                y_intermediate,
                                                                shuffle=False,
                                                                test_size=0.25,
                                                                random_state=42)

# delete intermediate variables
del X_intermediate, y_intermediate

# print proportions
print('train: {}% | validation: {}% | test {}%'.format(round(len(y_train)/len(y),2),
                                                       round(len(y_validation)/len(y),2),
                                                       round(len(y_test)/len(y),2)))
alphas = [0.001, 0.01, 0.1, 1, 10]
print('All errors are RMSE')
print('-'*76)
for alpha in alphas:
    # instantiate and fit model
    ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    ridge.fit(X_train, y_train)
    # calculate errors
    new_train_error = np.sqrt(mean_squared_error(y_train, ridge.predict(X_train)))
    new_validation_error = np.sqrt(mean_squared_error(y_validation, ridge.predict(X_validation)))
    new_test_error = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test)))
    # print errors as report
    print('alpha: {:7} | train error: {:5} | val error: {:6} | test error: {}'.
          format(alpha,
                 round(new_train_error,3),
                 round(new_validation_error,3),
                 round(new_test_error,3)))
    
    # train/test split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    shuffle=True,
                                                    test_size=0.2, 
                                                    random_state=42)

# instantiate model
ridge = Ridge(alpha=0.11, fit_intercept=True, random_state=42)

# fit and calculate errors
new_train_error1, new_test_error1 = calc_metrics(X_train, y_train, X_test, y_test, ridge)
new_train_error1, new_test_error1 = round(new_train_error1, 3), round(new_test_error1, 3)

print('ORIGINAL ERROR')
print('-' * 40)
print('train error: {} | test error: {}\n'.format(new_train_error, new_test_error))
print('ERROR w/REGULARIZATION')
print('-' * 40)
print('train error: {} | test error: {}'.format(new_train_error1, new_test_error1))

### 10 folds cross-validation along the previous OLS Regression


alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]

val_errors = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, fit_intercept=True, random_state=42)
    errors = np.sum(-cross_val_score(lasso, 
                                     X, 
                                     y, 
                                     scoring='neg_mean_squared_error', 
                                     cv=10, 
                                     n_jobs=-1))
    val_errors.append(np.sqrt(errors))
# RMSE
print(val_errors)

print('best alpha: {}'.format(alphas[np.argmin(val_errors)]))

K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=42)

for alpha in alphas:
    train_errors = []
    validation_errors = []
    for train_index, val_index in kf.split(X, y):
        
        # split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # instantiate model
        lasso = Lasso(alpha=alpha, fit_intercept=True, random_state=42)
        
        #calculate errors
        train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, lasso)
        
        # append to appropriate list
        train_errors.append(train_error)
        validation_errors.append(val_error)
    
    # generate report
    print('alpha: {:6} | mean(train_error): {:7} | mean(val_error): {}'.
          format(alpha,
                 round(np.mean(train_errors),4),
                 round(np.mean(validation_errors),4)))
import statsmodels.formula.api as sm
X= np.append(arr=X, values=np.ones((119924,1)).astype(int),axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9]]
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


