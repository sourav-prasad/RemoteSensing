import pandas as pd
from keras.layers import Dense, Activation,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot

# Importing the dataset
# Importing the dataset
dataset = pd.read_csv('data2csv.csv')

X = dataset.iloc[:,1:10].values
y = dataset.iloc[:, :1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer

model.add(Dense(5, activation = 'relu', input_dim = 9))
model.add(Dropout(0.5))

# Adding the second hidden layer
model.add(Dense(units = 5, activation = 'relu'))


# Adding the third hidden layer
model.add(Dense(units = 5, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 5, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 5, activation = 'relu'))

# Adding the output layer

model.add(Dense(units = 1))

#model.add(Dense(1))
# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['mae','mse','mape','cosine'])

# Fitting the ANN to the Training set
history=model.fit(X_train, y_train,validation_data=(X_val, y_val) ,batch_size = 1000, epochs = 5)
test_loss = model.evaluate(X_test,y_test)
print(model.metrics_names)

loss = history.history['loss']
acc = history.history['mean_absolute_error']
val_loss = history.history['val_loss']
val_acc = history.history['val_mean_absolute_error']
mape_loss=history.history['mean_absolute_percentage_error']
cosine_los=history.history['cosine_proximity']
pyplot.plot(history.history['mean_squared_error'])
pyplot.plot(history.history['mean_absolute_error'])
pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.plot(history.history['cosine_proximity'])
pyplot.show()
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='validation loss')

plt.legend()
plt.show()

y_pred = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()
