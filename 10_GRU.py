import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from  keras.models import Sequential
from  keras.layers import LSTM, Dense,Bidirectional,Dropout,GRU
import random
import numpy as np
import tensorflow as tf


# Set random seed
seed_value = 123
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

df = pd.read_csv('data2')#read data

#del df['date']

X = df.drop(['depth_to_groundwater'],axis=1)#input
y = df['depth_to_groundwater']#output
X_length = X.shape[0]# Division ratio
split = int(X_length*0.98)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = y[:split], y[split:]
y_test = Y_test
y_pred = Y_test
Y_train = pd.DataFrame(Y_train)
x_scaler = MinMaxScaler()
x_scaled = x_scaler.fit_transform(X_train)
x_train = x_scaler.transform(X_train)
y_scaler = MinMaxScaler()#Normalized
y_scaler = y_scaler.fit(Y_train)
y_train = y_scaler.transform(Y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))#Reshape the structure
from keras import optimizers

opt = optimizers.legacy.Adam(learning_rate=0.001) # optimizer
def build_model(input):#Model structure
    model = Sequential()# Create Sequential model
    model.add((GRU(128, input_shape=(input[1], input[2]), activation="relu",return_sequences=True)))# Add the first GRU layer
    model.add(Dropout(0.4))# Add Dropout layer to prevent overfitting
    model.add((GRU(64)))# Add a second GRU layer
    model.add(Dense(1))# Add output layer
    model.compile(loss='mse', optimizer=opt)# Compile the model, using mean squared error as the loss function, Adam optimizer
    return model
model = build_model([x_train.shape[0], 1, x_train.shape[2]])
batch_size = 32
epochs = 50
from timeit import default_timer as timer

start = timer()
history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2)

x_test = x_scaler.transform(X_test)
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
print('x_test', x_test.shape)

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

y_pred = model.predict(x_test)#prediction
y_pred = y_scaler.inverse_transform(y_pred)#denormalization
a = pd.DataFrame()
a['Predicted_value'] = list(y_pred)
a['Actual_value'] = list(Y_test)

# 计算评价指标
y_test=np.array(y_test).reshape(-1,1)
R2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

print("MSE", MSE)
print("MAE", MAE)
print("r^2:", R2)

# Draw a forecast comparison chart
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual_values', color='tab:blue', linewidth=2)
plt.plot(y_pred, label='Predicted_values', color='tab:red', linestyle='dashed', linewidth=2)
plt.xlabel('Sample_ID', fontsize=12)
plt.ylabel('Target_value', fontsize=12)
plt.title('Actual vs Predicted (GRU)', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Plot scatter plots and fitted lines
plt.scatter(y_test, y_pred, color='b', alpha=0.5)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='darkred', linestyle='--',linewidth=2)
#plt.title("Scatter Plot of True Values vs Predictions")
plt.title(f'GRU  R²: {R2:.4f}  MSE: {MSE:.4f}  MAE: {MAE:.4f}', fontsize=14)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.savefig(fname='figure/Scatter_plot_GRU.png',dpi=600)
plt.show()
# Generate error histogram
y_test=np.array(y_test).reshape(-1,1)
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Error_Histogram_Distribution', fontsize=14)
# Add average error marker
mean_error = np.mean(errors)
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'average_error = {mean_error:.2f}')
plt.legend()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('Error_histogram_main.png', dpi=600)  # Specify the saved file name and file format
plt.show()