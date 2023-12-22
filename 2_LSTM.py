import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers.legacy import Adam
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


data = pd.read_csv('data2')

#del data['date']
X = data.drop(['depth_to_groundwater'],axis=1)
y = data['depth_to_groundwater']

# Create a normalized object
scaler = MinMaxScaler()
# Normalize X
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.02, random_state=42,shuffle=False)

# Reshape X_train as (number of samples, time steps, number of features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
#,kernel_regularizer=L2(0.05)
# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=10, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error')

# Training model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions on the test set
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_pred = model.predict(X_test_reshaped)

# Calculate evaluation indicators
y_test=np.array(y_test).reshape(-1,1)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R_square (coefficient of determination):", r2)
print("Mean square error (MSE):", mse)
print("Mean square absolute error (MAE):", mae)

# Draw a forecast comparison chart
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual_value', color='tab:blue', linewidth=2)
plt.plot(y_pred, label='Predictive_value', color='tab:red', linestyle='dashed', linewidth=2)
plt.xlabel('Sample_ID', fontsize=12)
plt.ylabel('Target_value', fontsize=12)
plt.title('Actual_value vs Predictive_value', fontsize=14)
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
plt.title(f'LSTM  R²: {r2:.4f}  MSE: {mse:.4f}  MAE: {mae:.4f}', fontsize=14)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.savefig(fname='figure/Scatter_plot_LSTM.png',dpi=600)
plt.show()
# Generate error histogram
y_test=np.array(y_test).reshape(-1,1)
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('error', fontsize=12)
plt.ylabel('frequency', fontsize=12)
plt.title('Error_histogram_Distribution', fontsize=14)
# Add average error marker
mean_error = np.mean(errors)
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'average_error = {mean_error:.2f}')
plt.legend()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#plt.savefig('Error_histogram_Distribution.png', dpi=600) # Specify the saved file name and file format
plt.show()