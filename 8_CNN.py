import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers.legacy import Adam
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


data = pd.read_csv('data2')

X = data.drop(['depth_to_groundwater'], axis=1)
y = data['depth_to_groundwater']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.02, random_state=42, shuffle=False)

# Reshape X_train for Conv1D model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Define CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=1))

model.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Reshape X_test for prediction
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_pred = model.predict(X_test_reshaped)

y_test = np.array(y_test).reshape(-1, 1)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R square (coefficient of determination):", r2)
print("Mean square error (MSE):", mse)
print("Mean square absolute error (MAE):", mae)

# Draw a forecast comparison chart
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual_value', color='tab:blue', linewidth=2)
plt.plot(y_pred, label='Predicted_value', color='tab:red', linestyle='dashed', linewidth=2)
plt.xlabel('Sample_ID', fontsize=12)
plt.ylabel('Target_value', fontsize=12)
plt.title('Actual vs Predicted (CNN)', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.scatter(y_test, y_pred, color='b', alpha=0.5)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='darkred', linestyle='--',linewidth=2)
#plt.title("Scatter Plot of True Values vs Predictions")
plt.title(f'CNN  R²: {r2:.4f}  MSE: {mse:.4f}  MAE: {mae:.4f}', fontsize=14)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.savefig(fname='figure/Scatter_plot_CNN.png',dpi=600)
plt.show()

y_test = np.array(y_test).reshape(-1, 1)
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Error_histogram_Distribution', fontsize=14)
# Add average error marker
mean_error = np.mean(errors)
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Average_error = {mean_error:.2f}')
plt.legend()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()