import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt



data = pd.read_csv('data2')

#del data['date']
X = data.drop(['depth_to_groundwater'],axis=1)
y = data['depth_to_groundwater']
#Create a normalized object
scaler = MinMaxScaler()
#Normalize X
X_scaled = scaler.fit_transform(X)

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.02, random_state=42,shuffle=False)

# Define XGBoost model
model = xgb.XGBRegressor(
    objective='reg:squarederror', #Regression task
    n_estimators=20, #Number of iterations
    max_depth=5, # The maximum depth of the tree
    learning_rate=0.1 # Learning rate
)

#Train XGBoost model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation indicators
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R square (coefficient of determination):", r2)
print("Mean square error (MSE):", mse)
print("Mean square absolute error (MAE):", mae)
y_test=np.array(y_test).reshape(-1,1)
# Draw a forecast comparison chart
plt.figure(figsize=(8, 6))
plt.plot(y_test, label='Actual value', color='tab:blue', linewidth=2)
plt.plot(y_pred, label='Predicted_value', color='tab:red', linestyle='dashed', linewidth=2)
plt.xlabel('Sample_ID', fontsize=12)
plt.ylabel('Target_value', fontsize=12)
plt.title('Actual vs Predicted (XGBoost)', fontsize=14)
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
plt.title(f'XGBoost  R²: {r2:.4f}  MSE: {mse:.4f}  MAE: {mae:.4f}', fontsize=14)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.savefig(fname='figure/Scatter_plot_XGBoost.png',dpi=600)
plt.show()
# Generate error histogram
y_pred=np.array(y_pred).reshape(-1,1)

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
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'average_value = {mean_error:.2f}')
plt.legend()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#plt.savefig('Error_histogram.png', dpi=600) # Specify the saved file name and file format
plt.show()