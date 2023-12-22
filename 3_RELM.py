import numpy as np
from sklearn.preprocessing import OneHotEncoder
import numpy as np



class RELM_HiddenLayer:

    """
      Regularized extreme learning machine
        :param x: training set attribute X when initializing the learning machine
        :param num: Number of hidden layer nodes of the learning machine
        :param C: the reciprocal of the regularization coefficient
    """

    def __init__(self, x, num, C=10):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState()
        # weights w
        self.w = rnd.uniform(-1, 1, (columns, num))
        # bias b
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        self.H0 = np.matrix(self.softplus(np.dot(x, self.w) + self.b))
        self.C = C
        self.P = (self.H0.H * self.H0 + len(x) / self.C).I

    @staticmethod
    def sigmoid(x):
        """
            Activation function sigmoid
            :param x: X in the training set
            :return: activation value
        """
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def softplus(x):
        """
           Activation function softplus
            :param x: X in the training set
            :return: activation value
        """
        return np.log(1 + np.exp(x))

    @staticmethod
    def tanh(x):
        """
            Activation function tanh
            :param x: X in the training set
            :return: activation value
        """
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    # regression problem training
    def regressor_train(self, T):
        """
            After initializing the learning machine, you need to pass in the corresponding label T
            :param T: tag T corresponding to attribute X
            :return: Hidden layer output weight beta
        """
        all_m = np.dot(self.P, self.H0.H)
        self.beta = np.dot(all_m, T)
        return self.beta

    # Regression problem testing
    def regressor_test(self, test_x):
        """
            Pass in the attribute X to be predicted and perform prediction to obtain the predicted value
            :param test_x: attribute X of the predicted label
            :return: predicted value T of the predicted label
        """
        b_row = test_x.shape[0]
        h = self.softplus(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta.T)
        return result

    # Classification problem training
    def classifisor_train(self, T):
        """
            After initializing the learning machine, you need to pass in the corresponding label T
            :param T: tag T corresponding to attribute X
            :return: Hidden layer output weight beta
        """
        if len(T.shape) > 1:
            pass
        else:
            self.en_one = OneHotEncoder()
            T = self.en_one.fit_transform(T.reshape(-1, 1)).toarray()
            pass
        all_m = np.dot(self.P, self.H0.H)
        self.beta = np.dot(all_m, T)
        return self.beta
        pass

    # Classification Questions Test
    def classifisor_test(self, test_x):
        """
            Pass in the attribute X to be predicted and perform prediction to obtain the predicted value
            :param test_x: attribute X of the predicted label
            :return: predicted value T of the predicted label
        """
        b_row = test_x.shape[0]
        h = self.softplus(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result = [item.tolist().index(max(item.tolist())) for item in result]
        return result
        pass


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
#
# # Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.02, random_state=42,shuffle=False)

import numpy as np


# Train the RELM model
my_EML = RELM_HiddenLayer(X_train,100)#5，30
my_EML.regressor_train(y_train)
# Make predictions on the test set
y_pred = np.asarray(my_EML.regressor_test(X_test))


# Calculate evaluation indicators
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R square (coefficient of determination):", r2)
print("Mean square error (MSE):", mse)
print("Mean square absolute error (MAE):", mae)
y_test=np.array(y_test).reshape(-1,1)

# Draw a forecast comparison chart
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual_value', color='black', linewidth=2)
plt.plot(y_pred, label='Predict_value', color='tab:red', linestyle='dashed', linewidth=2)
plt.xlabel('Sample_ID', fontsize=12)
plt.ylabel('Target_value', fontsize=12)
plt.title('Actual vs Predicted (RELM)', fontsize=14)
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
plt.title(f'RELM  R²: {r2:.4f}  MSE: {mse:.4f}  MAE: {mae:.4f}', fontsize=14)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.savefig(fname='figure/Scatter_plot_RELM.png',dpi=600)
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
plt.title('Error_histogram_Distribution', fontsize=14)
# Add average error marker
mean_error = np.mean(errors)
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'average_error = {mean_error:.2f}')
plt.legend()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
