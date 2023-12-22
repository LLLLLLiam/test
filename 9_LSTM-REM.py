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

df = pd.read_csv('data2')

#del df['date']

X = df.drop(['depth_to_groundwater'],axis=1)
y = df['depth_to_groundwater']
X_length = X.shape[0]
split = int(X_length*0.98)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = y[:split], y[split:]
y_test = Y_test
y_pred = Y_test
Y_train = pd.DataFrame(Y_train)
x_scaler = MinMaxScaler()
x_scaled = x_scaler.fit_transform(X_train)
x_train = x_scaler.transform(X_train)
y_scaler = MinMaxScaler()
y_scaler = y_scaler.fit(Y_train)
y_train = y_scaler.transform(Y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
from keras import optimizers

opt = optimizers.legacy.Adam(learning_rate=0.001) # optimizer

def build_model(input):
    model = Sequential()
    model.add((LSTM(128, input_shape=(input[1], input[2]), activation="relu",return_sequences=True)))
    model.add(Dropout(0.4))
    model.add((LSTM(64)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=opt)
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

pre = model.predict(x_test)
y_test_pre = y_scaler.inverse_transform(pre)
a = pd.DataFrame()
a['Predicted_value'] = list(y_test_pre)
a['Actual_value'] = list(Y_test)
print(a)
print('y_test_pre', y_test_pre.shape)
iters = np.arange(len(y_test_pre))
MSE = mean_squared_error(y_test_pre, y_pred)
MAE = mean_absolute_error(y_test_pre, y_pred)
R2 = r2_score(y_test_pre, y_pred)
print("MSE",MSE)
print("MAE",MAE)
print("r^2:",R2)
y_pred_0 = np.array(y_pred).reshape(-1, 1)
#y_pred_0 = []
#for i in range(len(y_pred)):
#    y_pred_0.append([y_pred[i]])
y_test_error_1 = y_pred_0 - y_test_pre
print(y_test_error_1)

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
        self.H0 = np.matrix(self.sigmoid(np.dot(x, self.w) + self.b))
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
            :param test_x:features
            :return: predicted value
        """
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result


X = df.drop(['depth_to_groundwater'],axis=1)
X_length = X.shape[0]
split = int(X_length*0.98)
X_train, X_test = X[:split], X[split:]
Y_train = pd.DataFrame(Y_train)
x_scaler = MinMaxScaler()
x_scaled = x_scaler.fit_transform(X_test)
x_train = x_scaler.transform(X_test)
y_scaler = MinMaxScaler()
y_scaler = y_scaler.fit(y_test_error_1)
y_train = y_scaler.transform(y_test_error_1)
print("x_train", x_train.shape)
print("y_train", y_train.shape)

my_EML = RELM_HiddenLayer(x_train,30)#5，30
my_EML.regressor_train(y_train)
x_test = x_scaler.transform(X_test)
pre_1 = my_EML.regressor_test(x_test)
pre_1 = np.array(pre_1).reshape(-1, 1)
y_test_error_2 = y_scaler.inverse_transform(pre_1)
y_test_correct = y_test_pre + y_test_error_2
end = timer()
print("time_consuming：", end - start)
a = pd.DataFrame()
a['Predicted_value'] = list(y_test_correct)
a['Actual_value'] = list(y_pred)
print(a)
iters = np.arange(len(y_test))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Times New Roman')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(iters, y_test_correct, color='#FF4500', linestyle='--', linewidth=2.5, label='GRU-RELM Pred')
ax.plot(iters, y_pred.values, color='#4169E1', linestyle='-', linewidth=2.5, label='Actual')
# Set axis labels and titles
ax.set_xlabel('Number')
ax.set_ylabel('Depth_to_Groundwater_P24')
ax.set_title('Actual vs. Predicted Groundwater_P24')
# Set legend
ax.legend()
# set grid lines
ax.grid(True)
# Set background color to white
ax.set_facecolor('white')
# Save the image as PNG format, dpi is the resolution
#plt.savefig('LSTM-RELM.png', dpi=600)
# display
plt.show()

plt.scatter(y_test_correct, y_pred.values, color='b', alpha=0.5)
plt.plot([np.min(y_test_correct), np.max(y_test_correct)], [np.min(y_test_correct), np.max(y_test_correct)], color='darkred', linestyle='--',linewidth=2)
#plt.title("Scatter Plot of True Values vs Predictions")
plt.title(f'LSTM-RELM  R²: {R2:.4f}  MSE: {MSE:.4f}  MAE: {MAE:.4f}', fontsize=14)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.savefig(fname='figure/Scatter_plot_LSTM-RELM.png',dpi=600)
plt.show()

# Generate error histogram
y_pred=np.array(y_pred).reshape(-1,1)

errors = y_test_correct - y_pred
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

# add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Error Histogram Distribution', fontsize=14)
# Add average error marker
mean_error = np.mean(errors)
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'mean_error = {mean_error:.2f}')
plt.legend()
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('Error_histogram.png', dpi=600) # Specify the saved file name and file format
plt.show()


MSE = mean_squared_error(y_test_correct,y_pred)
MAE = mean_absolute_error(y_test_correct,y_pred)
R2 = r2_score(y_test_correct,y_pred)
y_pred_1 = np.array(y_pred).reshape(-1, 1)
print("MSE",MSE)
print("MAE",MAE)
print("r^2:",R2)
