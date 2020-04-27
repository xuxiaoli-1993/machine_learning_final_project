import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show, colorbar
import tensorflow as tf
from random import shuffle, seed, sample
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.layers import LeakyReLU,PReLU
from tensorflow import keras
import statistics

def filter_levelset(ls):
    nrows = ls.shape[0]
    ncols = ls.shape[1]
    fls = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            if ls[i][j] <= 0.0:
                fls[i][j] = 0.0
            else:
                fls[i][j] = 1.0
    return fls

def filter_Qabs(ls, Q):
    nrows = ls.shape[0]
    ncols = ls.shape[1]
    fQ = np.zeros((nrows, ncols))
    for i in range(1, nrows-1):
        for j in range(1, ncols-1):
            if ls[i][j] * ls[i+1][j] < 0 or ls[i][j] * ls[i-1][j] < 0 or ls[i][j] * ls[i][j+1] < 0 or ls[i][j] * ls[i][j-1] < 0:
                fQ[i][j] = Q[i][j]
    return fQ

def get_rmsError(ls, Qtruth, Qpred):
    nrows = ls.shape[0]
    ncols = ls.shape[1]
    err = 0.0
    ct = 0
    for i in range(1, nrows-1):
        for j in range(1, ncols-1):
            if ls[i][j] * ls[i+1][j] < 0 or ls[i][j] * ls[i-1][j] < 0 or ls[i][j] * ls[i][j+1] < 0 or ls[i][j] * ls[i][j-1] < 0:
                err = err + (Qtruth[i][j] - Qpred[i][j])**2
                ct += 1
    err = err / ct
    err = np.sqrt(err)
    return err

class EarlyStopByRMS(keras.callbacks.Callback):
    def __init__(self, xval, yval, patience=5, value=0, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.value = value
        self.verbose = verbose
        self.patience = patience
        self.xval = xval
        self.yval = yval
        self.rmsError = []
        self.stop_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        predict = self.model.predict(self.xval)
        N = self.xval.shape[0]
        err = 0.0
        for i in range(N):
            err = err + get_rmsError(self.xval[i].reshape(Nx, Ny), 
                    self.yval[i].reshape(Nx, Ny), predict[i].reshape(Nx, Ny))
        err = err / N
        print("   rms error = ", err)

        if len(self.rmsError) > self.patience:
            if err > self.rmsError[-self.patience]:
                print("Epoch %05d: early stopping Threshold" % epoch)
                self.model.stop_training = True

        self.stop_epoch = epoch
        self.rmsError.append(err)

def generate_model(filter_number, filter_size, dropout, dense_number):
    model = Sequential()
    model.add(InputLayer(input_shape=[Nx,Ny,1]))
    Nlayers = len(filter_number)
    for i in range(Nlayers):
        model.add(Conv2D(filters=filter_number[i], kernel_size=filter_size[i], activation='relu'))
        model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(dropout))
    model.add(layers.Flatten())
    model.add(Dense(dense_number, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(Nx*Ny, activation='linear'))
    return model

def generate_network(filter_number, filter_size, dropout, dense_number, ne):
    model = generate_model(filter_number, filter_size, dropout, dense_number)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, validation_data=(x_test, y_test), verbose=2, epochs=ne)
    return model


def evaluate_network(filter_number, filter_size, dropout, dense_number, ne):
    print("Evaluating hyperparameters for neural network using bootstrapping")
    SPLITS = 5

    boot = ShuffleSplit(n_splits=SPLITS, test_size=0.2)

    rms_benchmark = []
    epochs_needed = []
    num = 0

    for boot_train, boot_test in boot.split(x_train, y_train):
        num += 1
        print("Bootstrap ", num)
        xboot_train = x_train[boot_train]
        yboot_train = y_train[boot_train]
        xboot_test = x_train[boot_test]
        yboot_test = y_train[boot_test]

        print("training ......")
        model = generate_model(filter_number, filter_size, dropout, dense_number)
        model.compile(optimizer='adam', loss='mean_squared_error')
        mycallbacks = [EarlyStopByRMS(xval=xboot_test, yval=yboot_test, patience=5)]
        model.fit(xboot_train, yboot_train, validation_data=(xboot_test, yboot_test), 
                callbacks=mycallbacks, verbose=2, epochs=ne)

        print("calculating root mean squared error")
        preds = model.predict(xboot_test)
        Nboot = xboot_test.shape[0]
        rmsErr = 0.0
        for i in range(Nboot):
            rmsErr = rmsErr + get_rmsError(xboot_test[i].reshape(Nx, Ny), yboot_test[i].reshape(Nx, Ny), preds[i].reshape(Nx, Ny))
        rmsErr = rmsErr / Nboot 

        print("rms error = ", rmsErr)
        rms_benchmark.append(rmsErr)
        m = statistics.mean(rms_benchmark)

        epochs_needed.append(mycallbacks[0].stop_epoch)
        e = statistics.mean(epochs_needed)

    tf.keras.backend.clear_session()
    return m, e



seed(100)
data_path = './data'

os.system("rm -f " + data_path + "/.*.swp")

Nhead = 7
Nx = 60
Ny = 60
Nskip = Nhead + 2 * (Nx+1) * (Ny+1)

# Normalization factor
Q0 = 1e10
phi0 = 400e-6
visualize = 5

data_with_label = []
print('Reading data')
for file_name in os.listdir(data_path):
    path = os.path.join(data_path, file_name)
    f = open(path, 'r')
    lines = f.readlines()
    ls_lines = lines[Nskip:Nskip+Nx*Ny]
    Qabs_lines = lines[Nskip+Nx*Ny:]

    ls = np.zeros((Nx, Ny), dtype=np.float64)
    Qabs = np.zeros((Nx, Ny), dtype=np.float64)
    ct_line = 0
    for j in range(Ny):
        for i in range(Nx):
            ls[i][j] = np.float64(ls_lines[ct_line]) / phi0
            ct_line += 1

    ct_line = 0
    for j in range(Ny):
        for i in range(Nx):
            Qabs[i][j] = np.float64(Qabs_lines[ct_line]) / Q0
            ct_line += 1

    data_with_label.append([ls, Qabs])
    f.close()

print('Finished reading data')

shuffle(data_with_label)
Nsamples = len(data_with_label)
Ntrain = int(Nsamples * 0.9)
Ntest = Nsamples - Ntrain
train_data_with_label = data_with_label[:Ntrain]
test_data_with_label = data_with_label[Ntrain:]

x_train = np.array([i[0] for i in train_data_with_label]).reshape(-1,Nx,Ny,1)  # Size: No. of Samples * Nx * Ny * 1
y_train = np.array([i[1] for i in train_data_with_label]).reshape(-1,Nx*Ny)

x_test = np.array([i[0] for i in test_data_with_label]).reshape(-1,Nx,Ny,1)
y_test = np.array([i[1] for i in test_data_with_label]).reshape(-1,Nx*Ny)

print(x_train.shape)
print(y_train.shape)

Nfilter = [32, 64, 128]
Fsize = [5, 5, 5]
Ndrop = 0.25
Ndense = 256
Nepochs = 100

mean_rms, mean_Nepochs = evaluate_network(Nfilter, Fsize, Ndrop, Ndense, Nepochs)
print("Mean rms = ", mean_rms)
print("Mean epochs = ", mean_Nepochs)
mean_Nepochs = int(mean_Nepochs)
# model.summary()
# exit()

model = generate_network(Nfilter, Fsize, Ndrop, Ndense, mean_Nepochs) 
# Check prediction from test data
preds = model.predict(x_test)

# Visualize data
if visualize > 0:
    id_list = list(range(Ntest))
    selected_list = sample(id_list, visualize)
    print('visualize from testing examples: ', selected_list)
    
    fig1, axes1 = plt.subplots(visualize, 4, figsize=(4*2, visualize*2))
    for n in range(visualize):
        i = selected_list[n] 
        axes1[n][0].imshow(np.rot90(x_test[i].reshape(Nx, Ny)))
        axes1[n][1].imshow(np.rot90(filter_levelset(x_test[i].reshape(Nx, Ny))))
        axes1[n][2].imshow(np.rot90(filter_Qabs(x_test[i].reshape(Nx, Ny), y_test[i].reshape(Nx, Ny))), cmap='hot')
        axes1[n][3].imshow(np.rot90(filter_Qabs(x_test[i].reshape(Nx, Ny), preds[i].reshape(Nx, Ny))), cmap='hot')
        if n == 0:
            axes1[n][0].set_title('Levelset')
            axes1[n][1].set_title('Boundary')
            axes1[n][2].set_title('True')
            axes1[n][3].set_title('Predicted')
    plt.setp(axes1, xticks=[], yticks=[])
    fig1.suptitle("Testing Data", fontsize=14)

# Calculate root mean squared error
err_test = 0.0
for i in range(Ntest):
    err_test = err_test + get_rmsError(x_test[i].reshape(Nx, Ny), y_test[i].reshape(Nx, Ny), preds[i].reshape(Nx, Ny))

err_test = err_test / Ntest
print("Average testing error = %20.10f " % err_test)


# Check predictions from training data
preds = model.predict(x_train)

# Visualize data
if visualize > 0:
    id_list = list(range(Ntrain))
    selected_list = sample(id_list, visualize)
    print('visualize from training examples', selected_list)
    
    fig2, axes2 = plt.subplots(visualize, 4, figsize=(4*2, visualize*2))
    for n in range(visualize):
        i = selected_list[n]
        axes2[n][0].imshow(np.rot90(x_train[i].reshape(Nx, Ny)))
        axes2[n][1].imshow(np.rot90(filter_levelset(x_train[i].reshape(Nx, Ny))))
        axes2[n][2].imshow(np.rot90(filter_Qabs(x_train[i].reshape(Nx, Ny), y_train[i].reshape(Nx, Ny))), cmap='hot')
        axes2[n][3].imshow(np.rot90(filter_Qabs(x_train[i].reshape(Nx, Ny), preds[i].reshape(Nx, Ny))), cmap='hot')
        if n == 0:
            axes2[n][0].set_title('Levelset')
            axes2[n][1].set_title('Boundary')
            axes2[n][2].set_title('True')
            axes2[n][3].set_title('Predicted')
    plt.setp(axes2, xticks=[], yticks=[])
    fig2.suptitle("Training Data", fontsize=14)

# Calculate root mean squared error
err_train = 0.0
for i in range(Ntrain):
    err_train = err_train + get_rmsError(x_train[i].reshape(Nx, Ny), y_train[i].reshape(Nx, Ny), preds[i].reshape(Nx, Ny))

err_train = err_train / Ntrain
print("Average training error = %20.10f" % err_train)

if visualize > 0:
    show()
