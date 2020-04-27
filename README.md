Machine Learning (CS6350) Final Project
by Xuxiao Li (UID: u1013882)
April, 28, 2020
#################################

The project report is located in the folder "Report".

The data is located in the folder "data" where 1490 pieces of data can be found. These will be split
into the training and testing set.

The code is the file "Bootstrap.py" where 5 cycles of bootstrapping are conducted for benchmarking a
certain combination of hyperparameters. You will need to have python3.6 or higher to run this code.
Also, you have to install the package "tensorflow", "sklearn", "numpy", and "matplotlib" which are
utilized in the code.

To run the code, use:
$ python3 Bootstrap.py

The tunable hyperparameters are the number of convolutional layers, the filter number and kernel
size in each convolutional layer, the dropout rate, and the number of nodes in the second last dense
layer (see Fig. 5 of the report). To tune these parameter, you need to manually adjust them at line
206 - 209 in the code "Bootstrap.py". 

For a certain combination of hyperparameters, the optimal number of epochs is the average stopping epoch
from the 5 bootstrapping cycle (see report), which will be used to train the entire training set.
The benchmark for a certain combination of hyperparameters is the average RMS error from the 5
bootstrapping cycles (see report).

