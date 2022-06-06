# 1-Dimensional Convolutional Neural Network for recognizing human pose action


# from matplotlib import pyplot as plt
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical


# Needs visualization, add later

# I did two versions of different data pre-processing, one by combining everything
# into a csv file (which is too big and too slow to deal with) and another by stitching files together. Here is the
# second version.

# NOTE: Please put all the dataset files under the same directory as this one, in a folder called "HARDataset". My code heavily rely on that name, unfortunately


# Function to add a single file to numpy
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# Append all files into a list, make it into a 3d array and return that
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # dstack flattens the nested file list
    loaded = dstack(loaded)
    print (loaded)
    return loaded


# Function to organize train or test data into their separate groups; did cut some slack using the naming here
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'

    # List to contain data
    filenames = list()
    # add in total acceleration data
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # X is then loaded with input data
    X = load_group(filenames, filepath)
    # Y with output
    Y = load_file(prefix + group + '/y_' + group + '.txt')
    print("prefix: " + prefix + "group: " + group)
    return X, Y


# Load all training and test data into dataset
def load_dataset(prefix=''):
    # Training
    trainX, trainY = load_dataset_group('train', prefix + 'HARDataset/')
    print(trainX.shape, trainY.shape)
    print("The trainX looks like this: |||")
    print("                             ")
    print(trainX)
    # Test
    testX, testY = load_dataset_group('test', prefix + 'HARDataset/')
    print(testX.shape, testY.shape)

    # Do this or the output label go overflow
    trainY = trainY - 1
    testY = testY - 1

    # hot code Y sets so they can be read correctly
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    print("trainX shape: ", trainX.shape, "trainY shape: ", trainY.shape, "testX shape: ", testX.shape, "testY shape: ",
          testY.shape)
    return trainX, trainY, testX, testY


# fitting and evaluation
def evaluate_model(trainX, trainy, testX, testy):
    verbose = 0
    epochs = 10
    batch_size = 32

    n_timesteps = trainX.shape[1] # 128 timesteps
    n_features = trainX.shape[2] # 9 different features
    n_outputs = trainy.shape[1]

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))  # Only 1-dimensional CNN requires the input_shape?
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the model with training class
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=2)
    return accuracy


# Accuracy Score Summary
def summarize_results(scores):
    print(scores)
    m = mean(scores)
    s = std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# Runs the experiment, change repeats here
def run_experiment(repeats=3):
    # load training data and test data
    trainX, trainy, testX, testy = load_dataset()

    # repeat experiment multiple times
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0   # Make it a percentage
        print('Score for repeat number >#%d: %.3f' % (r + 1, score))
        scores.append(score)

        # Use summarize function to see a overview
    summarize_results(scores)


# Entrance here
run_experiment()
