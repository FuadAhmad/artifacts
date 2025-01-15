from __future__ import division, absolute_import, print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.datasets import mnist, cifar10
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2

import sys
sys.path.append('../')  # Add the parent directory to the Python path
import pandas as pd
import random
from sklearn.neighbors import KernelDensity


# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
STDEVS = {
    'mnist': {'fgsm': 0.310, 'bim-a': 0.128, 'bim-b': 0.265},
    'cifar': {'fgsm': 0.050, 'bim-a': 0.009, 'bim-b': 0.039},
    'svhn': {'fgsm': 0.132, 'bim-a': 0.015, 'bim-b': 0.122},
    'nsl': {'fgsm': 0.310, 'bim-a': 0.128, 'bim-b': 0.265, 'pgd':0.310}
}
# Set random seed
np.random.seed(0)

nClass=10

def get_data(dataset='mnist'):
    """
    TODO
    :param dataset:
    :return:
    """
    assert dataset in ['mnist', 'cifar', 'svhn', 'nsl'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn' OR 'nsl'"
    
    if dataset == 'nsl':

        file_path = "../data/nsl/"
        df = pd.read_csv(file_path+"train.csv")
        Y_train = df["attack"]
        X_train = df.drop(["attack"], axis=1)
        
        df = pd.read_csv(file_path+"test.csv")
        Y_test = df["attack"]
        X_test = df.drop(["attack"], axis=1)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.2)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) 
        nClass = len(np.unique(Y_train))
        #return X_train, Y_train, X_test, Y_test # Binary loss='binary_crossentropy' optimizer='adam', 
        return X_train, to_categorical(Y_train, nClass), X_test, to_categorical(Y_test, nClass)
    
    if dataset == 'mnist':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # reshape to (n_samples, 28, 28, 1)
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        nClass = len(np.unique(y_train))
    elif dataset == 'cifar':
        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        nClass = len(np.unique(y_train))
    else:
        if not os.path.isfile("../data/svhn_train.mat"):
            print('Downloading SVHN train set...')
            call(
                "curl -o ../data/svhn_train.mat "
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                shell=True
            )
        if not os.path.isfile("../data/svhn_test.mat"):
            print('Downloading SVHN test set...')
            call(
                "curl -o ../data/svhn_test.mat "
                "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                shell=True
            )
        train = sio.loadmat('../data/svhn_train.mat')
        test = sio.loadmat('../data/svhn_test.mat')
        X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
        X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
        # reshape (n_samples, 1) to (n_samples,) and change 1-index
        # to 0-index
        y_train = np.reshape(train['y'], (-1,)) - 1
        y_test = np.reshape(test['y'], (-1,)) - 1

    # cast pixels to floats, normalize to [0, 1] range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # one-hot-encode the labels
    Y_train = to_categorical(y_train, nClass)
    Y_test = to_categorical(y_test, nClass)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    return X_train, Y_train, X_test, Y_test


def get_model(dataset='mnist', nClass = 2):
    """
    Takes in a parameter indicating which model type to use ('mnist',
    'cifar' or 'svhn') and returns the appropriate Keras model.
    :param dataset: A string indicating which dataset we are building
                    a model for.
    :return: The model; a Keras 'Sequential' instance.
    """
    assert dataset in ['mnist', 'cifar', 'svhn', 'nsl'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'nsl':
        # model for NSL KDD dataset 
        
        model = Sequential([
            Dense(64, input_dim=42, activation='relu'),
            Dense(128, activation='relu'),
            # Dropout(0.5),  # Uncomment for dropout
            Dense(64, activation='relu'),
            # Dropout(0.5),  # Uncomment for dropout  
        ])
        #if nClass == 2:
        #model.add(Dense(1, activation='sigmoid'))  # For binary classification)
        #else:
        model.add(Dense(nClass, activation='softmax'))
            
        return model
    
    elif dataset == 'mnist':
        # MNIST model
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]
    elif dataset == 'cifar':
        # CIFAR-10 model
        layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
            Activation('relu'),
            Conv2D(32, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(64, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding='same'),
            Activation('relu'),
            Conv2D(128, (3, 3), padding='same'),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation('relu'),
            Dropout(0.5),
            Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]
    else:
        # SVHN model
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(32, 32, 3)),
            Activation('relu'),
            Conv2D(64, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(512),
            Activation('relu'),
            Dropout(0.5),
            Dense(128),
            Activation('relu'),
            Dropout(0.5),
            Dense(10),
            Activation('softmax')
        ]

    model = Sequential()
    for layer in layers:
        model.add(layer)

    return model


def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < 0.99)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = 1.

    return np.reshape(x, original_shape)


def get_noisy_samples_ori(X_test, X_test_adv, dataset, attack):
    """
    TODO
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    if attack in ['jsma', 'cw']:
        X_test_noisy = np.zeros_like(X_test)
        for i in range(len(X_test)):
            # Count the number of pixels that are different
            nb_diff = len(np.where(X_test[i] != X_test_adv[i])[0])
            # Randomly flip an equal number of pixels (flip means move to max
            # value of 1)
            X_test_noisy[i] = flip(X_test[i], nb_diff)
    else:
        warnings.warn("Using pre-set Gaussian scale sizes to craft noisy "
                      "samples. If you've altered the eps/eps-iter parameters "
                      "of the attacks used, you'll need to update these. In "
                      "the future, scale sizes will be inferred automatically "
                      "from the adversarial samples.")
        # Add Gaussian noise to the samples
        X_test_noisy = np.minimum(
            np.maximum(
                X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                          size=X_test.shape),
                0
            ),
            1
        )
    if dataset in ['nsl']:
        print("keep features..........4 ....nsl")
        X_test_noisy = keep_features([1,2,3], X_test, X_test_noisy)
    return X_test_noisy


def keep_features(unchange_feature_indexes, X, craftedX):
    for i in range(len(X)):
        for indx in unchange_feature_indexes:
            craftedX[i][indx] = X[i][indx]
    return craftedX

def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
    """
    TODO
    :param model:
    :param X:
    :param nb_iter:
    :param batch_size:
    :return:
    """
    output_dim = model.layers[-1].output.shape[-1]#.value
    #get_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
    intermediate_model = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[-1].output)

    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        for i in range(n_batches):
            output[i * batch_size:(i + 1) * batch_size] = \
                intermediate_model(X[i * batch_size:(i + 1) * batch_size], training=True)
                #model(X[i * batch_size:(i + 1) * batch_size], training=True) #get_output([X[i * batch_size:(i + 1) * batch_size], 1])[0]
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())

    return np.asarray(preds_mc)


def get_deep_representations(model, X, batch_size=256, last_hidden_layer = -4):
    """
    TODO
    :param model:
    :param X:
    :param batch_size:
    :return:
    """
    # last hidden layer is always at index -4
    output_dim = model.layers[last_hidden_layer].output.shape[-1]#.value
    #get_encoding = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-4].output])
    # Create a new model to extract intermediate activations
    intermediate_model = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[last_hidden_layer].output)

    # Get the activations for a given input
    #intermediate_output = intermediate_model.predict(input_data)

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            intermediate_model.predict(X[i * batch_size:(i + 1) * batch_size])#get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    return output


def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds, n_jobs=None):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results


def normalize(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]

def normalize_std(normal, adv, noisy, clean):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    scaler = StandardScaler()
    #total = scaler.fit_transform(np.concatenate((normal, adv, noisy, clean)))
    total = scale(np.concatenate((normal, adv, noisy, clean)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:3*n_samples], total[3*n_samples:]

def get_neg_values_for_lr(densities_neg, uncerts_neg):
    """
    :param densities_neg:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])

    #values = np.concatenate((values_neg, values_pos))
    labels = np.zeros_like(densities_neg)

    return values_neg, labels

def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr

def get_neg_values_for_lr(densities_neg, uncerts_neg):
    """
    :param densities_neg:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])

    #values = np.concatenate((values_neg, values_pos))
    labels = np.zeros_like(densities_neg)

    return values_neg, labels

def compute_roc(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

def print_performance_metrics(lr, vals, labs):

    y_pred = lr.predict(vals)

    # Accuracy
    accuracy = metrics.accuracy_score(labs, y_pred)
    print(f"Accuracy: {accuracy}")

    # Precision
    precision = metrics.precision_score(labs, y_pred)
    print(f"Precision: {precision}")

    # Recall
    recall = metrics.recall_score(labs, y_pred)
    print(f"Recall: {recall}")

    # F1-score
    f1 = metrics.f1_score(labs, y_pred)
    print(f"F1-score: {f1}")

    # Confusion matrix
    cm = metrics.confusion_matrix(labs, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # Classification report (provides precision, recall, F1-score, and support for each class)
    cr = metrics.classification_report(labs, y_pred)
    print(f"Classification Report:\n{cr}")

def get_max_min(X_train):
    mx = []
    mn = []
    for x in X_train:
        mx.append(np.max(x))#mx = np.max(mx, np.max(x))
        mn.append(np.min(x))
        #print(np.max(x))
        #break
    return np.max(mx), np.min(mn)

def get_noisy_samples(X_test, max_val=1, min_val=0, STDEVS = 0.310):
    # Add Gaussian noise to the samples
    X_test_noisy = np.minimum(
        np.maximum(X_test + np.random.normal(loc=0, scale=STDEVS, size=X_test.shape), min_val),
        max_val
    )
    return X_test_noisy

def get_success_advs(model, samples, true_labels, target=None):
    preds = model.predict(samples) #model(samples).eval()
    if target is None:
        pos1 = np.where(np.argmax(preds, 1) != np.argmax(true_labels, 1))
    else:
        pos1 = np.where(np.argmax(preds, 1) == target)
    x_sel = samples[pos1]
    y_sel = true_labels[pos1]

    return x_sel, y_sel, pos1

def random_select(max, num):
    lst = [i for i in range(max)]
    random.shuffle(lst)
    idxs = lst[0:num]
    return idxs

def get_claen_data(model, X, y, adv_X):
    predictions = model.predict(adv_X)
    preds_test = np.argmax(predictions, axis=1)
    adv_inds = np.where(preds_test != y.argmax(axis=1))[0]
    clean_samples = X[adv_inds]
    labs_clean = y[adv_inds]
    return clean_samples, labs_clean

def get_uncertainties_and_densities(model, X_train, Y_train, X_test, X_test_adv, clean_data, bandwidth=1.0, batch_size = 128):
    print('Getting -uncertainty- Monte Carlo dropout variance predictions...')
    uncerts_normal = get_mc_predictions(model, X_test, batch_size=batch_size).var(axis=0).mean(axis=1)
    uncerts_adv = get_mc_predictions(model, X_test_adv, batch_size=batch_size).var(axis=0).mean(axis=1)
    uncerts_clean = get_mc_predictions(model, clean_data, batch_size=batch_size).var(axis=0).mean(axis=1)

    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(model, X_train, batch_size=batch_size, last_hidden_layer = -2)
    X_test_normal_features = get_deep_representations(model, X_test, batch_size=batch_size, last_hidden_layer = -2)
    X_test_adv_features = get_deep_representations(model, X_test_adv, batch_size=batch_size, last_hidden_layer = -2)
    X_test_clean_features = get_deep_representations(model, clean_data, batch_size=batch_size, last_hidden_layer = -2)

    numClass = len(np.unique(Y_train))
    print("--------------number of class label: ", numClass)
    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]

    kdes = {}
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=bandwidth) \
            .fit(X_train_features[class_inds[i]])
        
    preds_test_normal = np.argmax(model.predict(X_test), axis=1)
    preds_test_adv = np.argmax(model.predict(X_test_adv), axis=1)
    preds_test_clean = np.argmax(model.predict(clean_data), axis=1)

    # Get density estimates
    print('computing densities estimates...: data1', end='...  ')
    densities_normal = score_samples(kdes, X_test_normal_features, preds_test_normal)
    print('data2', end='...  ')
    densities_adv = score_samples(kdes, X_test_adv_features, preds_test_adv)
    print('data3', end='...  ')
    densities_clean = score_samples(kdes, X_test_clean_features, preds_test_clean)
    print('done.')
    # model, X_train, Y_train, X_test, X_test_adv, clean_data, bandwidth=BANDWIDTHS[args.dataset],

    ## Z-score the uncertainty and density values
    uncerts_normal_z, uncerts_adv_z, uncerts_clean_z = \
        normalize(uncerts_normal, uncerts_adv,uncerts_clean)
    densities_normal_z, densities_adv_z, densities_clean_z = \
        normalize(densities_normal, densities_adv,densities_clean)
    
    return uncerts_normal_z, uncerts_adv_z, uncerts_clean_z, densities_normal_z, densities_adv_z, densities_clean_z


#only suc_adv and clean_data operation
def normalize2(normal, adv):
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv)))

    return total[:n_samples], total[n_samples:]
def get_uncertainties_and_densities2(model, X_train, Y_train, success_advs, clean_data, bandwidth=1.0, batch_size = 128):
    print('Getting -uncertainty- Monte Carlo dropout variance predictions...')
    uncerts_adv = get_mc_predictions(model, success_advs, batch_size=batch_size).var(axis=0).mean(axis=1)
    uncerts_clean = get_mc_predictions(model, clean_data, batch_size=batch_size).var(axis=0).mean(axis=1)

    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(model, X_train, batch_size=batch_size, last_hidden_layer = -2)
    X_test_adv_features = get_deep_representations(model, success_advs, batch_size=batch_size, last_hidden_layer = -2)
    X_test_clean_features = get_deep_representations(model, clean_data, batch_size=batch_size, last_hidden_layer = -2)

    numClass = len(np.unique(Y_train))
    print("--------------number of class label: ", numClass)
    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]

    kdes = {}
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=bandwidth) \
            .fit(X_train_features[class_inds[i]])
        

    preds_test_adv = np.argmax(model.predict(success_advs), axis=1)
    preds_test_clean = np.argmax(model.predict(clean_data), axis=1)

    # Get density estimates
    print('computing densities estimates...: data1', end='...  ')
    densities_adv = score_samples(kdes, X_test_adv_features, preds_test_adv)
    print('data2', end='...  ')
    densities_clean = score_samples(kdes, X_test_clean_features, preds_test_clean)
    print('done.')
    ## Z-score the uncertainty and density values
    uncerts_adv_z, uncerts_clean_z = \
        normalize2(uncerts_adv,uncerts_clean)
    densities_adv_z, densities_clean_z = \
        normalize2(densities_adv,densities_clean)
    
    return uncerts_adv_z, uncerts_clean_z, densities_adv_z, densities_clean_z

'''model architecture
class NSLModel(torch.nn.Module):
   def __init__(self):
       super(NSLModel, self).__init__()
       self.fc1 = torch.nn.Linear(121, 50)
       self.fc2 = torch.nn.Linear(50, 2)

   def forward(self, x):
       x = torch.relu(self.fc1(x))
       x = torch.softmax(self.fc2(x), dim=1)
       return x
   
def create_nsl_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(50, activation='relu', input_shape=(121,)),  # Input layer
        tf.keras.layers.Dense(2, activation='softmax')  # Output layer
    ])
    return model


#training for 20 epochs, 
model = create_nsl_model()
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit( X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test)
    )
model.save('NSLKDD/Models/nsl_kdd_Dos2.h5') # epoch=20

'''