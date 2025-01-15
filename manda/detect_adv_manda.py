from __future__ import division, absolute_import, print_function

import os
import argparse
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from keras.models import load_model
import sys

from util import*

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00, 'nsl': 1.00, 'c17':1.00, 'c19':1.00, 'nb15':1.00}


def main(args):
    assert args.dataset in ['mnist', 'cifar', 'c17', 'nsl', 'c19', 'nb15'], \
        "Dataset parameter must be either 'c17', 'c19', 'nb15'"
    assert args.attack in ['fgsm', 'jsma', 'cw', 'pgd', 'diff'], \
        "Attack parameter must be either 'pgd', 'diff', or 'cw'"

    print('Loading the data and model...')
    if args.dataset == 'nsl':
        folder_path = "NSLKDD/"
        #model = load_model('../data/model_nsl.h5')#correct
        model = load_model('NSLKDD/Models/nsl_kdd_Dos2.h5')
    elif args.dataset == 'c19':
        folder_path = "CICDDoS2019/" #manda/
        model = load_model('CICDDoS2019/Models/cicddos_binary.h5')
    elif args.dataset == 'nb15':
        folder_path = "UNSW-NB15/" #manda/UNSW-NB15/Models/unsw_binary.h5
        model = load_model('UNSW-NB15/Models/unsw_binary.h5')
        #manda/UNSW-NB15/PGD/X_test.npy 
    else:
        folder_path = "CICIDS2017/"
        model = load_model('CICIDS2017/Models/cicids_binary.h5') #correct

    data_path = folder_path + "Original/" # + "OriginalData/" #/manda/NSLKDD/OriginalData/X_test.npy

    # Load the dataset
    X_train, y_train = np.load(data_path+'X_train.npy'), np.load(data_path+'y_train.npy')
    X_test, y_test =  np.load(data_path+'X_test.npy'), np.load(data_path+'y_test.npy')
    print("Original X_train shape:", X_train.shape, "  Original y_train shape:", y_train.shape )
    print("Original X_test shape:", X_test.shape, "  Original Y_test shape:", y_test.shape)

    # Load adv
    print('Loading noisy and adversarial samples...')
    if args.attack == 'diff':
        adv_path = folder_path + "NetDiffuser/" #  "DiffAttack/" 
    elif args.attack == 'pgd':
        adv_path = folder_path + "PGD/" # "Org_PGDAttack/" #manda/CICIDS2017/NetDiffuser/X_test.npy
    else:
        adv_path = folder_path + "Synt_PGDAttack/" # "Org_PGDAttack/" 

    X_test_adv = np.load(adv_path + 'X_test.npy')
    y_test_adv = np.load(adv_path + 'y_test.npy')
    print("Original X_test_adv shape:", X_test_adv.shape)

    # change y to categorical
    nClass = len(np.unique(y_train))
    Y_train, Y_test =  to_categorical(y_train, nClass), to_categorical(y_test, nClass)
    Y_test_adv = to_categorical(y_test_adv, nClass) 
    
    _, acc = model.evaluate(X_test, Y_test, batch_size=args.batch_size, verbose=0)
    print("Model accuracy on the test set / Before Attack: %0.2f%%" % (100 * acc))
    _, acc = model.evaluate(X_test_adv, Y_test_adv, batch_size=args.batch_size, verbose=0)
    print("Model accuracy on the adv set / After Attack: %0.2f%%" % (100 * acc))
    
    # get_success_advs
    X_test_adv, labs_adv, pos = get_success_advs(model, X_test_adv, Y_test_adv)
    print("success_advs shape:", X_test_adv.shape)
    # get clean data
    idxs = random_select(len(X_test), len(X_test_adv))
    clean_data = X_test[idxs]
    labs_clean = Y_test[idxs]
    print("clean data shape: ", clean_data.shape)
    X_test_clean = clean_data
    Y_test_clean = labs_clean
    # get noise data
    max_val, min_val = get_max_min(X_train)
    print("Max min Values: ", max_val, min_val)
    X_test_noisy = get_noisy_samples(X_test_clean, max_val=max_val, min_val=min_val, STDEVS = 0.310)
    
    _, acc = model.evaluate(clean_data, labs_clean, batch_size=args.batch_size, verbose=0)
    print("Model accuracy on the purified (remove detect adv) test set: %0.2f%%" % (100 * acc))

    
    #X_test_clean, X_test_noisy, X_test_adv
    uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z, densities_normal_z, densities_adv_z, densities_noisy_z = \
        get_uncertainties_and_densities(model, X_train, Y_train, X_test_clean, X_test_adv, X_test_noisy,\
                                                bandwidth=BANDWIDTHS[args.dataset], batch_size=args.batch_size)

    

    
 
    ## Build detector
    values, labels, lr = train_lr(
        densities_pos=densities_adv_z,
        densities_neg=np.concatenate((densities_normal_z, densities_noisy_z)),
        uncerts_pos=uncerts_adv_z,
        uncerts_neg=np.concatenate((uncerts_normal_z, uncerts_noisy_z))
    )

    ## Evaluate detector
    print('################### Detector Model: Overall Performance ###########################')
    print_performance_metrics(lr, values, labels)
    # Compute logistic regression model predictions
    probs = lr.predict_proba(values)[:, 1]
    # Compute AUC
    n_samples = len(X_test_clean) #len(X_test)
    # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
    # and the last 1/3 is the positive class (adversarial samples).
    _, _, auc_score = compute_roc(
        probs_neg=probs[:2 * n_samples],
        probs_pos=probs[2 * n_samples:]
    )
    print('Detector ROC-AUC score: %0.4f' % auc_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)
