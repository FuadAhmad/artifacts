from __future__ import division, absolute_import, print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import sys
sys.path.append('../')  # Add the parent directory to the Python path
from util import get_data
from attacks import (fast_gradient_sign_method, basic_iterative_method,
                            saliency_map_method, pgd_attack)

# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010},
    'nsl': {'eps': 0.830, 'eps_iter': 0.020}
}


def craft_one_type(model, X, Y, dataset, attack, batch_size):
    """
    TODO
    :param sess:
    :param model:
    :param X:
    :param Y:
    :param dataset:
    :param attack:
    :param batch_size:
    :return:
    """
    if attack == 'fgsm':
        # FGSM attack
        print('Crafting fgsm adversarial samples...')
        if dataset in ['nsl']:
            print('................................... for NSL Dataset')
            X_adv = fast_gradient_sign_method(
                    model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], batch_size=batch_size)
            X_adv = np.array(X_adv)
            for i in range(len(X)):
                X_adv[i][1] = X[i][1]
                X_adv[i][2] = X[i][2]
                X_adv[i][3] = X[i][3]
            print(type(X_adv))
        else:
            X_adv = fast_gradient_sign_method(
            model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], clip_min=0.,
            clip_max=1., batch_size=batch_size
        )
    elif attack == 'pgd':
        # PGD attack
        print('Crafting pgd adversarial samples...')
        if dataset in ['nsl']:
            print('................................... for NSL Dataset')
            X_adv = pgd_attack(
                    model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], batch_size=batch_size)
            X_adv = np.array(X_adv)
            for i in range(len(X)):
                X_adv[i][1] = X[i][1]
                X_adv[i][2] = X[i][2]
                X_adv[i][3] = X[i][3]

    elif attack in ['bim-a', 'bim-b']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method(
            model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'],
            eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], clip_min=0.,
            clip_max=1., batch_size=batch_size
        )
        if attack == 'bim-a':
            # BIM-A
            # For each sample, select the time step where that sample first
            # became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B
            # For each sample, select the very last time step
            X_adv = results[-1]
    elif attack == 'jsma':
        # JSMA attack
        print('Crafting jsma adversarial samples. This may take a while...')
        X_adv = saliency_map_method(
            model, X, Y, theta=1, gamma=0.1, clip_min=0., clip_max=1.
        )
    else:
        # TODO: CW attack
        raise NotImplementedError('CW attack not yet implemented.')
    _, acc = model.evaluate(X_adv, Y, batch_size=batch_size,
                            verbose=0)
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    np.save('../data/Adv_%s_%s.npy' % (args.dataset, args.attack), X_adv)


def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn', 'nsl'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all', 'pgd'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile('../data/model_%s.h5' % args.dataset), \
        'model file not found... must first train model using train_model.py.'
    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    # Create TF session, set it as Keras backend
    #sess = tf.Session()
    #K.set_session(sess)
    #K.set_learning_phase(0)
    model = load_model('../data/model_%s.h5' % args.dataset)
    _, _, X_test, Y_test = get_data(args.dataset)
    _, acc = model.evaluate(X_test, Y_test, batch_size=args.batch_size,
                            verbose=0)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))
    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw']:
            craft_one_type(model, X_test, Y_test, args.dataset, attack,
                           args.batch_size)
    else:
        # Craft one specific attack type
        craft_one_type(model, X_test, Y_test, args.dataset, args.attack,
                       args.batch_size)
    print('Adversarial samples crafted and saved to data/ subfolder.')
    #sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw' "
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
