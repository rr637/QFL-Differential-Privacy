import os
import sys
import io

import pennylane as qml
from pennylane import numpy as np
# import numpy as np
# from pennylane.optimize import NesterovMomentumOptimizer

import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import UtilsN

import time

from pyvacy import optim, analysis, sampling

log_dir = '/Users/rodrofougaran/Downloads/QPPAI-FL'
writer = SummaryWriter()
# Predefined optimization procedures

def traceBackwards(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                traceBackwards(n[0])


def _accuracy(model, xb, yb):
    """
    THIS IS WILL'S ACCURACY FUNCTION.
    defines the accuracy between an output of the net and the labels yb
    Define the prediction for a single sample to the maximal component
    of the ten-vector output
    The Sequential model only takes single inputs, for-loop has model input
    only a single x from batch at a time
    Potentially xb and yb are both CUDA tensors. We want to run the model on the GPU
    so we keep the xb as CUDA tensors but for the evaluation of accuracy preds == yb
    it is fine if that is on the CPU
    """
    out_list = [model(x) for x in xb]
    # preds are loaded as cpu tensors since GPU isn't needed for the next part
    preds = torch.tensor([torch.argmax(out) for out in out_list], device='cpu')
    yb = yb.cpu()
    print(preds)
    print(preds == yb)
    print((preds == yb).double())
    print((preds == yb).double().mean())
    return (preds == yb).double().mean(), len(xb)


def accuracy(labels, predictions):
    """ Share of equal labels and predictions

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions
    Returns:
        float: accuracy
    """

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l.item() - p) < 1e-2:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


def lost_function_cross_entropy(labels, predictions):
    ## numpy array
    loss = nn.CrossEntropyLoss()
    output = loss(predictions, labels)
    print("LOSS AVG: ", output)
    return output


def cost(VQC, X, Y):
    """Cost (error) function to be minimized."""
    # print(f"Optimization.cost has args X:{X} \nAnd Y:{Y}")
    # predictions = torch.stack([variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=item) for item in X])

    ## This method still not fully use the CPU resource...
    loss = nn.CrossEntropyLoss()
    output = loss(torch.stack([VQC.forward(item) for item in X]), Y)
    print("LOSS AVG: ", output)
    return output


def MSEcost(VQC, X, Y):
    """Cost (error) function to be minimized."""

    # predictions = torch.stack([variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=item) for item in X])

    ## This method still not fully use the CPU resource...
    loss = nn.MSELoss()
    output = loss(torch.stack([VQC.forward(item) for item in X]), Y)
    print("LOSS AVG: ", output)
    return output


# This cost is with the user-defined LossFunction
# PyTorch Loss Function
def cost_function(VQC, LossFunction, X, Y):
    """Cost (error) function to be minimized."""

    # predictions = torch.stack([variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=item) for item in X])

    ## This method still not fully use the CPU resource...

    output = LossFunction(torch.stack([VQC.forward(item) for item in X]), Y)
    print("LOSS AVG: ", output)
    return output


def train_epoch_full_dp(opt, VQC, train_ds, batch_size, params):
    """
    Will implemented a differentially private train_epoch_full
    batch training function
    Important to use this with pyvacy since pyvacy requires extra
    processing through microbatches contained in each minibatch
    the opt is necessarily a DPOptimizerClass torch optim
    VQC is assumed to have a parent class L2Tracker
    This is to simply keep track of the L2 norms, so make sure that
    the L2 norm cutoff is not too high, this leads to bad results
    """
    print("Differentially Private Epoch...")
    losses = []
    minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        params['mini_bs'],
        params['micro_bs'],
        params['iters']
    )
    iteration = 0
    for X_minibatch, y_minibatch in minibatch_loader(train_ds):
        opt.zero_grad()
        print("CALCULATING LOSS...")
        for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
            since_batch = time.time()
            # testing new bit
            if (params['useGPU'] and not X_microbatch.is_cuda):
                print("moving microbatch to cuda...")
                X_microbatch = X_microbatch.cuda()
                Y_microbatch = Y_microbatch.cuda()
            opt.zero_microbatch_grad()
            loss = cost(VQC=VQC, X=X_microbatch, Y=y_microbatch)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            # record real gradients, loss.backward() isn't overridden
            VQC.update_l2_norm_list(opt, isDP=False)
            # microbatch_step() clips and applies noise to gradients
            opt.microbatch_step()
        print("LOSS AVG: ", loss)
        print("BACKWARD..")
        opt.step()
        VQC.update_l2_norm_list(opt, isDP=True)
        print("FINISHED OPT.")
        print("Batch time: ", time.time() - since_batch)

    losses = np.array(losses)
    return losses.mean()


def train_epoch_full(opt, VQC, X, Y, batch_size):
    losses = []
    for beg_i in range(0, X.shape[0], batch_size):
        X_train_batch = X[beg_i:beg_i + batch_size]
        # print(x_batch.shape)
        Y_train_batch = Y[beg_i:beg_i + batch_size]
        # print(f"X and Y batch devices, respectively: {X_train_batch.device}, {Y_train_batch.device}")
        # opt.step(closure)
        since_batch = time.time()
        opt.zero_grad()
        print("CALCULATING LOSS...")
        loss = cost(VQC=VQC, X=X_train_batch, Y=Y_train_batch)
        print("BACKWARD..")
        loss.backward()
        # traceBackwards(loss.grad_fn)
        # why does this have loss.data.cpu()????
        losses.append(loss.data.cpu().numpy())
        opt.step()
        #       print("LOSS IN CLOSURE: ", loss)
        print("FINISHED OPT.")
        print("Batch time: ", time.time() - since_batch)
        # print("CALCULATING PREDICTION.")
    losses = np.array(losses)
    return losses.mean()


def train_epoch(opt, VQC, X, Y, batch_size, sampling_iteration):
    """ train epoch, each epoch is 100 times random sampling"""
    losses = []
    for i in range(sampling_iteration):
        # Test Saving
        UtilsN.Saving.save_test()
        since_batch = time.time()

        batch_index = np.random.randint(0, len(X), (batch_size,))
        X_train_batch = X[batch_index]
        Y_train_batch = Y[batch_index]
        # opt.step(closure)
        opt.zero_grad()
        print("CALCULATING LOSS...")
        loss = cost(VQC=VQC, X=X_train_batch, Y=Y_train_batch)
        print("BACKWARD..")
        loss.backward()
        # why does this have loss.data.cpu()????
        losses.append(loss.data.cpu().numpy())
        opt.step()
        #       print("LOSS IN CLOSURE: ", loss)
        print("FINISHED OPT.")
        print("Batch time: ", time.time() - since_batch)
        # print("CALCULATING PREDICTION.")
    losses = np.array(losses)
    return losses.mean()


def BinaryCrossEntropy(opt, vqc, X, Y, batch_size):
    return train_epoch(opt, vqc, X, Y, batch_size)


# This is for classification, not for reinforcement learning or timr-series modeling.
def train_model(opt,
                VQC,
                x_for_train,
                y_for_train,
                x_for_val,
                y_for_val,
                x_for_test,
                y_for_test,
                exp_name,
                exp_index,
                params,
                saving_files=True,
                batch_size=10,
                epoch_num=100,
                sampling_iteration=100,
                full_epoch=False,
                show_params=False,
                torch_first_model=False):
    iter_index = []
    cost_train_list = []
    cost_test_list = []
    acc_train_list = []
    acc_val_list = []
    acc_test_list = []

    file_title = exp_name + datetime.now().strftime("NO%Y%m%d%H%M%S")
    exp_name = exp_name + "Exp_" + str(exp_index)

    # 2019-12-26
    # What is the var_Q_circuit here?????
    var_Q_circuit = ''
    if torch_first_model == True:
        var_Q_circuit = VQC.state_dict()
    else:
        var_Q_circuit = VQC.var_Q_array
    # No var_q_bias currently
    # should implement a more general method to handle the situations
    var_Q_bias = ''

    # print(var_Q_circuit)

    if not os.path.exists(exp_name):

        os.makedirs(exp_name)

    # Need to be able to save the dataset for publication
    if saving_files == True:
        UtilsN.Saving.save_training_and_testing(exp_name=exp_name, file_title=file_title, training_x=x_for_train,
                                               training_y=y_for_train, val_x=x_for_val, val_y=y_for_val,
                                               testing_x=x_for_test, testing_y=y_for_test)
    # print(VQC.var_Q_array)
    if show_params == True:
        print(var_Q_circuit)

    train_ds = TensorDataset(x_for_train, y_for_train)
    for it in range(epoch_num):
        # Need to save data
        if full_epoch == True and type(opt).__name__ == "DPOptimizerClass":
            avg_loss_in_epoch = train_epoch_full_dp(opt, VQC, train_ds, batch_size, params)
        elif full_epoch == True:
            avg_loss_in_epoch = train_epoch_full(opt, VQC, x_for_train, y_for_train, batch_size)
        else:
            avg_loss_in_epoch = train_epoch(opt, VQC, x_for_train, y_for_train, batch_size, sampling_iteration)
        if show_params == True:
            print(var_Q_circuit)

        # print(var_Q_circuit)
        # print(var_Q_circuit_1)
        # print(var_Q_bias_1)
        # print(var_Q_circuit_2)
        # print(var_Q_bias_2)
        # print(var_Q_circuit_3)
        # print(var_Q_bias_3)

        # print(VQC.var_Q_array)

        # Output from the Circuit and the Y_label may have different type, causing the accuracy function not working
        print("CALCULATE PRED TRAIN ... ")
        predictions_train = [torch.argmax(VQC.forward(item)).item() for item in x_for_train]
        print("CALCULATE PRED VALIDATION ... ")
        predictions_val = [torch.argmax(VQC.forward(item)).item() for item in x_for_val]
        print("CALCULATE PRED TEST ... ")
        predictions_test = [torch.argmax(VQC.forward(item)).item() for item in x_for_test]
        print("CALCULATE TRAIN COST ... ")

        # COST calculation not fully use all CPU resource
        # Probably need to initialize a new VQC with no-grad params
        # cost_train = cost_for_result(VQC, x_train, y_train).item()
        cost_train = cost(VQC, x_for_train, y_for_train).item()

        print("CALCULATE TEST COST ... ")
        # COST calculation not fully use all CPU resource
        # cost_test = cost_for_result(VQC, x_test, y_test).item()
        cost_test = cost(VQC, x_for_test, y_for_test).item()

        # print('Y_for_train: ',Y_for_train)
        # print('predictions_train: ', predictions_train)

        # predictions are already made in the argmax
        # but y_for_train is the hot-shot encoding!
        # gotta convert it back to labels in the function
        acc_train = accuracy(y_for_train, predictions_train)
        acc_val = accuracy(y_for_val, predictions_val)
        acc_test = accuracy(y_for_test, predictions_test)

        iter_index.append(it + 1)
        acc_train_list.append(acc_train)
        acc_val_list.append(acc_val)
        acc_test_list.append(acc_test)
        cost_train_list.append(cost_train)
        cost_test_list.append(cost_test)

        # Need to be able to save the info during the training
        # Add a torch.save() for saving the torch model
        if saving_files == True:
            UtilsN.Saving.save_all_the_current_info(exp_name, file_title, iter_index, var_Q_circuit, var_Q_bias,
                                                   cost_train_list, cost_test_list, acc_train_list, acc_val_list,
                                                   acc_test_list)
            if torch_first_model == True:
                UtilsN.Saving.saving_torch_model(exp_name, file_title, iter_index, VQC.state_dict())

        print(
            "Epoch: {:5d} | Cost train: {:0.7f} | Cost test: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} | Acc test: {:0.7f}"
            "".format(it + 1, cost_train, cost_test, acc_train, acc_val, acc_test))

    return cost_train_list