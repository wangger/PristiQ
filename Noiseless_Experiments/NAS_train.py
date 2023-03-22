# from lib_qc import *
# from lib_util import *
# from lib_net import *
import argparse
import time
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import os
import sys
sys.path.append("../interfae/")
# from lib_model_summary import summary
from collections import Counter
from pathlib import Path
# from qiskit_simulator_wbn import run_simulator
from c_input import *
from S_Encode import Encoding_Circuit
from NAS_Net import VQC_Net
from utils import fix_random_seeds
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import time
import copy
from NAS_Net_Qiskit import IBMQ_forward, Noisy_Aer_forward


def train(model, optimizer, epoch, criterion, train_loader, logger, args, enc_circuit=None):
    # TODO(Note): Training for 1 epoch
    model.train()

    correct = 0
    loss = 0
    # iter_loss = []
    iter = 0
    total_data = 0  # TODO(Note): We use drop last for data loader
    Is_S_Encoding = args.Is_S_Encoding
    device = args.device

    for batch_idx, (data, target) in enumerate(train_loader):
        # target, new_target = modify_target(target, interest_num)
        data, target = data.to(device), target.to(device)

        # TODO(Note): encoding the data before computing
        if Is_S_Encoding:
            # print("DATA IS ENCODED!!!!")
            data = enc_circuit(data)    # [BS, 2 ** num_tol_qubits]

        output = model(data)  # output value is prob
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max output value
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        cur_loss = criterion(output, target)
        # iter_loss.append(cur_loss.item())
        loss += cur_loss  # It is not normalized

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        iter += 1
        total_data += len(data)

        if batch_idx % 20 == 0:
                # TODO(Note): The loss and accuracy are the metrics until now
                # logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)'.format(
                #            epoch, batch_idx * len(data), len(train_loader.dataset),
                #            100. * batch_idx / len(train_loader), loss/(batch_idx + 1), correct, total_data,
                #            100. * float(correct) / float(total_data)))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%)'.format(
                           epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss/(batch_idx + 1), correct, total_data,
                           100. * float(correct) / float(total_data)))

    print("-" * 20, "Train Epoch: {} done".format(epoch), "-" * 20)
    # logger.info("-" * 20, "Train Epoch: {} done".format(epoch), "-" * 20)
    final_loss = loss/iter
    final_loss = float(final_loss)
    final_acc = 100. * float(correct)/float(total_data)
    print("Training Set: Average accuracy: {}, Average loss: {}".format(round(final_acc, 4), round(final_loss, 6)))
    # logger.info("Training Set: Average accuracy, Average loss: {}".format(round(final_acc, 4), round(final_loss, 6)))
    return final_loss


def test(model, criterion, test_loader, logger, args, enc_circuit=None):
    model.eval()
    test_loss = 0
    correct = 0
    iter = 0
    total_data = 0  # TODO(NOte): We use drop last for data loader
    Is_S_Encoding = args.Is_S_Encoding
    device = args.device

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # target, new_target = modify_target(target, interest_num)
            data, target = data.to(device), target.to(device)
            # TODO(Note): encoding the data before computing
            if Is_S_Encoding:
                data = enc_circuit(data)    # [BS, 2 ** num_tol_qubits]

            output = model(data)  # output value is prob
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # print(pred.shape)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss += criterion(output, target)  # sum up batch loss

            iter += 1
            total_data += len(data)

    final_loss = test_loss/iter
    final_acc = 100. * float(correct)/float(total_data)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          final_loss, correct, len(test_loader.dataset), final_acc))
    # logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #       final_loss, correct, len(test_loader.dataset), final_acc))

    return final_acc/100, final_loss


def Training_Ideal(train_loader, valid_loader, model, logger, training_args, args, enc_circuit=None):

    # TODO(NOTE): Decode the input parameters
    # data related
    device = args.device

    # training related
    init_lr = training_args.init_lr
    weight_decay = training_args.weight_decay
    max_epoch = training_args.max_epoch
    milestones = training_args.milestones
    # if args.milestones is not None:
    #     milestones = [int(x.strip()) for x in args.milestones.split(",")]   # for -m
    # else:
    #     milestones = [max_epoch * 0.5, max_epoch * 0.75]
    opt_type = training_args.optimizer
    sch_type = training_args.scheduler
    lr_decay_rate = training_args.lr_decay_rate

    # TODO (NOTE): Prepare for training
    criterion = nn.CrossEntropyLoss()   # TODO: Handle different choice of loss
    # criterion = nn.NLLLoss()

    model = model.to(device)
    print("The model of VQC is as follows")
    print(model)

    if opt_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
    else:
        raise Exception("The optimizer is not supported!")

    if sch_type == 'none':
        scheduler = None
    elif sch_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch)
    elif sch_type == 'linear':
        scheduler = MultiStepLR(optimizer, milestones, gamma=lr_decay_rate)
    else:
        raise Exception("The scheduler is not supported!")

    best_epoch, best_acc = 0, 0
    best_model = None
    epoch_init = 0
    loss_array = []

    init_acc, init_loss = test(model, criterion, train_loader, logger, args, enc_circuit=enc_circuit)
    print("The initial accuracy on training set is ", init_acc)
    print("The initial loss on training set is ", init_loss)
    loss_array.append(init_loss)

    init_acc, init_loss = test(model, criterion, valid_loader, logger, args, enc_circuit=enc_circuit)
    print("The initial accuracy on test set is ", init_acc)
    print("The initial loss on test set is ", init_loss)

    for epoch in range(epoch_init, max_epoch):
        print("=" * 20, epoch, "epoch", "=" * 20)
        print("Epoch Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))

        print("-" * 20, "learning rates", "-" * 20)
        for param_group in optimizer.param_groups:
            print(param_group['lr'], end=",")  # we can set different lr for different part of params
        print()

        print("-" * 20, "training", "-" * 20)
        print("Training Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        # TODO(Note): Train QNN
        loss = train(model, optimizer, epoch, criterion, train_loader, logger, args, enc_circuit=enc_circuit)
        loss_array.append(loss)
        print("Training End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        print("-" * 60)

        print()

        print("-" * 20, "testing", "-" * 20)
        print("Testing Start at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        cur_acc, _ = test(model, criterion, valid_loader, logger, args, enc_circuit=enc_circuit)
        print("Testing End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        print("-" * 60)
        print()
        if scheduler is not None:
            scheduler.step()

        # is_best = False
        if cur_acc > best_acc:
            # is_best = True
            best_acc = cur_acc
            best_epoch = epoch
            del best_model  # Can it release some space?
            # TODO(Note): Should save the best model, but test set will become validation set
            best_model = copy.deepcopy(model)  # reference is not enough

        print("Best accuracy: {} at epoch {}; Current accuracy {}. Checkpointing".format(best_acc, best_epoch, cur_acc))

        # TODO(Note): Save checkpoint at each epoch
        # if save_chkp:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'best_epoch': best_epoch,
        #         'acc': acc,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #     }, is_best, save_path, 'checkpoint_{}_{}.pth.tar'.format(epoch, round(cur_acc, 4)))  # Save 4 floating point
        print("Epoch End at:", time.strftime("%m/%d/%Y %H:%M:%S"))
        print("=" * 60)
        print()

    print("The final loss array is shown as follows.")
    print(loss_array)

    return best_acc, best_epoch, best_model  # acc < 1


def Training_Noise_Sim():   # noise aware training with noisy simulators
    pass


def Training_Noise_QC():   # noise aware training with noisy real quantum machine
    pass


def Testing_QC(enc_model_qc, test_loader, logger, args):    # test on noisy real quantum machine
    """
    return: acc < 1
    """
    # test_loss = 0
    correct = 0
    iter = 0
    total_data = 0  # TODO(NOte): We use drop last for data loader
    # Is_S_Encoding = args.Is_S_Encoding

    for batch_idx, (data, target) in enumerate(test_loader):
        output = IBMQ_forward(data, enc_model_qc, args)  # output value is prob
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        iter += 1
        total_data += len(data)

    final_acc = 100. * float(correct) / float(total_data)
    logger.info('Test set: Average Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset), final_acc))
    return final_acc / 100


def Testing_Noisy_QC(enc_model_qc, test_loader, logger, args):  # test on noisy quantum simulator
    """
    return: acc < 1
    """
    # test_loss = 0
    correct = 0
    iter = 0
    total_data = 0  # TODO(NOte): We use drop last for data loader
    # Is_S_Encoding = args.Is_S_Encoding

    for batch_idx, (data, target) in enumerate(test_loader):
        output = Noisy_Aer_forward(data, enc_model_qc, args)  # output value is prob
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        iter += 1
        total_data += len(data)

    final_acc = 100. * float(correct) / float(total_data)
    logger.info('Test set: Average Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset), final_acc))
    return final_acc / 100
