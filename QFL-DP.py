#!/bin/bash
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from pyvacy import optim, analysis, sampling
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import models
from metaquantum.CircuitComponents import VariationalQuantumClassifierInterBlock_M_IN_N_OUT
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch import tensor
import csv
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import pennylane as qml
import torch.multiprocessing as mp


dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

qdevice = "default.qubit"
# torch.backends.cudnn.benchmark=True

# VQC Class Definition and Initialization

class VariationalQuantumClassifierInterBlock_M_IN_N_OUT:
	def __init__(
			self,
			num_of_input= 10,
			num_of_output= 4,
			num_of_wires = 10,
			num_of_layers = 2,
			var_Q_circuit = None,
			var_Q_bias = None,
			qdevice = "default.qubit",
			hadamard_gate = False,
			more_entangle = False,
			gpu = False):

		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias
		self.num_of_input = num_of_input
		self.num_of_output = num_of_output
		self.num_of_wires = num_of_wires
		self.num_of_layers = num_of_layers

		self.qdevice = qdevice

		self.hadamard_gate = hadamard_gate
		self.more_entangle = more_entangle

		

		if gpu == True and qdevice == "qulacs.simulator":
			print("GOT QULACS AND GPU")
			self.dev = qml.device(self.qdevice, wires = num_of_wires, gpu = True)
		else:
			self.dev = qml.device(self.qdevice, wires = num_of_wires)


	def set_params(self, var_Q_circuit, var_Q_bias):
		self.var_Q_circuit = var_Q_circuit
		self.var_Q_bias = var_Q_bias

	def init_params(self):
		self.var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(self.num_of_layers, self.num_of_wires, 3), device=device).type(dtype), requires_grad=True)
		return self.var_Q_circuit

	def _statepreparation(self, angles):

		"""Quantum circuit to encode a the input vector into variational params

		Args:
			a: feature vector of rad and rad_square => np.array([rad_X_0, rad_X_1, rad_square_X_0, rad_square_X_1])
		"""

		if self.hadamard_gate == True:
			for i in range(self.num_of_input):
				qml.Hadamard(wires=i)

		for i in range(self.num_of_input):
			qml.RY(angles[i,0], wires=i)
			qml.RZ(angles[i,1], wires=i)

	def _layer(self, W):
		""" Single layer of the variational classifier.

		Args:
			W (array[float]): 2-d array of variables for one layer

		"""


		# Entanglement Layer

		for i in range(self.num_of_wires):
			qml.CNOT(wires=[i, (i + 1) % self.num_of_wires])

		if self.more_entangle == True:
			for j in range(self.num_of_wires):
				qml.CNOT(wires=[j, (j + 2) % self.num_of_wires])

		# Rotation Layer
		for j in range(self.num_of_wires):
			qml.Rot(W[j, 0], W[j, 1], W[j, 2], wires=j)

	def circuit(self, angles):

		@qml.qnode(self.dev, interface='torch')
		def _circuit(var_Q_circuit, angles):
			"""The circuit of the variational classifier."""
		
			self._statepreparation(angles)

			weights = var_Q_circuit
			
			for W in weights:
				self._layer(W)

			
			return [qml.expval(qml.PauliZ(k)) for k in range(self.num_of_output)]

		return _circuit(self.var_Q_circuit, angles)

	def _forward(self, angles):
		"""The variational classifier."""
		
		bias = self.var_Q_bias 

		
		raw_output = self.circuit(angles)

		
		
		return raw_output

	def forward(self, angles):
	
		fw = self._forward(angles)
		return fw

dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

qdevice = "default.qubit"

params = {'isQuantum': True, 'usePyvacy': True, 'useGpu': True, 
         'epochs': 1, 'delta': 1e-05,'l2_clip': 0.1, 
          'l2_penalty': 0.001,'lr': 0.02, 'micro_bs': 16, 'mini_bs': 128, 'noise':1.5, 'noise_min' : 1.5,
          'noise_max':2.0, 'noise_incr' : 0.5, 'rounds': 1, 'selected': 5,'clients':100, 'runs':1, 
         'eps_vs_acc': True, 'optimizer': 'SGD'}


vqc = VariationalQuantumClassifierInterBlock_M_IN_N_OUT(
    num_of_input=4,
    num_of_output=2,
    num_of_wires=4,
    num_of_layers=2,
    qdevice=qdevice,
    hadamard_gate=False,
    more_entangle=False
)



# PyTorch Wrapper class for VQC
    
class VQCTorch(nn.Module):
    def __init__(self):
        super().__init__()

        
        self.q_params = nn.Parameter(0.01 * torch.randn(2, 4, 3))

    def get_angles_atan(self, in_x):
        return torch.stack([torch.stack([torch.atan(item), torch.atan(item ** 2)]) for item in in_x])

    def forward(self, batch_item):
        # print('line 99')
        vqc.var_Q_circuit = self.q_params
        # print(self.vqc, 'at line 100')
        score_batch = []

        for single_item in batch_item:
            res_temp = self.get_angles_atan(single_item)
            # print(res_temp)

            # print(self.vqc, 'at line 107')
            q_out_elem = vqc.forward(res_temp)
            # print(q_out_elem)

            clamp = 1e-9
            
            # pdb.set_trace()
            normalized_output = torch.clamp(torch.stack(q_out_elem), min=clamp)
            score_batch.append(normalized_output)

        scores = torch.stack(score_batch).view(len(batch_item), 2)

        return scores

# Data loading 

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(
            224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

####
print(device)

data_dir = '/global/u2/r/rr637/QPPAI-FL/data/'


image_datasets = {x: datasets.ImageFolder(os.path.join(
    data_dir, x), data_transforms[x]) for x in ['train', 'val']}



dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(class_names)
print(f"Train image size: {dataset_sizes['train']}")
print(f'Validation image size: {dataset_sizes["val"]}')
####

# Dividing the training data into num_clients, with each client having equal number of images

traindata = image_datasets['train']
client_train_size = int(dataset_sizes['train'] / params['clients'])
print(f"client_train_size: {client_train_size}")

if params['usePyvacy']:
    epsilon = analysis.epsilon(dataset_sizes['train'],params['mini_bs'],
                                    params['noise'], params['rounds'],params['delta'])
    params['epsilon'] = epsilon
    
    print(epsilon)

traindata_split = torch.utils.data.random_split(traindata,
                                                [client_train_size for _ in range(params['clients'])])

# Creating a pytorch loader for a Deep Learning model
train_loader = [torch.utils.data.DataLoader(
    x, batch_size=params['mini_bs'], shuffle=True) for x in traindata_split]



# Loading the test iamges and thus converting them into a test_loader
testdata = image_datasets['val']
test_loader = torch.utils.data.DataLoader(
    testdata, batch_size=params['mini_bs'], shuffle=True)

# Hybrid Quantum-Classical model and becnhmark fully classical

class QuantumTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4, bias=True),
            VQCTorch())

    def forward(self, in_x):
        return self.net(in_x)

class ClassicalTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')
        for param in self.net.parameters():
            param.requires_grad = False

        self.net.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4, out_features=2, bias=True),
        )
            

    def forward(self, in_x):
        return self.net(in_x)



    
# Differentially privacte local update

def client_update(client_model, optimizer, train_loader, epoch, index, r):
    """
    This function updates/trains client model on client data with differential privacy
    """

    client_model.train()
    client_model = client_model.to(device)
    loss_list = []
    acc_list =[]
    if params['usePyvacy']:
        
        for e in range(params['epochs']):
            print("EPOCH: ", e)
            acc_mini = []
            for batch_idx, (data, target) in enumerate(train_loader):
                # print("BATCH IDX: ", batch_idx)
                data, target = data.to(device), target.to(device)
               
                optimizer.zero_grad()
                acc_micro = []
                for microbatch_idx, (microbatch_data, microbatch_target) in enumerate(
                        zip(data.split(optimizer.microbatch_size), target.split(optimizer.microbatch_size))):
#                    
                    microbatch_data, microbatch_target = microbatch_data.to(
                        device), microbatch_target.to(device)
#                     
                    optimizer.zero_microbatch_grad() 
                    output = client_model(microbatch_data)
                    criterion = nn.CrossEntropyLoss()
#        
                    loss = criterion(output, microbatch_target)
                    
                    # print("WithDP-Loss: ", loss.item())
                   
                    loss.backward()
#                     client_model.update_l2_norm_list(optimizer, isDP=False)
                    optimizer.microbatch_step()
                    preds = torch.argmax(output, dim=1).to(device)
                    # print(f"Shape of target: {microbatch_target.shape} and preds: {preds.shape}", microbatch_target, preds)
                    
                    accuracy = accuracy_score(microbatch_target.cpu().numpy(), preds.cpu().numpy())
                    acc_micro.append(accuracy)
                    # print("Accuracy = {}".format(accuracy))
                    
                    
            
            
                    
                    
                micro_avg = sum(acc_micro)/len(acc_micro)
                acc_mini.append(micro_avg)
                optimizer.step()
#                 client_model.update_l2_norm_list(optimizer, isDP=True)
                optimizer.zero_grad()
                
            
                
            mini_avg = sum(acc_mini)/len(acc_mini)
            acc_list.append(mini_avg)
            loss_list.append(loss.item())
            
        
            
            print(f"Loss per epoch: {loss.item()}")
            print(f"Epoch {e} Accuracy: {mini_avg}")
                
    else:
            
        for e in range(epoch):
            acc_mini = []
            print("Client: ",index+1)
            acc_mini=[]
            for batch_idx, (data, target) in enumerate(train_loader):
                # print("BATCH IDX: ", batch_idx)
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model(data)
                criterion = nn.CrossEntropyLoss()

                # loss = F.nll_loss(output, target)
                loss = criterion(output, target)
                # print("withoutDP-Loss: ", loss.item())
                loss.backward()
                optimizer.step()
                preds = torch.argmax(output, dim=1).to(device)
                accuracy = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
                print(f"accuracy_per_mini: {accuracy}")
                acc_mini.append(accuracy)
        
            
            print(f"Loss per epoch: {loss.item()}")
            print(f"acc_mini_list: {acc_mini}")
            mini_avg = sum(acc_mini)/len(acc_mini)
            print(f"Epoch {e} Accuracy: {mini_avg}")
            acc_list.append(mini_avg)
            loss_list.append(loss.item())
    avg_acc = sum(acc_list)/params['epochs']
    return loss.item(), avg_acc

#Local model aggregation and sharing of global model

def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
   
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))],
                                        0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
            
        model.load_state_dict(global_model.state_dict())

#Testing

def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data.to(torch.float))
            criterion = nn.CrossEntropyLoss(reduction='sum')
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            # test_loss.append(criterion(output, target).item())
            # get the index of the max log-probability

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc



def saving_and_plotting(time, model_lists, ltr_lists, lt_lists, at_lists, atr_lists, params):
    # DO not save the pretrained VGG part
    runs = len(model_lists)
    isQuantum = params['isQuantum']
    exp_name = f'{time}_Quantum:_{isQuantum}'
    directory = f"Exp:_{exp_name}"
    parent_dir = '/global/u2/r/rr637/QPPAI-FL/Results'
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    path_plots = path + '/Plots'
    os.mkdir(path_plots)
    path_models = path + '/Models'
    os.mkdir(path_models)
    for i in range(runs):
        plot_acc_loss(atr_lists[i], ltr_lists[i], at_lists[i], lt_lists[i], params['isQuantum'],
                      exp_name, f'Accuracy/Loss per Round - Iteration {i + 1}', i + 1)
        torch.save(model_lists[i].net.classifier.state_dict(), f"{path_models}/model_iteration={i}")
    lt_ave = list(np.average(np.array(lt_lists), axis=0))
    at_ave = list(np.average(np.array(at_lists), axis=0))
    ltr_ave = list(np.average(np.array(ltr_lists), axis=0))
    atr_ave = list(np.average(np.array(atr_lists), axis=0))
    if runs > 1:
        plot_acc_loss(atr_ave, ltr_ave, at_ave, lt_ave, params['isQuantum'], exp_name,
                      f'Averaged Accuracy/Loss per Round After {runs} Iterations', 0)

    # Save lists as CSV files
    with open(path + "/avg_loss_test.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(lt_ave)

    with open(path + "/avg_loss_train.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ltr_ave)

    with open(path + "/avg_acc_test.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(at_ave)

    with open(path + "/avg_acc_train.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(atr_ave)

    # Save the dictionary as a CSV file
    with open(path + "/params.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, value in params.items():
            writer.writerow([key, value])

    return


def plot_epsilon_v_acc(epsilon_list, acc_list, exp_index, title):
    plt.scatter(epsilon_list, acc_list),
    plt.xlabel('epsilon'),
    plt.ylabel('final_test_accuracy'),
    plt.title(title),
    plt.savefig(f"Results/Epsilon_vs_Acc/{exp_index}_avg_epsilon_vs_accuracy.png"),
    plt.show()


def plot_acc_loss(tr_a, tr_l, t_a, t_l, is_Quantum, exp_name, title, i):
    plt.plot(tr_a, label="train accuracy"),
    plt.plot(tr_l, label="train loss"),
    plt.plot(t_a, label="test accuracy"),
    plt.plot(t_l, label="test loss"),
    plt.xlabel('Round'),
    plt.ylabel('Accuracy/Loss'),
    plt.title(title),
    plt.legend(),
    plt.savefig(f"Results/Exp:_{exp_name}/Plots/Acc_loss_plot_{i}.png"),
    plt.show()

#Main QFL-DP Training

def main(params):
    if params['isQuantum']:
        global_model = QuantumTransfer().to(device)
        client_models = [QuantumTransfer().to(device)
                         for _ in range(params['selected'])]
    else:
        global_model = ClassicalTransfer().to(device)
        client_models = [ClassicalTransfer().to(device)
                         for _ in range(params['selected'])]

    for model in client_models:
        # initial synchronizing with global model
        model.load_state_dict(global_model.state_dict())

    if params['usePyvacy'] == True:

        opt = [optim.DPSGD(
            l2_norm_clip=params['l2_clip'],
            noise_multiplier=params['noise'],
            minibatch_size=params['mini_bs'],
            microbatch_size=params['micro_bs'],
            params=model.net.classifier.parameters(),
            lr=params['lr'],
            weight_decay=params['l2_penalty'])
            for model in client_models]


    else:
        opt = [optim.SGD(model.net.classifier.parameters(), lr=params['lr']) for model in client_models]

    losses_train = []
    losses_test = []
    acc_train = []
    acc_test = []
    if params['usePyvacy']:
        epsilon = params['epsilon']
        delta = params['delta']
        print(f'Achieves ({epsilon}, {delta})-DP')

    for r in range(params['rounds']):
        # select random clients
        num_selected = params['selected']
        client_idx = np.random.permutation(params['clients'])[:num_selected]
        # client update
        loss = 0
        acc = 0
        for i in range(params['selected']):
            loss_temp, acc_temp = client_update(client_models[i], opt[i],
                                                train_loader[client_idx[i]], epoch=params['epochs']
                                                , index=i, r=r)
            loss += loss_temp
            acc += acc_temp
        # print(f"client_loss_sum:{loss} for round {r}")
        losses_train.append(loss / params['selected'])
        acc_train.append(acc / params['selected'])
        # print(f"loss_train_list_per_round: {losses_train}")
        # print(f"acc_train_list_per_round: {acc_train}")
        trained_client_models = client_models
        trained_global_model = global_model
        server_aggregate(trained_global_model, trained_client_models)
        test_loss, test_acc = test(global_model, test_loader)
        losses_test.append(test_loss)
        acc_test.append(test_acc)
        print(f"loss_test_list_per_round: {losses_test}")
        print(f"acc_test_list_per_round: {acc_test}")
    print(f"FINAL TEST ACC : {acc_test[-1]} After {params['rounds']} Rounds")

    final_dict = {}
    final_dict['lt'] = losses_test
    final_dict['at'] = acc_test
    final_dict['ltr'] = losses_train
    final_dict['atr'] = acc_train
    final_dict['model'] = global_model

    return final_dict


#Script for running experiments

exp_index = datetime.now().strftime("%m_%d_%H_%M_%S")
if not params['eps_vs_acc']:

    lt_lists = []
    at_lists = []
    ltr_lists = []
    atr_lists = []
    model_lists = []
    for i in range(params['runs']):
        results = main(params)
        lt_lists.append(results['lt'])
        at_lists.append(results['at'])
        ltr_lists.append(results['ltr'])
        atr_lists.append(results['atr'])
        model_lists.append(results['model'])
    saving_and_plotting(exp_index, model_lists, ltr_lists, lt_lists, at_lists, atr_lists, params)


else:
    fin_acc_lists = []
    epsilon_lists = []
    noise_min = params['noise_min']
    noise_max = params['noise_max']
    noise_incr = params['noise_incr']

    for i in range(params['runs']):
        epsilon_list = []
        final_accuracy_list = []

        for noise in np.arange(noise_min, noise_max, noise_incr):  # exclusive
            params['noise'] = noise
            dict = main(params)
            final_accuracy_list.append(dict['at'][-1])
            epsilon = analysis.epsilon(dataset_sizes['train'], params['mini_bs'],
                                       noise, params['rounds'], params['delta'])
            epsilon_list.append(epsilon)
        fin_acc_lists.append(final_accuracy_list)
        epsilon_lists.append(epsilon_list)
    fin_acc_avg = list(np.average(np.array(fin_acc_lists), axis=0))
    eps_ave = list(np.average(np.array(epsilon_lists), axis=0))
    print(fin_acc_lists)
    print(fin_acc_avg)
    print(epsilon_lists)
    print(eps_ave)
    runs = params['runs']
    plot_epsilon_v_acc(eps_ave, fin_acc_avg, exp_index, f'Averaged Epsilon vs Acc After {runs} Trials')
    eps_v_acc_path = '/global/u2/r/rr637/QPPAI-FL/Results/Epsilon_vs_Acc'
    with open(eps_v_acc_path + "/params" + ".txt", "wb") as fp:
        pickle.dump(params, fp)












