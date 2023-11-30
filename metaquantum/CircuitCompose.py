# 2019 12 04
# Modify for the MetaQuantum.ML package

# 2019 11 15
# Meta Definition of VQC
# Possibly for the future meta-QML
# Future: Use computation graph to define the circuit

import os
import argparse

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

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline

# from keras.datasets import mnist 

import pickle


dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MNIST


# Should be able to adapt to arbitrary vqc_definition
# Sequential
# angle preprocessing here but not in the single VQC components
# Need to move out this component into the VQC components
# since in the future there may be many different kinds of preprocessing.
class Stacked_VQC:
	def __init__(self, classical_scaling = False, encoding_scheme = 0, variational_encoding_angle = "atan", vqc_sequence = []):
		# encoding_scheme = 0 ==> amplitude encoding
		# encoding_scheme = 1 ==> variational encoding
		# get_angles function currently fixed
		# should be arbitraty in the future

		self.encoding_scheme = encoding_scheme
		self.vqc_sequence = vqc_sequence
		self.classical_scaling = classical_scaling
		self.variational_encoding_angle = variational_encoding_angle
		self.var_Q_array = []

	def get_angles(self, in_x):
		return torch.stack([torch.stack([torch.asin(item), torch.acos(item**2)]) for item in in_x])

	def get_angles_atan(self, in_x):
		return torch.stack([torch.stack([torch.atan(item), torch.atan(item**2)]) for item in in_x])

	def forward(self, single_item):
		if self.classical_scaling == True:
			if self.encoding_scheme == 0:
				res_temp = self.vqc_sequence[0].forward(single_item)

				for i in range(1,len(self.vqc_sequence)):
					res_temp = self.get_angles(res_temp)
					res_temp = self.vqc_sequence[i].forward(res_temp)

				return res_temp

			# if self.encoding_scheme == 0:
			# 	res_temp = self.vqc_sequence[0].forward(single_item)
			# 	# print(res_temp)
			# 	res_temp = self.vqc_sequence[1].forward(res_temp)
			# 	# print(res_temp)

			# 	for i in range(2,len(self.vqc_sequence)):
			# 		res_temp = self.get_angles(res_temp)
			# 		res_temp = self.vqc_sequence[i].forward(res_temp)

			# 	return res_temp

			elif self.encoding_scheme == 1:
				# classical scaling currently no effect in the variational encoding
				# just keep the original code
				res_temp = self.get_angles(single_item)
				res_temp = self.vqc_sequence[0].forward(res_temp)

				for i in range(1,len(self.vqc_sequence)):
					res_temp = self.get_angles(res_temp)
					res_temp = self.vqc_sequence[i].forward(res_temp)

				return res_temp

		else:
			if self.encoding_scheme == 0:
				res_temp = self.vqc_sequence[0].forward(single_item)
				#print(f"Res Temp after 0th VQC Forward: {res_temp}")
				for i in range(1,len(self.vqc_sequence)):
					res_temp = self.get_angles(res_temp)
					#print(f"Res Temp after {i}th Get Angles: {res_temp}")
					res_temp = self.vqc_sequence[i].forward(res_temp)
					#print(f"Res Temp after {i}th VQC Forward: {res_temp}")

				return res_temp

			elif self.encoding_scheme == 1:
				res_temp = None
				
				if self.variational_encoding_angle == 'atan':
					res_temp = self.get_angles_atan(single_item) # Successful in TJET
				else:
					res_temp = self.get_angles(single_item)  # Successful in MNIST binary
				#print(f"Res Temp after 0th Get Angles: {res_temp}")
				res_temp = self.vqc_sequence[0].forward(res_temp)
				#print(f"Res Temp after VQC sequence 0 forward: {res_temp}")
				for i in range(1,len(self.vqc_sequence)):
					res_temp = self.get_angles(res_temp)
					#print(f"Res Temp after {i}th Get Angles: {res_temp}")
					res_temp = self.vqc_sequence[i].forward(res_temp)
					#print(f"Res Temp after VQC sequence {i} forward: {res_temp}")
				return res_temp

# Analogous to Sequential in Keras
# The input is list of VQC objects
# 2020 01 29 Need to implement load_model() function
class Sequential(Stacked_VQC):
	# Need to resolve the problem of "Final VQC"
	def add(self, single_vqc):
		var_Q_circuit = single_vqc.init_params()
		self.vqc_sequence.append(single_vqc)
		self.var_Q_array.append(var_Q_circuit)

	def load_model(self, circuit_parameters):
		return


	

# Conv Layer
# Parallel located a series of filters
# Filter size = M * M
# Stride      = N
# Num of Filters = F
# This should send into Sequential Class
# keep the gradient function
# A sequence of ConvFilter

class Conv2D:
	def __init__(self, encoding_scheme = 1, num_of_filters = 6, kernel_size = (2,2), num_layers = 4, stride = 1, input_shape = None):
		# encoding_scheme = 0 ==> amplitude encoding
		# encoding_scheme = 1 ==> variational encoding
		# get_angles function currently fixed
		# should be arbitraty in the future


		self.encoding_scheme = encoding_scheme
		self.num_of_filters = num_of_filters
		self.kernel_size = kernel_size
		self.num_layers
		self.stride = stride
		self.input_shape = input_shape
		self.list_of_ConvFilters = None

	def generate_conv_filters(self):
		for i in range(self.num_of_filters):
			num_qubits = self.kernel_size[0] * self.kernel_size[1]
			num_layers = self.num_layers
			var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device=device).type(dtype), requires_grad=True)
			convFilter = ConvFilter(kernel_size = self.kernel_size, stride = self.stride)
			convFilter.set_params(var_Q_circuit)
			self.list_of_ConvFilters.append(convFilter)

	def get_angles(self, in_x):
		return torch.stack([torch.stack([torch.asin(item), torch.acos(item**2)]) for item in in_x])

	def forward(self, single_item):
		# For each single_item, go through all ConvFilters


		######
		if self.encoding_scheme == 0:
			res_temp = self.vqc_sequence[0].forward(single_item)

			for i in range(1,len(self.vqc_sequence)):
				res_temp = self.get_angles(res_temp)
				res_temp = self.vqc_sequence[i].forward(res_temp)

			return res_temp

		elif self.encoding_scheme == 1:
			res_temp = self.get_angles(single_item)
			res_temp = self.vqc_sequence[0].forward(res_temp)

			for i in range(1,len(self.vqc_sequence)):
				res_temp = self.get_angles(res_temp)
				res_temp = self.vqc_sequence[i].forward(res_temp)

			return res_temp


# This is the single conv filter
# The input for this layer is two-dimensional array
class ConvFilter:
	def __init__(self, kernel_size = (2,2), stride = 1):
		pass

	def set_params(self):
		return

	def forward(self, single_item):
		return



# Pooling
# return a pooled tensor
# keep the gradient function
def Pooling():

	return



def vqc_sequence_parser(block_idx, vqc_class_definition, vqc_cell_definition):
	'''
	Parsing the vqc_block_definition and then return the vqc block

	circuit_block_definition:
		0 ==> VariationalQuantumClassifierBinaryLoadingBlock_10_IN_4_OUT
		1 ==> VariationalQuantumClassifierBinaryInterBlock_10_IN_4_OUT
		2 ==> VariationalQuantumClassifierBinaryLoadingBlock_10_IN_10_OUT
		3 ==> VariationalQuantumClassifierBinaryInterBlock_10_IN_10_OUT
		4 ==> VariationalQuantumClassifierInterBlock_4_IN_4_OUT
		5 ==> VariationalQuantumClassifierInterBlock_4_IN_2_OUT

	Modify for the M_IN N_OUT situations

	'''

	# block_class = vqc_cell_definition["circuit_block"]
	num_layers = vqc_cell_definition["num_layers"]
	num_qubits = vqc_cell_definition["num_qubits"]
	num_of_input = vqc_cell_definition["num_of_input"]
	num_of_output = vqc_cell_definition["num_of_output"]
	classical_scaling = vqc_cell_definition["classical_scaling"]
	loading_block = vqc_cell_definition["loading_block"]
	final_block = vqc_cell_definition["final_block"]

	if classical_scaling == True:
		# var_Q_circuit = Variable(torch.tensor(np.ones(num_qubits), device=device).type(dtype), requires_grad=True)
		var_C_scaling = Variable(torch.ones(1024, dtype = torch.double), requires_grad=True)
		var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device=device).type(dtype), requires_grad=True)

		block_class = 0

		VQC = vqc_class_definition[block_class](num_of_input = num_of_input, num_of_output = num_of_output, num_of_wires = num_qubits, var_Q_circuit = var_Q_circuit, var_Q_bias = None, var_C_scaling = var_C_scaling)
		
		return VQC, var_Q_circuit, var_C_scaling

	else:
		var_Q_circuit = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device=device).type(dtype), requires_grad=True)
		
		block_class = 2
		if loading_block == True:
			block_class = 1
		if final_block == True:
			block_class = 3

		VQC = vqc_class_definition[block_class](num_of_input = num_of_input, num_of_output = num_of_output, num_of_wires = num_qubits, var_Q_circuit = var_Q_circuit, var_Q_bias = None)
		
		return VQC, var_Q_circuit

def Generate_VQC(vqc_class_definition, vqc_definition):
	'''
	Parsing the vqc_definition and then return the stacked VQC for training and inference
	'''

	encoding_scheme = vqc_definition["encoding_scheme"]
	vqc_sequence = vqc_definition["vqc_sequence"]
	classical_scaling = vqc_sequence[0]["classical_scaling"]
	vqc_instance_list = []
	vqc_param_list = []

	if classical_scaling == True:
		VQC, var_Q_circuit, var_C_scaling = vqc_sequence_parser(0, vqc_class_definition, vqc_sequence[0])
		vqc_instance_list.append(VQC)
		vqc_param_list.append(var_Q_circuit)
		vqc_param_list.append(var_C_scaling)

		for idx, item in enumerate(vqc_sequence[1:]):
			VQC, var_Q_circuit = vqc_sequence_parser(idx, vqc_class_definition, item)
			vqc_instance_list.append(VQC)
			vqc_param_list.append(var_Q_circuit)

	else:
		for idx, item in enumerate(vqc_sequence):
			VQC, var_Q_circuit = vqc_sequence_parser(idx, vqc_class_definition, item)
			vqc_instance_list.append(VQC)
			vqc_param_list.append(var_Q_circuit)

	stackedVQC = Stacked_VQC(classical_scaling = classical_scaling, encoding_scheme = encoding_scheme, vqc_sequence = vqc_instance_list)


	return stackedVQC, vqc_param_list
