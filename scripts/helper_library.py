from __future__ import division
import re
import datetime
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import os
#from PIL import Image
import matplotlib.image as mpimg
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import qiskit
import datetime
from qiskit.providers.aer import noise
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
from qiskit import IBMQ, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import numpy as np
import math
from qiskit.providers.aer.noise.errors import standard_errors as SE
from qiskit.providers.aer.noise.device import models
from collections import Counter
import statistics
import queue
from random import randint
#IBMQ.save_account('f0f61055f98741e1e793cc5e0dddbb89567e59362c7ec34687938a3fe50cb765d6749943e8e41ed14fe9798c1663adf7bc0cfa6389f272c54765833936e7c713')
#print("Available backends:")
#provider = IBMQ.get_provider(hub='ibm-q')
#provider = IBMQ.get_provider(hub='ibm-q-ornl')
#provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='csc440')
#print(provider.backends())
from qiskit.qasm import Qasm

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error


#### Helper Functions that can be used across projects ####
def normalize_dict(input_dict):
    '''
    Function to normalize a dictionary 
    '''
    epsilon = 0.0000001
    if sum(input_dict.values()) == 0:
        ##print('Error, dictionary with total zero elements!!')    
        for k,v in input_dict.items():
            input_dict[k] = epsilon
    factor=1.0/sum(input_dict.values())
    #if(factor == 0):
    #    print(factor,sum(input_dict.values())) 
    for k in input_dict:
        input_dict[k] = input_dict[k]*factor

    for k,v in input_dict.items():
        if(v==1):
            input_dict[k] = 1-epsilon

    return input_dict


def update_dist(dict1,dict2):
    '''
    Function to merge two dictionaries in to a third one
    '''
    dict3 = Counter(dict1) + Counter(dict2) 
    dict3 = dict(dict3)
    return dict3

def weighted_update_dist(dict1,dict2,weight):
    '''
    Function to merge two dictionaries in to a third one using a weight factor- useful for weighted EDM
    '''
    _dict2 = dict2.copy()
    for key, value in _dict2.items():
        _dict2[key] = value*weight
    dict3 = Counter(dict1) + Counter(_dict2)
    dict3 = dict(dict3)
    return dict3

def get_number_of_trials(num_qubits):
    '''
    Function to estimate total number of trials to be executed
    '''
    state_space = int(math.pow(2,num_qubits))
    num_trials = int(55.26*state_space)
    return num_trials

def write_qasm_file_from_qobj(output_file,qobj):
    '''
    Function to write a Quantum Object into a given output file QASM
    '''
    f= open(output_file,"w+")
    f.write(qobj.qasm())
    f.close()

def get_counts_given_key(counts,search_key):
    '''
    Function to get the counter value for a given key in a distribution, if not found, return 0
    This function is useful when the errors are too large and the distribution does not contain the given key
    '''
    for k,v in counts.items():
        if (k == search_key):
            return v
    return 0

def find_top_K_keys(distribution,K):
    '''
    Function to create a shortlisted candidates of top K keys
    '''
    sorted_histogram = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    total_shots = sum(distribution.values())
    key_storage = []
    occurence = []
    for i in range(K):
        key_storage.append(sorted_histogram[i][0])
        occurence.append(sorted_histogram[i][1]/total_shots)
        
    return key_storage,occurence
               
def get_complimentary_key(key):
    '''
    Funtion to get the complimentary key (useful for QAOA with two peaks)
    '''
    complimentary_key = ''
    for i in key:
        if(i == '0'):
            complimentary_key = complimentary_key + '1'
        else:
            complimentary_key = complimentary_key + '0'
    return complimentary_key

def convert_key_to_decimal(string, width):
    '''
    Function to convert a key to decimal
    '''
    power = width-1;
    dec_key = 0
    for c in string: # go through every character
        dec_key = dec_key + np.power(2,power)*int(c)
        power = power -1
    return dec_key

def convert_integer_to_bstring(num):
    '''
    Function to convert an integer into bitstring
    '''
    bstring = ''
    flag = 0
    if(num>1):
        bstring = convert_integer_to_bstring(num // 2)
    bstring = bstring+ str(num % 2)
    return bstring

def padding_for_binary(bstring, expected_length):
    '''
    Function to pad a bitstring with 0s and stretch it to a given length
    '''
    curr_length = len(bstring)
    if(expected_length > curr_length):
        diff_length = expected_length - curr_length
        padding = ''
        for i in range(diff_length):
            padding = padding + str(0)
        bstring = padding + bstring
    return bstring

def get_key_from_decimal(num,length):
    '''
    Function to convert a decimal to a key of a given length
    '''
    bstr = convert_integer_to_bstring(num)
    key = padding_for_binary(bstr, length)
    return key

def n_choose_k(n,k): # k choices out of n
    '''
    Function to compute nCk (sometimes we require this to estimate the overheads)
    '''
    n_fact = np.math.factorial(n)
    k_fact = np.math.factorial(k)
    n_minus_k_fact = np.math.factorial(int(n-k))
    n_choose_k_out = int((n_fact/(k_fact*n_minus_k_fact)))
    return n_choose_k_out

#def get_initial_layout_from_qasm(filename):
#    '''
#    Function to read initial layout from a file -> in case program needs to be re-executed, this is helpful
#    Was useful in earlier version of using SABRE
#    '''
#    with open(filename) as f:
#        for line in f:
#            if 'Initial Layout' in line:
#                start_pos = line.find('= ')+3
#                end_pos   = line.find('\n')-1
#                layout = line[start_pos:end_pos]
#                layout = list(layout.split(","))
#                layout = [int(i) for i in layout] 
#    f.close()
#    return layout
def count_gates(program):
    '''
    Function to collect gate type statistics from a program
    '''
    gate_count = np.zeros(6) #cx, x, h, y,swap
    total_gates = 0
    with open(program) as f:
        for line in f:
            if 'cx' in line:
                gate_count[0] +=1
                total_gates +=1
            elif 'x' in line or 'u3' in line:
                gate_count[1] +=1
                total_gates +=1
            elif 'h' in line or 'u2' in line:
                gate_count[2] +=1
                total_gates +=1
            elif 'y' in line:
                gate_count[3] +=1
                total_gates +=1
            elif 'z' in line:
                gate_count[4] +=1
                total_gates +=1
            elif 'swap' in line:
                gate_count[0] +=3
                gate_count[5] +=1
                total_gates +=3
    f.close()
    return gate_count, total_gates

def generate_all_possible_combinations(a):
    '''
    This function returns all possible combinations of a given list
    '''
    if len(a) == 0:
        return [[]]
    cs = []
    for c in generate_all_possible_combinations(a[1:]):
        cs += [c, c+[a[0]]]
    return cs

def xor_c(a, b):
    return '0' if(a == b) else '1';

# Helper function to flip the bit
def flip(c):
    return '1' if(c == '0') else '0';

# function to convert binary string
# to gray string
def binarytoGray(binary_list):
    gray_list =[]
    for binary in binary_list:
        gray = "";

        # MSB of gray code is same as
        # binary code
        gray += binary[0];

        # Compute remaining bits, next bit
        # is comuted by doing XOR of previous
        # and current in Binary
        for i in range(1,len(binary)):

            # Concatenate XOR of previous
            # bit with current bit
            gray += xor_c(binary[i - 1],
                          binary[i]);
        gray_list.append(gray)

    return gray_list;


def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

## useful for JigSaw studies, get partial bitstring from a larger string based on qubits in qlist
def extract_partial_answer(Solution,Qlist):
    Partial_keys = []
    for key in Solution:
        copy_key = ''+key
        copy_key = copy_key[::-1]
        partial_answer = ''
        for j in Qlist:
            partial_answer = partial_answer + copy_key[j]
        Partial_keys.append(partial_answer)
    return Partial_keys

## truncate a floating point number upto certain decimal places
def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

## l1 is a list, l2 is a list of lists; for example: l1 = [0,1,2] and l2 = [[0,1,3],[1,2,4],[0,1,2]]
## check if list l1 is in l2 and the elements are in the same order in both the lists
def listinlist(l1,l2):
    if(len(l2)==0):
       return 0
    for list_entry in l2:
        match = 1 # match found
        for ele in range(len(l1)):
            if(l1[ele]!=list_entry[ele]):
                match = 0
                break
        if(match ==1):
            return match
    return match

def determine_heavy_output_dist(dist):
	'''
	 Function useful for some early QV studies
	'''
	_dist = dist.copy()
	_norm_dist = normalize_dict(_dist)
	_sorted_dist = sorted(_norm_dist.items(), key=lambda x: x[1])	

	list_of_probs = []	
	for entry in _sorted_dist:
		list_of_probs.append(entry[1])
	
	p_median = statistics.median(list_of_probs)

	heavy_output_dist = {}
	for entry in _sorted_dist:
		if(entry[1] >= p_median):
			heavy_output_dist[entry[0]] = entry[1]

	_norm_heavy_output_dict = normalize_dict(heavy_output_dist)
	
	return _norm_heavy_output_dict

## compute hamming distance between two key strings
def compute_hamming_distance(ref_key,alt_key):
	hamming_distance = 0
	for c in range(len(ref_key)):
		if(ref_key[c] !=alt_key[c]):
			hamming_distance = hamming_distance + 1
	#print('Key ', alt_key, ' Distance ', hamming_distance)
	return hamming_distance


## find the bit at a given index in a search string- useful sometimes because Qiskit has reverse ordering of bitstrings 
def find_bit_at_location(key,loc):
	'''
	Function that returns whether the bit at location loc is 0 or 1 in a given key
	'''
	reverse_key = ''
	for idx in range(len(key)-1, -1, -1):
		reverse_key = reverse_key + key[idx]
	return int(reverse_key[loc])	


## find the rank of a given search key
def find_roca(top_keys, search_key):
	for index in range(len(top_keys)):
		if(search_key == top_keys[index]):
			return index+1
	return -1

# compute the entropy of a dictionary/distribution
def compute_entropy(in_dict):
    norm_in = normalize_dict(in_dict.copy())
    norm_in_filtered = {key:val for key, val in norm_in.items() if val != 0.0}
    list_p = norm_in_filtered.values()
    return sum([(-i* math.log2(i)) for i in list_p])


def per_qubit_histograms(dictionary):
    '''
    Function takes a dictionary as input and returns the histograms of each qubit in return 
    Output is a list of entries indexed by the qubit id 
    (for example qubit_prob_0[4] is the probability of qubit 4 in 0 state)
    '''
    num_qubits_in_dict = len(str((list(dictionary.keys())[0])))
    
    qubit_prob_0 = [0 for _ in range(num_qubits_in_dict)]
    qubit_prob_1 = [0 for _ in range(num_qubits_in_dict)]
    
    
    for key,value in dictionary.items():
        reverse_key = key[::-1]
        for i in range(len(reverse_key)):
            if reverse_key[i] == '0':
                qubit_prob_0[i] = qubit_prob_0[i] + value
            else:
                qubit_prob_1[i] = qubit_prob_1[i] + value
    
    ## normalize 
    total_trials = sum(dictionary.values())
    qubit_prob_0 = [i/total_trials for i in qubit_prob_0]
    qubit_prob_1 = [i/total_trials for i in qubit_prob_1]
    
    return qubit_prob_0,qubit_prob_1

def inner_product_fidelity(dist_a,dist_b):
    _in1 = normalize_dict(dist_a.copy())
    _in2 = normalize_dict(dist_b.copy())

    epsilon = 0.00000001
    # update the dictionaries

    for key in _in1.keys():
        if key not in _in2:
            _in2[key] = epsilon # add new entry

    for key in _in2.keys():
        if key not in _in1:
            _in1[key] = epsilon # add new entry

    # both dictionaries should have the same keys by now
    if set(_in1.keys()) != set(_in2.keys()):
        print('Error : dictionaries need to be re-adjusted')

    ## normalize the dictionaries

    _in1 = normalize_dict(_in1)
    _in2 = normalize_dict(_in2)

    #print(_in1)
    #print(_in2)

    list_of_roots = []
    for key,p in _in1.items():
        for _key,q in _in2.items():
            if key == _key:
                s = math.sqrt(p*q)
                list_of_roots.append(s)
                break
    # calculate the sum of squares
    inner_prod_fidelity = sum(list_of_roots)**2
    return inner_prod_fidelity

def root_mean_square_error(dist_a,dist_b):
    _in1 = normalize_dict(dist_a.copy())
    _in2 = normalize_dict(dist_b.copy())

    epsilon = 0.00000001
    # update the dictionaries

    for key in _in1.keys():
        if key not in _in2:
            _in2[key] = epsilon # add new entry

    for key in _in2.keys():
        if key not in _in1:
            _in1[key] = epsilon # add new entry

    # both dictionaries should have the same keys by now
    if set(_in1.keys()) != set(_in2.keys()):
        print('Error : dictionaries need to be re-adjusted')

    ## normalize the dictionaries

    _in1 = normalize_dict(_in1)
    _in2 = normalize_dict(_in2)

    #print(_in1)
    #print(_in2)


    list_of_squares = []
    for key,p in _in1.items():
        for _key,q in _in2.items():
            if key == _key:
                s = (p-q)**2
                list_of_squares.append(s)
                break
    # calculate the sum of squares
    root_mean_square_error = math.sqrt(sum(list_of_squares))/len(list_of_squares)
    return root_mean_square_error

## moved this to data logging functions for now
##def write_data_dictionary_into_logfile(data_dictionary,outputfname):
##    f = open(outputfname,"+w")
##    f.write('data_dictionary = '+str(data_dictionary)+'\n')
##    f.close()
##def read_data_dictionary_from_logfile(logname):
##    with open(logname) as f:
##        for line in f:
##            start_pos = line.find('= ') + 2
##            end_pos   = line.find('\n')
##            data_from_log = ast.literal_eval(line[start_pos:end_pos])
##    f.close()
##    return data_from_log

def convert_binary_string(list_int):
    
    output = []
    bitwidth = int(math.log(len(list_int),2))
    bitwidth = str("{0:0"+ str(bitwidth) + "b}")
    for ele in list_int:    
        output.append(bitwidth.format(ele))
    
    return output

def find_binary_substring(input_dict, substring, location_list):
    
    filtered_dict = dict(filter(lambda item: substring in item[0], input_dict.items())) 
    output_dict = {}
    for key in filtered_dict:
        temp_string = ""
        for ele in location_list:
            temp_string += key[len(key)-ele-1]
        if temp_string == substring:
            output_dict.update({key:filtered_dict[key]})
    
    return output_dict

def Create_Marginals(orignal_count, marginal_order):
    
    norm_orignal_dict = normalize_dict(orignal_count)
    list_binary_string = convert_binary_string([*range(2**len(marginal_order))])
    output ={} 
    
    if sum(marginal_order) == 0.5*len(marginal_order)*(2*min(marginal_order)+(len(marginal_order)-1)):
        #print("Serial Marginal")
        for key in list_binary_string:
            matching_dict = find_binary_substring(norm_orignal_dict,key,marginal_order) 
            val = sum(matching_dict.values())
            output.update({key:val})
    else:
        #print("Distrubuted Marginal")
        for key in list_binary_string:
            matching_dict = find_distrubuted_binary_substring(norm_orignal_dict,key,marginal_order)
            val = sum(matching_dict.values())
            output.update({key:val})
    output = normalize_dict(output)    
    return output
