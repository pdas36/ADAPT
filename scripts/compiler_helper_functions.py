import os
import ast
import numpy as np
from numpy import pi
# importing Qiskit
from qiskit import QuantumCircuit, execute, Aer, IBMQ
## packages required for qaoa
from networkx.generators.random_graphs import erdos_renyi_graph
from qiskit import IBMQ
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
from qiskit.qasm import Qasm
import matplotlib.pyplot as plt 
import matplotlib.axes as axes
import numpy as np
import networkx as nx
import os.path
from os import path
import pickle 
from collections import Counter

from qiskit.providers.aer.noise import NoiseModel

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
import sys

### Update the provider in some of the functions below (marked under FIXME)


def read_qasm(filepath, verbo=0):
    ''' 
    Function to read a QASM into a Quantum circuit object
    '''
    circ = QuantumCircuit.from_qasm_file(filepath)
    if(verbo):
        circ.draw()
        print(circ)
        list_ops = circ.size()
        print("Total Number of Operations", list_ops)
        print('Circuit Depth: ', circ.depth())
        print('Number of Qubits in program:', circ.width())
    return circ

def write_qasm_file_from_qobj(output_file,qobj):
    ''' 
    Function to write a Quantum Object into a given output file QASM
    '''
    f= open(output_file,"w+")
    f.write(qobj.qasm())
    f.close()
def get_device_information(machine_name):
    ''' 
    Function to obtain the device name, coupling map, and basis gates of a machine 
    '''
    ### FIXME: Insert provider name here ###
    device = provider.get_backend(machine_name)
    noise_model = NoiseModel.from_backend(device)
    coupling_map = device.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    
    return device,coupling_map,basis_gates
## function to return the number of qubits in a program
def num_qubits_from_qasm(qasmfile):
    qobj = read_qasm(qasmfile)
    total = qobj.width()
    ops = qobj.count_ops()
    cregs = ops['measure']
    return total-cregs


def recursive_compile_noise_adaptive(quantum_circs,machine_name,opt_level=3,recur_count=25,basis_gates=None):
    '''
    Function to recursively compile a list of quantum circuits
    Input: quantum_circs -> list of quantum circuits 
           machine_name  -> name of the machine on which to run the quantum circuits
           opt_level     -> Optimization level for compilation from qiskit
           recur_count   -> terminate after these many rounds and select the circuit with min cnots
           basis_gates   -> basis gates supported on the device to be used for compilation
    '''
    ### FIXME: Insert provider name here ###
    device = provider.get_backend(machine_name)
    noise_model = NoiseModel.from_backend(device)
    coupling_map = device.configuration().coupling_map
    if basis_gates is None:
        basis_gates = noise_model.basis_gates
    if type(quantum_circs) is not list:
        quantum_circs = [quantum_circs]
    post_compile_circs,post_compile_cx_counts,post_compile_op_counts = [],[],[]
    total_compiled = 0
    for quantum_circ in quantum_circs:
        qobjs, all_ops = [], []
        cx_cts = np.zeros(recur_count)
        for i in range(recur_count):
            seed = np.random.randint(100) ## random seed 
            circ_out=transpile(quantum_circ, backend=device, seed_transpiler=seed,basis_gates=basis_gates, coupling_map=coupling_map,layout_method='noise_adaptive',routing_method='sabre',optimization_level=opt_level);
            qobjs.append(circ_out)
            ops = circ_out.count_ops()
            all_ops.append(ops)
            if 'cx' in ops:
                cx_cts[i] = ops['cx']     
        ## pick the circ with the lowest cx count
        ind = np.argmin(cx_cts)
        circ_out = qobjs[ind]
        cx_counts = cx_cts[ind]
        op_counts = all_ops[ind]
        post_compile_circs.append(circ_out)
        post_compile_cx_counts.append(cx_counts)
        post_compile_op_counts.append(op_counts)
        if len(quantum_circs)>1000:
           total_compiled = total_compiled + 1
           if total_compiled%50==0:
               print('Total compiled ', total_compiled, ' Remaining ', len(quantum_circs)-total_compiled)

    if len(quantum_circs)!=0:
        return post_compile_circs,post_compile_cx_counts,post_compile_op_counts
    else:  
        return circ_out,cx_counts,op_counts

def update_dist(dict1,dict2):
    ''' 
    Function to merge two dictionaries in to a third one
    '''
    dict3 = Counter(dict1) + Counter(dict2) 
    dict3 = dict(dict3)
    return dict3

## execute a list of qobjs (post compiled) on a given machine by batching them
def execute_on_real_machine(compiled_qobjs, shots=8192, machine_name='ibmq_paris',max_acceptable_circuits_by_device=900, repeats=1,execution_mode='Default'):
    ### FIXME: Insert provider name here ###
    if execution_mode == 'dedicated':
        provider = IBMQ.get_provider(hub='ibm-q-research', group='gatech-2', project='main')
        print('Running Experiments in Dedicated Mode ')
    device = provider.get_backend(machine_name)
    ## replicate the qobjs based on shots 
    post_compile_qobjs = []
    for i in range(len(compiled_qobjs)):
        for _ in range(repeats): 
            post_compile_qobjs.append(compiled_qobjs[i]) ## simply replicate the qobjs as many times as required

    #determine the batch size 
    _batch_size = min(max_acceptable_circuits_by_device,len(post_compile_qobjs))
    batch_size = []
    if(int(len(post_compile_qobjs)%_batch_size)!= 0):
        num_batches = int(len(post_compile_qobjs)/_batch_size)+1
        for x in range(num_batches-1):
            batch_size.append(_batch_size)
        batch_size.append(int(len(post_compile_qobjs)%_batch_size))
    else:
        num_batches = int(len(post_compile_qobjs)/_batch_size)
        for x in range(num_batches):
            batch_size.append(_batch_size)
    
    print('Number of compiled qobjs ', len(compiled_qobjs))
    print('Total number of circuits to be executed', len(post_compile_qobjs))
    print('Executed in batches of ', batch_size)
    
    _machine_counts = [] 
    job_ids = []
    job_noise_properties = []
    job_time = []
    job_results = []
    
    machine_shot_counts = shots
    print('Running a total of ', len(post_compile_qobjs), ' quantum objects in ', num_batches, 'batches')
    for batch_id in range(num_batches):
        quantum_objects = []
        for qc in range(batch_size[batch_id]):
            quantum_objects.append(post_compile_qobjs[int((batch_id*batch_size[batch_id])+qc)])
       	experiments = qiskit.assemble(quantum_objects, backend=device, shots=machine_shot_counts)
        print('Status of Circuit Batch :',batch_id)
        ibmq_job = device.run(experiments)
        job_monitor(ibmq_job)
        machine_results = ibmq_job.result()
        for qc in range(batch_size[batch_id]):
            _machine_counts.append(machine_results.get_counts(qc))
        
        ## log down the metadata
        job_results.append(machine_results)
        job_noise_properties.append(ibmq_job.properties())
        job_ids.append(ibmq_job.job_id())
        job_time.append(ibmq_job.time_per_step())

    ## need to re-organize the machine counts now
    machine_counts = []
    for i in range(len(compiled_qobjs)):
        merged_dictionary = {}
        for j in range(repeats):
            required_dictionary_index = i*repeats + j ## each qobj is repeated "repeats" number of times, so index of the dictionary is computed for entry into global list 
            merged_dictionary = update_dist(merged_dictionary, _machine_counts[required_dictionary_index]) 
        machine_counts.append(merged_dictionary)
        
    return machine_counts, job_ids, job_noise_properties, job_time, job_results

## execute a list of qobjs on ideal sim
def execute_on_ideal_machine(post_compile_qobjs, shots):
    '''
    Function to run a simulation on an ideal simulator
    '''
    ideal_counts_vector = []
    backend_ideal = Aer.get_backend('qasm_simulator')
        
    ideal_results = qiskit.execute(post_compile_qobjs, backend=backend_ideal, shots=shots).result()
    ## create a list of dictionaries from the results 
    ideal_counts_vector = [ideal_results.get_counts(k) for k in range(len(post_compile_qobjs))]
    return ideal_counts_vector
