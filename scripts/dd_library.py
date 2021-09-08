import numpy as np
from numpy import pi
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
#import qtip

import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, IBMQ, Aer, execute
from qiskit.tools.monitor import job_monitor
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag, circuit_to_instruction
from qiskit.visualization import dag_drawer
from qiskit.quantum_info import Operator

import copy

################################################################################################################################################
def get_transpiled_circuits(qc, backend, seed = 11):
    """
    Input: Untranspiled quantum circuit qc, transpiler seed, backend
    Output: 4 transpiled quantum circuits at various levels of transpilation
    """
    optimized_0 = transpile(qc, backend=backend, seed_transpiler=seed, optimization_level=0)
    optimized_1 = transpile(qc, backend=backend, seed_transpiler=seed, optimization_level=1)
    optimized_2 = transpile(qc, backend=backend, seed_transpiler=seed, optimization_level=2)
    optimized_3 = transpile(qc, backend=backend, seed_transpiler=seed, optimization_level=3)
    
    return optimized_0, optimized_1, optimized_2, optimized_3

################################################################################################################################################


def extract_qubit_index(qubit_obj):
    
    """Input: A qubit object
       Output: The index corresponding to the given qubit in the circuit 
    """
    qubit_obj_str = str(qubit_obj)
    return qubit_obj_str.split('), ')[1].split(')')[0]

################################################################################################################################################
    
def extract_cbit_index(cbit_obj):
    
    """Input: A classical bit object
       Output: The index corresponding to the given classical bit in the circuit
    """
    cbit_obj_str = str(cbit_obj)
    return cbit_obj_str.split('), ')[1].split(')')[0]

################################################################################################################################################
    
def get_qubit_set(dag):
    
    """Input: A dag
       Ouput: A set of all qubit indices which are used in the circuit corresponding to the dag 
    """
    
    #declaring an empty qubit set
    qubit_set = set()
    
    #extracting all the qubit indices from the different opeartion nodes in the dag
    for ele in dag.nodes(): 
        if ele.type =='op':
            node=ele
            
            #if the node is of type 'cx' then add the indices of the two qubits it acts on in the set of all
            #qubit indices
            if node.name == 'cx': 
                #print(node.name, extract_qubit_index(node.qargs[0]),extract_qubit_index(node.qargs[1]))
                q1=extract_qubit_index(node.qargs[0])
                q2=extract_qubit_index(node.qargs[1])
                qubit_set.add(q1)
                qubit_set.add(q2)
            
            #if the node is of type barrier then add all the qubit indices on which the barrier acts
            if node.name == 'barrier':
                for ele in node._qargs:
                    qubit_set.add(extract_qubit_index(ele))
            
            #all other nodes in a transpiled circuit except cx and barrier are single qubit nodes and have
            #only one qubit argument
            else:
                #print(node.name, extract_qubit_index(node.qargs[0]))
                q1=extract_qubit_index(node.qargs[0])
                qubit_set.add(q1)

    return qubit_set

################################################################################################################################################

def create_empty_InstructionTable(qubit_set, circuit_depth):
    """Input: Set of qubits in a transpiled DAG, Depth of the transapiled circuit
       Output: A Data frame  
    """
    
    IDT_dict ={}
    for ele in qubit_set:
        
        #To every qubit, assigning a list of zeros which has length equal to the circuit depth
        #This is to keep track of the operations acting on the qubit as the circuit progresses
        IDT_dict.update({str(ele):circuit_depth*[0]})
    
    #Creating a data frame out of the dictionary
    IDT_Frame = pd.DataFrame(IDT_dict)
    
    return IDT_Frame

################################################################################################################################################

def populate_InstructionTable(qubit_set,dag,empty_IDT, mode):
    """Input: Set of all qubits in the DAG, The DAG, The data table skeleton, mode parameter
       Output:  
    """
    
    IDT_Frame = copy.deepcopy(empty_IDT)
    
    #dictionary of all indices in the DAG to time step 0
    index_list =  {}
    for ele in qubit_set:
        index_list.update({ele:0}) 
    
    #iterating over all nodes in the dag
    for ele in dag.nodes(): 
        
        #iterating over the operator type of nodes only
        if ele.type =='op':
            node=ele
            
            if node.name == 'cx': 
                #print(node.name, extract_qubit_index(node.qargs[0]),extract_qubit_index(node.qargs[1]))
                q1=extract_qubit_index(node._qargs[0])
                q2=extract_qubit_index(node._qargs[1])
                
                # use max of the two index
                cur_index = max([index_list[q1],index_list[q2]])
                if mode =="Visual":
                    IDT_Frame[q1][cur_index] = node.name
                    IDT_Frame[q2][cur_index] = node.name 
                else:
                    IDT_Frame[q1][cur_index] = node.name + ' ' +str(q1)+ ' ' + str(q2)
                    IDT_Frame[q2][cur_index] = node.name + ' ' +str(q1)+ ' ' + str(q2)
                    
                index_list[q1] = cur_index+1
                index_list[q2] = cur_index+1

            if node.name == 'barrier':
                
                qargs_list = []
                _index_list =[]
                
                for ele in node._qargs:
                    qargs_list.append(extract_qubit_index(ele))
                    _index_list.append(index_list[extract_qubit_index(ele)])

                cur_index=max(_index_list)
                for ele1 in qargs_list:
                    IDT_Frame[ele1][cur_index] = node.name
                    index_list[ele1] = cur_index + 1

            if node.name == 'rz':
                if mode =="Visual":
                    q1=extract_qubit_index(node._qargs[0])
                    IDT_Frame[q1][index_list[q1]] = node.name
                    index_list[q1] = index_list[q1]+1
                else:
                    #print(node.name,tuple(node.op.params), extract_qubit_index(node.qargs[0]))
                    q1=extract_qubit_index(node._qargs[0])
                    rz_param = tuple(node.op.params)[0]
                    IDT_Frame[q1][index_list[q1]] = node.name + ' (' + str(rz_param) + ')'
                    index_list[q1] = index_list[q1]+1

            if node.name == 'x':
                if mode =="Visual":
                    q1=extract_qubit_index(node._qargs[0])
                    IDT_Frame[q1][index_list[q1]] = node.name
                    index_list[q1] = index_list[q1]+1
                
                else:
                    #print(node.name,tuple(node.op.params), extract_qubit_index(node.qargs[0]))
                    q1=extract_qubit_index(node._qargs[0])
                    IDT_Frame[q1][index_list[q1]] = node.name
                    index_list[q1] = index_list[q1]+1

            if node.name == 'id':
                if mode =="Visual":
                    q1=extract_qubit_index(node._qargs[0])
                    IDT_Frame[q1][index_list[q1]] = node.name
                    index_list[q1] = index_list[q1]+1
                
                else:
                    #print(node.name,tuple(node.op.params), extract_qubit_index(node.qargs[0]))
                    q1=extract_qubit_index(node._qargs[0])
                    IDT_Frame[q1][index_list[q1]] = node.name
                    index_list[q1] = index_list[q1]+1

            if node.name == 'sx':
                if mode =="Visual":
                    q1=extract_qubit_index(node._qargs[0])
                    IDT_Frame[q1][index_list[q1]] = node.name
                    index_list[q1] = index_list[q1]+1
                else:
                    #print(node.name,tuple(node.op.params), extract_qubit_index(node.qargs[0]))
                    q1=extract_qubit_index(node._qargs[0])
                    IDT_Frame[q1][index_list[q1]] = node.name
                    index_list[q1] = index_list[q1]+1
            
            if node.name == 'measure':
                if mode == "Visual":
                    q1 = extract_qubit_index(node._qargs[0])
                    c1 = extract_cbit_index(node.cargs[0])
                    IDT_Frame[q1][index_list[q1]] = node.name
                    index_list[q1] = index_list[q1] + 1
                else:
                    q1 = extract_qubit_index(node._qargs[0])
                    c1 = extract_cbit_index(node.cargs[0])
                    IDT_Frame[q1][index_list[q1]] = node.name + ' ' + c1
                    index_list[q1] = index_list[q1] + 1

    
    return IDT_Frame

################################################################################################################################################

def zero_filter(df):
    """
    """
    
    df = df[(df.T != 0).any()]
    for ele in df.columns:
        if 0 in set(df[ele]) and 'barrier' in set(df[ele]) and len(set(df[ele]))==2:
            del df[ele]

    return df   

################################################################################################################################################

def get_X_length(device_str, provider):
	
    list_output = []
    device = provider.get_backend(device_str)
    properties = device.properties()
    
    for i in range(len(properties.gates)):
        if len(properties.gates[i].qubits) == 1:
            if properties.gates[i].name == ('x'+ str(properties.gates[i].qubits[0])):
                list_output.append(properties.gates[i].parameters[1].to_dict()['value'])
            
    return list_output

################################################################################################################################################

def get_SX_length(device_str, provider):

    list_output = []
    device = provider.get_backend(device_str)
    properties = device.properties()
    
    for i in range(len(properties.gates)):
        if len(properties.gates[i].qubits) == 1:
            if properties.gates[i].name == ('sx'+ str(properties.gates[i].qubits[0])):
                list_output.append(properties.gates[i].parameters[1].to_dict()['value'])
            
    return list_output

################################################################################################################################################

def get_RZ_length(device_str, provider):
	
	
    list_output = []
    device = provider.get_backend(device_str)
    properties = device.properties()
    
    for i in range(len(properties.gates)):
        if len(properties.gates[i].qubits) == 1:
            if properties.gates[i].name == ('rz'+ str(properties.gates[i].qubits[0])):
                list_output.append(properties.gates[i].parameters[1].to_dict()['value'])
            
    return list_output

################################################################################################################################################

def get_ID_length(device_str, provider):
	
	
    list_output = []
    device = provider.get_backend(device_str)
    properties = device.properties()
    
    for i in range(len(properties.gates)):
        if len(properties.gates[i].qubits) == 1:
            if properties.gates[i].name == ('id'+ str(properties.gates[i].qubits[0])):
                list_output.append(properties.gates[i].parameters[1].to_dict()['value'])
            
    return list_output

################################################################################################################################################

def get_CNOT_length(device_str, provider):
    
    """Input: Device name, Provider name
       Output: Dictionary containing the execution time of the 
       CNOT gates present between two given qubits
    """
    device = provider.get_backend(device_str)
    
    #number of qubits in the device
    num_qubits = len(device.properties().to_dict()['qubits'])
    
    
    properties = device.properties()
    dict_CNOT_length = {}
    for i in range(len(properties.gates)):
        if len(properties.gates[i].qubits) > 1:
            dict_CNOT_length.update({properties.gates[i].name:properties.gates[i].parameters[1].to_dict()['value']})
    
    #converting the dictionary to a 2d array
    cnot_lengths = np.zeros((num_qubits, num_qubits))
    dict_keys = list(dict_CNOT_length.keys())
    
    for key in dict_keys:
        indices = key[2:].split('_')
        idx0 = int(indices[0])
        idx1 = int(indices[1])
        cnot_lengths[idx0][idx1] = dict_CNOT_length[key]
    
    return cnot_lengths


## get all the information at once
def get_all_instruction_lengths(machine,provider):
    cx_lengths = get_CNOT_length(machine, provider)
    x_lengths = get_X_length(machine, provider)
    sx_lengths = get_SX_length(machine, provider)
    id_lengths = get_ID_length(machine, provider)
    rz_lengths = get_RZ_length(machine,provider)
    
    return cx_lengths,x_lengths,sx_lengths,id_lengths,rz_lengths

#################################################################################################################################################

#convert a discrete IDT to an analog IDT
def adv_discrete_to_analog(IDT, gate_lengths, mode = 'NotVisual'):
    
    """
    Input: An IDT where the gates acting on qubits are shown in a discrete fashion
    Output: An IDT table where each row is the time value where all the operations in that particular 
    timestep end
    """
    #getting the various gate lengths:
    cx_lengths = gate_lengths['cx'] #list of lists
    sx_lengths = gate_lengths['sx']
    rz_lengths = gate_lengths['rz']
    x_lengths = gate_lengths['x']
    id_lengths = gate_lengths['id']
    barrier_length = 0
    meas_length = 0
    
    #
    IDT_shape = IDT.shape
    IDT_new = copy.deepcopy(IDT)
    
    #initiating the indices to t = 0
    new_indices = [0]
    
    #number of time steps
    n_rows = IDT_shape[0]
    
    #number of qubits
    n_cols = IDT_shape[1]
    
    #converting discrete data frame to analog
    qubits_in_table = list(IDT.columns)
    
    #for all operations except the final measurement operation
    for ts in range(n_rows):
        
        #an array that stores the gate lengths of all the gates in the current timestep
        gate_lengths = []
        
        for qubit in qubits_in_table:
            
            #obtaining the gate name depending on the type of the table
            if mode == 'Visual':
                gate = IDT.loc[ts, qubit]
            elif IDT.loc[ts, qubit] != 0:
                gate = IDT.loc[ts, qubit].split(' ')[0]
            else:
                gate = 0
            
            if gate == 0 or gate == 'barrier' or gate == 'measure':
                gate_lengths.append(0)
            
            elif gate == 'sx':
                gate_lengths.append(sx_lengths[int(qubit)])
            
            elif gate == 'x':
                gate_lengths.append(x_lengths[int(qubit)])
                
            elif gate == 'id':
                gate_lengths.append(id_lengths[int(qubit)])
            
            elif gate == 'rz':
                gate_lengths.append(rz_lengths[int(qubit)])
            
            elif gate == 'cx':
                #the qubit on which the controlled gate is being applied
                sec_qubit = int(IDT.loc[ts, qubit].split(' ')[2])
                gate_lengths.append(cx_lengths[int(qubit)][sec_qubit])
        
        #obtainting the max gate length at a particular times index
        max_gate_length_ts = max(gate_lengths)
        new_indices.append(new_indices[-1] + max_gate_length_ts)
    
    #removing the t = 0 index
    new_indices = new_indices[1:]
    
    #reindexing the IDT frame
    IDT_new.index = new_indices
    
    return IDT_new

################################################################################################################################################
#converts a string to a gate
def gate2matrix(gate_string):
    '''
    Input gate string:  u1(pi), h, t, s etc.
    Output: gate matrix
    '''
    circ = QuantumCircuit(1)
    eval("circ."+ gate_string +"(0)")
    Gate_Matrix=Operator(circ).data
    
    return Gate_Matrix

################################################################################################################################################

#returns the operator norm of the difference of two unitary matrices U and V
def operator_norm(U, V):
    """
    Input: Unitary matrices U, V 
    Output: The operator norm of M which is M = U-V. (which is the square root of the largest eigenvalue of MtM)
    """
    M = U-V
    MdM = np.matmul(M.conj().T, M)
    e_vals , e_vecs = np.linalg.eig(MdM)
    max_e_val = max(list(e_vals))
    
    return np.sqrt(max_e_val)

################################################################################################################################################

#finds 'closest' clifford gate
def closest_clifford(gate):
    """
    Input: A qiskit gate
    Output: The closest single qubit clifford gate out of [X, Z, S, Sdg]
    """
    X=gate2matrix('x')
    Z = gate2matrix('z')
    S=gate2matrix('s')
    Sdg = gate2matrix('sdg')
    #H = gate2matrix('h')
    #Z = gate2matrix('z')
    #Y = gate2matrix('y')
    
    #all single qubit clifford gates
    clifford_gates = [X, Z, S, Sdg]
    operator_norms = []
    
    for c_gate in clifford_gates:
        operator_norms.append(operator_norm(gate, c_gate).real)
    
    operator_norms = np.array(operator_norms)
    
    #the minimum index
    min_idx = np.where(operator_norms == np.amin(operator_norms))
    
    return min_idx
    
    
################################################################################################################################################

#helper function to apply the dd seqeunces
def check_and_apply(qc, idx, dd_time, gate_lengths, tm,dd_type='xyxy'):
    """Input: qc: A quantum circuit
              idx: the index of the qubit on which to possibly apply XY4
              tm: the array of timestamps where the last operation occured
              t_just_before_current_gate: time when the current gate execution started
              gate_lengths: the length of various gates
              current_discrete_time_step: the discrete time step where one is currently placed
              dd_type = The type of DD pulse
              
       Output: A quantum circuit with DD implemented if possible and feasible 
    """
    
    #extracting the gate length of the current qubit
    x_length = gate_lengths['x'][idx]
    sx_length = gate_lengths['sx'][idx]
    rz_length = gate_lengths['rz'][idx]
    id_length = gate_lengths['id'][idx]
    delay_unit = 'ns'

    print('dd_type ', dd_type)

    if dd_type == 'xx':
        print('Applying DD')
        # length of the DD sequence
        dd_seq_length = 2*x_length ## back to back X gates only 
        #the possible number of dd repetitions that can be applied
        k = int(dd_time//dd_seq_length)
        ## insert barriers to prevent any optimizations
        if( k >= 1):
            for i in range(k):
                qc.x(idx)
                qc.x(idx)

    if dd_type == 'xyxy':
        dd_seq_length = 2*x_length + 2*(2*sx_length + rz_length)
        #the possible number of dd repetitions that can be applied
        k = int(dd_time//dd_seq_length)
        ## insert barriers to prevent any optimizations
        if( k >= 1):
            for i in range(k):
                qc.x(idx)
                qc.sx(idx)
                qc.sx(idx)
                qc.rz(np.pi, idx)

                qc.x(idx)
                qc.sx(idx)
                qc.sx(idx)
                qc.rz(np.pi, idx)
                qc.barrier(idx)
    elif dd_type == 'ibmq_xx':
        dd_seq_length = 2*x_length 
        delay_duration = dd_time-dd_seq_length
        per_delay_slot_duration = int(delay_duration//4) ## split up into 4 delay slots
        if per_delay_slot_duration >0 : # insert delayed DD sequences only if there is enough time 
            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
            qc.x(idx)
            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
            qc.x(idx)
            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
        elif per_delay_slot_duration <=0 and (dd_seq_length<=dd_time): ## there is not enough time for identities but there is time for X(pi) and X(-pi)
            qc.x(idx)

    elif dd_type== 'ibmq_dd_delay': 
        ## insert dd sequence using X(pi) and X(-pi) rotations and identities ; adding the most optimal decomposition (obtained from optimization level=3 so that later when we use compiler optimization flag of 0 there are no surprises 
        ## X(pi) decomposes to Rz(pi/2), sx, sx, Rz(pi/2)
        ## X(-pi) decomposes to Rz(-pi/2), sx, sx, Rz(3pi/2)
        ## instead of identity gate- try inserting delay slots directly
        dd_seq_length = 4*(sx_length+rz_length)
        delay_duration = dd_time-dd_seq_length
        per_delay_slot_duration = int(delay_duration//4) ## split up into 4 delay slots
        if per_delay_slot_duration >0 : # insert delayed DD sequences only if there is enough time 
            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
            ## insert decomposition for X(pi)
            qc.rz((np.pi/2), idx)
            qc.sx(idx)
            qc.sx(idx)
            qc.rz((np.pi/2), idx)
            ## insert two delay slots- one post X(pi) and one pre X(-pi)
            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
            ## insert decomposition for X(-pi)
            qc.rz((-1*np.pi/2), idx)
            qc.sx(idx)
            qc.sx(idx)
            qc.rz((3*np.pi/2), idx)
            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
        
        elif per_delay_slot_duration <=0 and (dd_seq_length<=dd_time): ## there is not enough time for identities but there is time for X(pi) and X(-pi)
            ## insert decomposition for X(pi)
            qc.rz((np.pi/2), idx)
            qc.sx(idx)
            qc.sx(idx)
            qc.rz((np.pi/2), idx)
            ## insert decomposition for X(-pi)
            qc.rz((-1*np.pi/2), idx)
            qc.sx(idx)
            qc.sx(idx)
            qc.rz((3*np.pi/2), idx)
        
    ## Retiring this version because going forward we will only use exact delays
    ##elif dd_type== 'ibmq_dd_id': 
    ##    ## insert dd sequence using X(pi) and X(-pi) rotations and identities ; adding the most optimal decomposition (obtained from optimization level=3 so that later when we use compiler optimization flag of 0 there are no surprises 
    ##    ## X(pi) decomposes to Rz(pi/2), sx, sx, Rz(pi/2)
    ##    ## X(-pi) decomposes to Rz(-pi/2), sx, sx, Rz(3pi/2)
    ##    dd_seq_length = 4*(sx_length+rz_length)
    ##    identity_duration = dd_time-dd_seq_length
    ##    id_slots = int(identity_duration//(4*id_length)) ## since the identities would be evenly spread out at four regions
    ##    if dd_seq_length <= dd_time: ## insert the DD sequence only if there is enough time
    ##        for i in range(id_slots):
    ##            qc.id(idx)
    ##            qc.barrier(idx)
    ##        ## insert decomposition for X(pi)
    ##        qc.rz((np.pi/2), idx)
    ##        qc.barrier(idx)
    ##        qc.sx(idx)
    ##        qc.barrier(idx)
    ##        qc.sx(idx)
    ##        qc.barrier(idx)
    ##        qc.rz((np.pi/2), idx)
    ##        qc.barrier(idx)
    ##        ## insert two identity slots- one post X(pi) and one pre X(-pi)
    ##        for i in range(id_slots):
    ##            qc.id(idx)
    ##            qc.barrier(idx)
    ##        for i in range(id_slots):
    ##            qc.id(idx)
    ##            qc.barrier(idx)
    ##        ## insert decomposition for X(-pi)
    ##        qc.rz((-1*np.pi/2), idx)
    ##        qc.barrier(idx)
    ##        qc.sx(idx)
    ##        qc.barrier(idx)
    ##        qc.sx(idx)
    ##        qc.barrier(idx)
    ##        qc.rz((3*np.pi/2), idx)
    ##        qc.barrier(idx)
    ##        ## insert identity slots
    ##        for i in range(id_slots):
    ##            qc.id(idx)
    ##            qc.barrier(idx)

    return qc

#helper function to apply the dd seqeunces for h based skeletons- tbd
##def check_and_apply_skeleton(qc, idx, dd_time, gate_lengths, tm,dd_type='xyxy'):
##    """Input: qc: A quantum circuit
##              idx: the index of the qubit on which to possibly apply XY4
##              tm: the array of timestamps where the last operation occured
##              t_just_before_current_gate: time when the current gate execution started
##              gate_lengths: the length of various gates
##              current_discrete_time_step: the discrete time step where one is currently placed
##              dd_type = The type of DD pulse
##              
##       Output: A quantum circuit with DD implemented if possible and feasible 
##    """
##    
##    #extracting the gate length of the current qubit
##    x_length = gate_lengths['x'][idx]
##    sx_length = gate_lengths['sx'][idx]
##    rz_length = gate_lengths['rz'][idx]
##    id_length = gate_lengths['id'][idx]
##    delay_unit = 'ns' 
##
##    if dd_type == 'xyxy':
##        # length of the DD sequence
##        dd_seq_length = 2*x_length + 2*(2*sx_length + rz_length) - 2*(2*rz_length+sx_length) ## deduct twice the H gate duration
##     
##        #the possible number of dd repetitions that can be applied
##        k = int(dd_time//dd_seq_length)
##        ## insert barriers to prevent any optimizations
##        if( k >= 1):
##            # adding the decomposition of the H-gate
##            qc.rz(np.pi/2,idx)
##            qc.sx(idx)
##            qc.rz(np.pi/2,idx)
##            qc.barrier(idx)
##            for i in range(k):
##
##                qc.x(idx)
##                qc.barrier(idx)
##                qc.sx(idx)
##                qc.barrier(idx)
##                qc.sx(idx)
##                qc.barrier(idx)
##                qc.rz(np.pi, idx) 
##                qc.barrier(idx)
##
##                qc.x(idx)
##                qc.barrier(idx)
##
##                qc.sx(idx)
##                qc.barrier(idx)
##                qc.sx(idx)
##                qc.barrier(idx)
##                qc.rz(np.pi, idx) 
##                qc.barrier(idx)
##            # adding the decomposition of the H-gate
##            qc.rz(np.pi/2,idx)
##            qc.sx(idx)
##            qc.rz(np.pi/2,idx)
##    elif dd_type== 'ibmq_dd_delay': 
##        ## insert dd sequence using X(pi) and X(-pi) rotations and identities ; adding the most optimal decomposition (obtained from optimization level=3 so that later when we use compiler optimization flag of 0 there are no surprises 
##        ## X(pi) decomposes to Rz(pi/2), sx, sx, Rz(pi/2)
##        ## X(-pi) decomposes to Rz(-pi/2), sx, sx, Rz(3pi/2)
##        ## instead of identity gate- try inserting delay slots directly
##        dd_seq_length = 4*(sx_length+rz_length)- 2*(2*rz_length+sx_length)
##        delay_duration = dd_time-dd_seq_length
##        per_delay_slot_duration = int(delay_duration//4) ## split up into 4 delay slots
##        if per_delay_slot_duration >0 : # insert delayed DD sequences only if there is enough time 
##            # adding the decomposition of the H-gate
##            qc.rz(np.pi/2,idx)
##            qc.sx(idx)
##            qc.rz(np.pi/2,idx)
##            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
##            qc.barrier(idx)
##            ## insert decomposition for X(pi)
##            qc.rz((np.pi/2), idx) 
##            qc.barrier(idx)
##            qc.sx(idx)
##            qc.barrier(idx)
##            qc.sx(idx)
##            qc.barrier(idx)
##            qc.rz((np.pi/2), idx) 
##            qc.barrier(idx)
##            ## insert two delay slots- one post X(pi) and one pre X(-pi)
##            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
##            qc.barrier(idx)
##            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
##            ## insert decomposition for X(-pi)
##            qc.rz((-1*np.pi/2), idx)
##            qc.barrier(idx)
##            qc.sx(idx)
##            qc.barrier(idx)
##            qc.sx(idx)
##            qc.barrier(idx)
##            qc.rz((3*np.pi/2), idx)
##            qc.barrier(idx)
##            qc.delay(per_delay_slot_duration,idx,unit=delay_unit)
##            # adding the decomposition of the H-gate
##            qc.rz(np.pi/2,idx)
##            qc.sx(idx)
##            qc.rz(np.pi/2,idx)
##
##        elif per_delay_slot_duration <=0 and (dd_seq_length<=dd_time): ## there is not enough time for identities but there is time for X(pi) and X(-pi)
##            # adding the decomposition of the H-gate
##            qc.rz(np.pi/2,idx)
##            qc.sx(idx)
##            qc.rz(np.pi/2,idx)
##            ## insert decomposition for X(pi)
##            qc.rz((np.pi/2), idx)
##            qc.barrier(idx)
##            qc.sx(idx)
##            qc.barrier(idx)
##            qc.sx(idx)
##            qc.barrier(idx)
##            qc.rz((np.pi/2), idx)
##            qc.barrier(idx)
##            ## insert decomposition for X(-pi)
##            qc.rz((-1*np.pi/2), idx)
##            qc.barrier(idx)
##            qc.sx(idx)
##            qc.barrier(idx)
##            qc.sx(idx)
##            qc.barrier(idx)
##            qc.rz((3*np.pi/2), idx)
##            qc.barrier(idx)
##            # adding the decomposition of the H-gate
##            qc.rz(np.pi/2,idx)
##            qc.sx(idx)
##            qc.rz(np.pi/2,idx)
##
##
##    return qc



#################################################################################################################################################
#applies DD on specified qubits
def analog_IDT_to_circ(analog_IDT, gate_lengths, qubits_in_device, num_clbits, qubits_to_consider = None, mode = 'normal'):
    """Input: An analog IDT, threshold: 
       Output: A quantum circuit reconstructed by the IDT 
    """
    
    dd_modes = ['xx', 'xyxy', 'ibmq_xx', 'ibmq_dd_delay']
    print('DD insertion setup : Mode ', mode, ' Qubits to apply ', qubits_to_consider, ' Qubits in Device ', qubits_in_device, ' Num_clbits', num_clbits) 

    #getting the various gate lengths:
    cx_lengths = gate_lengths['cx']
    id_lengths = gate_lengths['id']
    sx_lengths = gate_lengths['sx']
    rz_lengths = gate_lengths['rz']
    x_lengths = gate_lengths['x']
    barrier_length = 0
    meas_length = 0
    
    #IDT shape
    analog_IDT_shape = analog_IDT.shape
    
    #number of rows in the analog IDT
    n_rows = analog_IDT_shape[0]
    
    #number of qubits in the analog IDT
    n_qubits = analog_IDT_shape[1]
    
    #the analog time values
    analog_time = list(analog_IDT.index)
    
    #the qubits names
    qubits_in_table = list(analog_IDT.columns)
    
    #initialising a new quantum circuit
    qc = QuantumCircuit(qubits_in_device, num_clbits)
    
    #time stamps for all qubits containing the previously engaged time step
    tm = qubits_in_device*[(0,0)]
    
    #time tab for each qubit
    tab = qubits_in_device*[0]
    
    #converting the analog_IDT to an array
    idt_array = analog_IDT.values
   
    #adding the operations to the quantum circuit
    for ts in range(n_rows):
        
        #getting the analog time value at the particular time step
        ts_val = analog_time[ts]
        
        #getting the time difference between current gate and previous gate
        diff = analog_time[ts]
        if ts != 0:
            diff = analog_time[ts] - analog_time[ts - 1]
        
        
        #the case when all qubits have a barrier
        flag = True
        for q__ in range(n_qubits):
            flag = flag and (idt_array[ts][q__] == 'barrier')
            ## PD: commenting out to pick up SD's changes flag = flag and (idt_array[ts][q__] == 'barrier' or idt_array[ts][q__] == 0)
        
        #checking the case where the barrier is on all qubits
        if(flag):
            
            if mode in dd_modes: 
                min_free_evolution_tab = min(tab)
                
                for qb in range(n_qubits):
                    qubit_val = int(qubits_in_table[qb])
                    if (int(tab[qubit_val]) != int(min_free_evolution_tab)):
                        
                        dd_time = tab[qubit_val] - min_free_evolution_tab
                        
                        if qubit_val in qubits_to_consider:
                            qc = check_and_apply(qc, qubit_val, dd_time, gate_lengths, tm, mode)
                            
                        tab[qubit_val] = min_free_evolution_tab
                        
                    tm[qubit_val] = (ts_val, ts)
                    
            qc.barrier()
        
        #all non-barrier operations
        else:
            for idx in range(n_qubits):
                
                #getting the physical qubit index on which the operation is acting 
                qubit_val = int(qubits_in_table[idx])
                #the gate being applied
                gate = idt_array[ts][idx]

                if gate == 0:
                    tab[qubit_val] += diff

                else:
                    gate_info = gate.split(' ')

                    if gate == 'barrier':
                        
                        #applying the dd protocol
                        if mode in dd_modes: 
                            
                            dd_time = tab[qubit_val]
                            
                            if qubit_val in qubits_to_consider:
                                qc = check_and_apply(qc, qubit_val, dd_time, gate_lengths, tm, mode)
                                
                            tab[qubit_val] = 0
                            
                        tm[qubit_val] = (ts_val, ts)
                        qc.barrier(qubit_val)
                        

                    elif gate_info[0] == 'cx' and int(gate_info[1]) == qubit_val:
                        print(gate_info)
                        
                        #the qubit which is being driven
                        sec_qubit = int(gate_info[2])
                        min_tab = min(tab[sec_qubit], tab[qubit_val])
                        
                        #applying the dd protocol on the driving qubit
                        if mode in dd_modes: 
                            print('Attempting to apply DD on ', qubits_to_consider) 
                            if(int(tab[qubit_val]) != int(min_tab)):
                                
                                dd_time = int(tab[qubit_val] - min_tab)
                                
                                if qubit_val in qubits_to_consider:
                                    
                                    qc = check_and_apply(qc, qubit_val, dd_time, gate_lengths, tm, mode)
                                    
                                tab[qubit_val] = min_tab
                                
                            tm[qubit_val] = (ts_val, ts)
                        
                        #applying the dd protocol on the qubit being driven
                        if mode in dd_modes:
                            
                            if(int(tab[sec_qubit]) != int(min_tab)):
                                
                                dd_time = int(tab[sec_qubit] - min_tab)
                                
                                if sec_qubit in qubits_to_consider:
                                    qc = check_and_apply(qc, sec_qubit, dd_time, gate_lengths, tm, mode)
                                    
                                tab[sec_qubit] = min_tab
                                
                            tm[sec_qubit] = (ts_val, ts)
                        
                        qc.cx(int(gate_info[1]), int(gate_info[2]))
                        
                        #adding to the tab now
                        tab_gate = max(diff - cx_lengths[qubit_val][sec_qubit], 0)
                        tab[qubit_val] += tab_gate
                        tab[sec_qubit] += tab_gate

                    elif gate_info[0] == 'x':
                        
                        tab_gate = max(diff - x_lengths[qubit_val], 0)
                        tab[qubit_val] += tab_gate
                        tm[qubit_val] = (ts_val, ts)
                        
                        qc.x(qubit_val)

                    elif gate_info[0] == 'rz':

                        tab_gate = max(diff - rz_lengths[qubit_val], 0)
                        tab[qubit_val] += tab_gate
                        tm[qubit_val] = (ts_val, ts)
                        
                        theta_ = float(gate_info[1][1:-1])
                        qc.rz(theta_, qubit_val)
                        

                    elif gate_info[0] == 'sx':
                        
                        tab_gate = max(diff - sx_lengths[qubit_val], 0)
                        tab[qubit_val] += tab_gate
                        tm[qubit_val] = (ts_val, ts)
                        
                        qc.sx(qubit_val)
                    
                    elif gate_info[0] == 'id':
                        
                        tab_gate = max(diff, 0)
                        tab[qubit_val] += tab_gate
                        tm[qubit_val] = (ts_val, ts)
                        #when the mode is 'dd' we replace the free evolution with DD pulses
                        #identity gate is implemented in IBM computers as a free evolution period.
                        if mode == 'normal':
                            qc.id(qubit_val)
                        else:
                            pass
                        ## PD: commenting out to pick up SD's bug fixes
                        ##tab_gate = max(diff - id_lengths[qubit_val], 0)
                        ##tab[qubit_val] += tab_gate
                        ##tm[qubit_val] = (ts_val, ts)
                        ##
                        ##qc.id(qubit_val)
                        ##
                    elif gate_info[0] == 'measure':
                        
                        #applying the dd protocol
                        if mode in dd_modes: 
                            
                            dd_time = tab[qubit_val]
                            
                            if qubit_val in qubits_to_consider:
                                qc = check_and_apply(qc, qubit_val, dd_time, gate_lengths, tm, mode)
                                
                            tm[qubit_val] = (ts_val, ts)
                        
                        clbit = int(gate_info[-1])
                        qc.measure(qubit_val, clbit)

                    else:
                        pass
    
    return qc

#################################################################################################################################################
def analog_IDT_to_skeleton_circ(analog_IDT, gate_lengths, qubits_in_device, num_clbits, qubits_to_consider = None, mode = 'normal'):
    """Input: An analog IDT 
       qubits_to_consider: All the qubits on which to apply DD
       Output: A quantum circuit reconstructed by the IDT 
    """
    dd_modes = ['xx', 'xyxy', 'ibmq_xx', 'ibmq_dd_delay']
    
    #getting the various gate lengths:
    cx_lengths = gate_lengths['cx']
    id_lengths = gate_lengths['id']
    sx_lengths = gate_lengths['sx']
    rz_lengths = gate_lengths['rz']
    x_lengths = gate_lengths['x']
    barrier_length = 0
    meas_length = 0
    
    #IDT shape
    analog_IDT_shape = analog_IDT.shape
    
    #number of rows in the analog IDT
    n_rows = analog_IDT_shape[0]
    
    #number of qubits in the analog IDT
    n_qubits = analog_IDT_shape[1]
    
    #the analog time values
    analog_time = list(analog_IDT.index)
    
    #the qubits names
    qubits_in_table = list(analog_IDT.columns)
    
    #initialising a new quantum circuit
    qc = QuantumCircuit(qubits_in_device, num_clbits)
    
    #time stamps for all qubits containing the previously engaged time step
    tm = qubits_in_device*[0]
    
    #time tab for each qubit
    tab = qubits_in_device*[0]
    
    #converting the analog_IDT to an array
    idt_array = analog_IDT.values
    
   
    #adding the operations to the quantum circuit
    for ts in range(n_rows):
        
        #getting the analog time value at the particular time step
        ts_val = analog_time[ts]
        
        #getting the time difference between current gate and previous gate
        diff = analog_time[ts]
        if ts != 0:
            diff = analog_time[ts] - analog_time[ts - 1]
        
        
        #the case when all qubits have a barrier
        flag = True
        for q__ in range(n_qubits):
            flag = flag and (idt_array[ts][q__] == 'barrier')
            ## PD: commenting out to pick up SD's bug fix flag = flag and (idt_array[ts][q__] == 'barrier' or idt_array[ts][q__] == 0)
        
        #checking the case where the barrier is on all qubits
        if(flag):
            
            if mode in dd_modes: 
                
                min_free_evolution_tab = min(tab)
                
                for qb in range(n_qubits):
                    qubit_val = int(qubits_in_table[qb])
                    if (int(tab[qubit_val]) != int(min_free_evolution_tab)):
                        
                        dd_time = tab[qubit_val] - min_free_evolution_tab
                        if (qubit_val in qubits_to_consider):
                            qc = check_and_apply(qc, qubit_val, dd_time, gate_lengths, tm, mode)
                            
                        tab[qubit_val] = min_free_evolution_tab
                        
                    tm[qubit_val] = (ts_val, ts)
                    
            qc.barrier()
        
        #all non-barrier operations
        else:
            for idx in range(n_qubits):
                
                #getting the physical qubit index on which the operation is acting 
                qubit_val = int(qubits_in_table[idx])
                
                #the gate being applied
                gate = idt_array[ts][idx]

                if gate == 0:
                    tab[qubit_val] += diff

                else:
                    gate_info = gate.split(' ')

                    if gate == 'barrier':
                        
                        #applying the dd protocol
                        if mode in dd_modes:                            
                            dd_time = tab[qubit_val]
                            if (qubit_val in qubits_to_consider):
                                qc = check_and_apply(qc, qubit_val, dd_time, gate_lengths, tm, mode)
                                
                            tab[qubit_val] = 0
                            
                        tm[qubit_val] = (ts_val, ts)
                        qc.barrier(qubit_val)
                        

                    elif gate_info[0] == 'cx' and int(gate_info[1]) == qubit_val:
                        
                        #the qubit which is being driven
                        sec_qubit = int(gate_info[2])
                        min_tab = min(tab[sec_qubit], tab[qubit_val])
                        
                        #applying the dd protocol on the driving qubit
                        if mode in dd_modes: 
                            
                            if(int(tab[qubit_val]) != int(min_tab)):
                                
                                dd_time = int(tab[qubit_val] - min_tab)
                                
                                if (qubit_val in qubits_to_consider):
                                    qc = check_and_apply(qc, qubit_val, dd_time, gate_lengths, tm, mode)
                                    
                                tab[qubit_val] = min_tab
                                
                            tm[qubit_val] = (ts_val, ts)
                        
                        #applying the dd protocol on the qubit being driven
                        if mode in dd_modes: 
                            
                            if(int(tab[sec_qubit]) != int(min_tab)):
                                
                                dd_time = int(tab[sec_qubit] - min_tab)
                                
                                if (sec_qubit in qubits_to_consider):
                                    qc = check_and_apply(qc, sec_qubit, dd_time, gate_lengths, tm, mode)
                                    
                                tab[sec_qubit] = min_tab
                                
                            tm[sec_qubit] = (ts_val, ts)
                        
                        qc.cx(int(gate_info[1]), int(gate_info[2]))
                        
                        #adding to the tab now
                        tab_gate = max(diff - cx_lengths[qubit_val][sec_qubit], 0)
                        tab[qubit_val] += tab_gate
                        tab[sec_qubit] += tab_gate

                    elif gate_info[0] == 'x':
                        
                        tab_gate = max(diff - x_lengths[qubit_val], 0)
                        tab[qubit_val] += tab_gate
                        tm[qubit_val] = (ts_val, ts)
                        
                        qc.x(qubit_val)

                    elif gate_info[0] == 'rz':

                        tab_gate = max(diff - rz_lengths[qubit_val], 0)
                        tab[qubit_val] += tab_gate
                        tm[qubit_val] = (ts_val, ts)
                        
                        #rz rotation angle
                        theta_ = float(gate_info[1][1:-1])
                        
                        #retrieve the gate
                        temp_circ = QuantumCircuit(1)
                        temp_circ.rz(theta_, 0)
                        gate = Operator(temp_circ).data
                        
                        #applying the closest clifford gate
                        idx = closest_clifford(gate)[0][0]
                        
                        if idx == 0:
                            qc.x(qubit_val)	# X gate
                            tab[qubit_val] -= max(diff - rz_lengths[qubit_val], 0)
                            tab[qubit_val] += max(diff - x_lengths[qubit_val], 0)
                        
                        elif idx == 1:
                            qc.rz(np.pi, qubit_val)	# Z gate
                        elif idx == 2:
                            qc.rz(np.pi/2, qubit_val)	# S gate
                        else:
                            qc.rz(-np.pi/2, qubit_val)	# Sdg gate
                            
                    elif gate_info[0] == 'sx':
                        
                        tab_gate = max(diff - sx_lengths[qubit_val], 0)
                        tab[qubit_val] += tab_gate
                        tm[qubit_val] = (ts_val, ts)
                        
                        #retrieve the gate
                        temp_circ = QuantumCircuit(1)
                        temp_circ.sx(0)
                        gate = Operator(temp_circ).data
                        
                        
                        # replacing the sx gate with x gate
                        qc.x(qubit_val)
                       
                    
                    elif gate_info[0] == 'id':
                        tab_gate = max(diff, 0)
                        tab[qubit_val] += tab_gate
                        tm[qubit_val] = (ts_val, ts)
                        #when the mode is 'dd' we replace the free evolution with DD pulses
                        #identity gate is implemented in IBM computers as a free evolution period.
                        if mode == 'normal':
                            qc.id(qubit_val)
                        else:
                            pass
                       
                        ## PD: Commenting out this code to pick up SD's bug fix
                        ## tab_gate = max(diff - id_lengths[qubit_val], 0)
                        ## tab[qubit_val] += tab_gate
                        ## tm[qubit_val] = (ts_val, ts) 
                        ## 
                        ## qc.id(qubit_val)
                        
                    elif gate_info[0] == 'measure':
                        
                        #applying the dd protocol
                        if mode in dd_modes: 
                            
                            dd_time = tab[qubit_val]
                            
                            if (qubit_val in qubits_to_consider):
                                qc = check_and_apply(qc, qubit_val, dd_time, gate_lengths, tm,mode)
                                
                            tm[qubit_val] = (ts_val, ts)
                        
                        clbit = int(gate_info[-1])
                    
                        qc.measure(qubit_val, clbit)

                    else:
                        pass
    
    return qc

################################################################################################################################################
