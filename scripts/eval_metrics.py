from collections import Counter
import numpy as np
import math

def normalize_dict(input_dict):
    '''
    Function to normalize an input dictionary
    '''
    epsilon = 0.0000001
    if sum(input_dict.values()) == 0:
        print(input_dict)
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


def compute_pst(ideal_histogram,noisy_histogram,num_sols):
    ''' 
    Function to compute PST
    '''
    # determine the total best solutions from ideal
    sorted_histogram = sorted(ideal_histogram.items(), key=lambda x: x[1], reverse=True)
    successful_trials_counter = 0 
    for i in range(num_sols):
        search_key = sorted_histogram[i][0]
        for key,value in noisy_histogram.items():
            if(key == search_key):
                successful_trials_counter = successful_trials_counter + value
    # compute PST
    total_trials = sum(noisy_histogram.values())
    if(successful_trials_counter <=1.0): #already a pdf
        pst = successful_trials_counter
    else:
        pst = successful_trials_counter/total_trials
    return pst 

def compute_ist(ideal_histogram,noisy_histogram,num_sols):
    ''' 
    Function to compute IST
    '''
    # determine the total best solutions from ideal
    sorted_histogram = sorted(ideal_histogram.items(), key=lambda x: x[1], reverse=True)
    # sort the noisy histogram
    sorted_noisy_histogram = sorted(noisy_histogram.items(), key=lambda x: x[1], reverse=True)
    # probability of correct answer
    successful_trials_counter = 0 
    for i in range(num_sols):
        search_key = sorted_histogram[i][0]
        for key,value in noisy_histogram.items():
            if(key == search_key):
                successful_trials_counter = successful_trials_counter + value
    # get the solution keys
    solution_keys = []
    for j in range(num_sols):
        solution_keys.append(sorted_histogram[j][0])
        
    error_counter = 0 
    for i in range(len(sorted_noisy_histogram)):
        search_key = sorted_noisy_histogram[i][0]
        if search_key not in solution_keys:
            error_counter = sorted_noisy_histogram[i][1]
            break
    ist = successful_trials_counter/error_counter

    return ist 



def update_dist(dict1,dict2):
    ''' 
    Function to merge two dictionaries in to a third one
    '''
    dict3 = Counter(dict1) + Counter(dict2) 
    dict3 = dict(dict3)
    return dict3
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
    
def removekey(d, key_list):
    for i in key_list:
        r = dict(d)
        del r[i]
    
    return r
    
def compute_hdist(dist_a,dist_b):
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
				s = (math.sqrt(p) - math.sqrt(q)) ** 2
				list_of_squares.append(s)
				break
	# calculate the sum of squares
	sosq = sum(list_of_squares)
	hdist = math.sqrt(sosq)/math.sqrt(2)
	corr = 1-hdist	
	return hdist,corr
				
## functions to evaluate the expectation value


def compute_weight_matrix(_G):
    n = len(_G.nodes())
    w = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            temp = _G.get_edge_data(i,j,default=0)
            if temp != 0:
                w[i,j] = temp['weight']
    return w

def compute_cost_of_cut(graph_cut,weight_matrix):
    
    n=len(graph_cut)
    cost = 0
    for i in range(n):
        for j in range(n):
            cost = cost + weight_matrix[i,j]* int(graph_cut[i])* (1- int(graph_cut[j]))
    
    return cost


def compute_expected_value(_out_dict,in_graph):
    
    # check if cut is valid
    
    out_dict = normalize_dict(_out_dict.copy())
#     print(out_dict)
    W = compute_weight_matrix(in_graph)
    E = 0
    for key in out_dict:
        key_lst=[] 
        key_lst[:0]=key 
        cost = compute_cost_of_cut(key_lst,W)
        E += out_dict[key]*cost
    
    return E

def obtain_approximation_ratio(_out_dict,in_graph,solution):
	## obtain value of cost function from mean of all samples
	#print(solution)
	W = compute_weight_matrix(in_graph)
	mean_from_all_samples = compute_expected_value(_out_dict,in_graph)
	best_cut_value = compute_cost_of_cut(solution,W)
	#print(mean_from_all_samples,best_cut_value)
	
	return mean_from_all_samples/best_cut_value

def obtain_approximation_ratio_gap(ideal_dict,noisy_dict,in_graph,solution):
	ar_ideal = obtain_approximation_ratio(ideal_dict,in_graph,solution)
	ar_noisy = obtain_approximation_ratio(noisy_dict,in_graph,solution)

	return 100*abs(ar_ideal-ar_noisy)/ar_ideal 

def norm(numbers):
    if isinstance(numbers,list)==1:
        sum_of_numbers = 0 
        for i in numbers:
            sum_of_numbers = sum_of_numbers + math.pow(i,2)
        return math.sqrt(sum_of_numbers)
    else:
        return math.sqrt(math.pow(numbers,2))

def tvd_two_dist(p,q):
    _p = p.copy()
    _p = normalize_dict(_p)
    _q = q.copy()
    _q = normalize_dict(_q)
    
    epsilon = 0.0000000001
    ## match both dictionaries
    for key in _p.keys():
        if key not in _q.keys():
            _q[key] = epsilon
    
    for key in _q.keys():
        if key not in _p.keys():
            _p[key] = epsilon

    _p = normalize_dict(_p)
    _q = normalize_dict(_q)

    _q_rearranged = {}
    for key,value in _p.items():
        _q_rearranged[key] = _q[key]

    ## compute_tvd
    tvd = 0 
    for key,value in _p.items():
        diff = value - _q_rearranged[key]
        tvd = tvd + norm(diff)
    return tvd/2

def fidelity_from_tvd(p,q):
    epsilon = 0.0000001
    tvd = tvd_two_dist(p,q)
    fidelity = 1-tvd
    return fidelity

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
