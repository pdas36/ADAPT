import ast
import pickle
def write_data_dictionary_into_logfile(data_dictionary,outputfname):
    f = open(outputfname,"+w")
    f.write('data_dictionary = '+str(data_dictionary)+'\n')
    f.close()
def read_data_dictionary_from_logfile(logname):
    with open(logname) as f:
        for line in f:
            start_pos = line.find('= ') + 2 
            end_pos   = line.find('\n')
            data_from_log = ast.literal_eval(line[start_pos:end_pos])
    f.close()
    return data_from_log
def dump_objects_into_pickle_file(fname,post_compile_qobjs,job_ids,job_noise_properties,job_time,job_results): 
    with open(fname, 'wb') as output: 
        pickle.dump(post_compile_qobjs, output, pickle.HIGHEST_PROTOCOL) 
        pickle.dump(job_ids, output, pickle.HIGHEST_PROTOCOL) 
        pickle.dump(job_noise_properties, output, pickle.HIGHEST_PROTOCOL) 
        pickle.dump(job_time, output, pickle.HIGHEST_PROTOCOL) 
        pickle.dump(job_results, output, pickle.HIGHEST_PROTOCOL)

def read_from_pickle_dump(fname): 
    with open(fname, 'rb') as input: 
        post_compile_qobjs = pickle.load(input) 
        job_ids = pickle.load(input) 
        job_noise_properties = pickle.load(input) 
        job_time = pickle.load(input) 
        job_results = pickle.load(input)

    return post_compile_qobjs, job_ids, job_noise_properties, job_time, job_results
