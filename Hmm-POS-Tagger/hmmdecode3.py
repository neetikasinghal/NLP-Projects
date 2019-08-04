import sys
import json
import numpy as np
import time

def read_input(file_name):
    document = []
    with open(file_name, 'r', encoding='utf-8') as fp:
        for each_line in fp:
            each_line = each_line.strip()
            document.append(each_line)
    return document

def cal_transition_matrix(model_data):
    transition_list = []
    # transition_list.append(list(model_data['start_states'].values()))
    transition_states = model_data['transition_states']
    for key, val in transition_states.items():
        transition_list.append(list(val.values()))
    transition_mat = np.array(transition_list)
    return transition_mat

def cal_emission_matrix(model_data):
    emission_states = model_data['emission_states']
    emission_list = []
    for key, val in emission_states.items():
        emission_list.append(list(val.values()))
    emission_mat = np.array(emission_list)
    return emission_mat

def viterbi_decoding(line, states, transition_mat, emission_mat, initial_states, word_dict):
    time_points = line.split(' ')
    states_len = len(states)
    time_points_len = len(time_points)
    probabilty = np.zeros((states_len, time_points_len))
    back_pointer = np.zeros((states_len, time_points_len))
    if word_dict.get(time_points[0]) is not None:
        emission_index = word_dict.get(time_points[0])
        probabilty[:, 0] = initial_states * emission_mat[emission_index, :]
    else:
        probabilty[:, 0] = initial_states

    for t in range(1, time_points_len):
        for i in range(0, states_len):
            emission_val = emission_mat[word_dict.get(time_points[t])][i] if word_dict.get(time_points[t]) is not None else 1
            mul_val = probabilty[:, t-1] * transition_mat[:, i]
            probabilty[i, t] = np.max([mul_val * emission_val])
            back_pointer[i, t] = np.argmax([mul_val])
    most_proba_state = np.argmax([probabilty[:, time_points_len-1]])

    labels = []
    for j in range(time_points_len-1, -1, -1):
        labels.append(states[most_proba_state])
        most_proba_state = int(back_pointer[most_proba_state, j])
    labels.reverse()
    ans_list = []
    for i in range(0, len(labels)):
        ans_list.append(time_points[i]+'/'+labels[i])
    ans = ' '.join(ans_list)
    return ans

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = "hmmoutput.txt"
    model_file = "hmmmodel.txt"
    document = read_input(input_file)
    start = time.time()
    with open(model_file) as f:
        model_data = json.load(f)
    length = len(model_data['transition_states'])
    transition_mat = cal_transition_matrix(model_data)
    emission_mat = cal_emission_matrix(model_data)
    words = list(model_data['emission_states'].keys())
    words_index = list(range(0, len(words)))
    word_dict = dict(zip(words, words_index))
    states = list(model_data['start_states'].keys())
    states_val = np.array(list(model_data['start_states'].values()))
    fout = open(output_file, "w")
    for line in document:
        result = viterbi_decoding(line, states, transition_mat, emission_mat, states_val, word_dict)
        fout.write('%s\n' % result)

    fout.close()
    print(time.time() - start)