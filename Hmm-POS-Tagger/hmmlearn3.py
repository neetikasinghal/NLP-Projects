import sys
import json

def read_input(file_name):
    document = []
    with open(file_name, 'r', encoding='utf-8') as fp:
        for each_line in fp:
            each_line = each_line.strip()
            document.append(each_line)
    return document

def process_doc(document):
    start_states = dict()
    transition_dict = dict()
    emission_dict = dict()
    word_dict = dict()
    for line in document:
        flag = 0
        prev_tag = None
        words = line.split(' ')
        for word in words:
            word_tag = word.rsplit('/',1)
            tag = word_tag[1]
            if flag == 0:
                flag = 1
                start_states[tag] = start_states[tag]+1 if start_states.get(tag) is not None else 1

            if prev_tag is not None:
                if transition_dict.get(prev_tag) is None:
                    transition_dict[prev_tag] = dict()
                transition_dict[prev_tag][tag] = transition_dict[prev_tag][tag] + 1 if transition_dict[prev_tag].get(tag) is not None else 1

            prev_tag = tag

            emission_dict[tag] = emission_dict[tag]+1 if emission_dict.get(tag) is not None else 1

            if word_dict.get(word_tag[0]) is None:
                word_dict[word_tag[0]] = dict()
            word_dict[word_tag[0]][tag] = word_dict[word_tag[0]][tag]+1 if word_dict[word_tag[0]].get(tag) is not None else 1

    return start_states, transition_dict, emission_dict, word_dict

def cal_transition_proba(start_states, transition_dict, emission_dict, total_lines):
    transition_proba = dict()
    transition_proba['start_states'] = dict()
    total_tags = len(emission_dict)
    vocab_size = ((total_tags) * (total_tags-1))/2
    transition_proba['transition_states'] = dict()

    for key, val in emission_dict.items():
        count = start_states[key] if start_states.get(key) is not None else 0
        transition_proba['start_states'][key] = (count + 1) / (total_lines + total_tags)
        if transition_proba['transition_states'].get(key) is None:
            transition_proba['transition_states'][key] = dict()
        for k, v in emission_dict.items():
            cnt = transition_dict[key][k] if transition_dict.get(key) is not None and transition_dict[key].get(k) is not None else 0
            sum_val = sum(transition_dict[key].values()) if transition_dict.get(key) is not None else 0
            transition_proba['transition_states'][key][k] = (cnt+1) / (sum_val+vocab_size)

    return transition_proba

def cal_emission_proba(emission_dict, word_dict):
    emission_proba = dict()
    emission_proba['emission_states'] = dict()
    for key, val in word_dict.items():
        emission_proba['emission_states'][key] = dict()
        for k, v in emission_dict.items():
            count = word_dict[key][k] if word_dict[key].get(k) is not None else 0
            emission_proba['emission_states'][key][k] = count/v;
    return emission_proba

if __name__ == "__main__":
    model_file = "hmmmodel.txt"
    output_file = "hmmoutput.txt"
    train_file = sys.argv[1]
    document = read_input(train_file)
    start_states, transition_dict, emission_dict, word_dict = process_doc(document)
    transition_proba = cal_transition_proba(start_states, transition_dict, emission_dict, len(document))
    emission_proba = cal_emission_proba(emission_dict, word_dict)
    master_dict = {**transition_proba, **emission_proba}
    json_model = json.dumps(master_dict, indent=2)
    f1 = open(model_file, "w")
    f1.write(json_model)
    f1.close()
