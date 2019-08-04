import json
import os
import re
import sys

count_dict = dict()
count_dict['positive_deceptive']=0
count_dict['negative_deceptive']=0
count_dict['positive_truthful']=0
count_dict['negative_truthful']=0

def read_training_data(path_to_training_data):
    training_data_dict = dict()
    pos_neg_dir_list = next(os.walk(path_to_training_data))[1]
    for pos_neg_dir in pos_neg_dir_list:
        if pos_neg_dir.startswith("positive"):
            key = "positive"
        elif pos_neg_dir.startswith("negative"):
            key = "negative"
        x = os.path.join(path_to_training_data, pos_neg_dir)
        tru_dec_dir_list = next(os.walk(x))[1]
        for tru_dec_dir in tru_dec_dir_list:
            if tru_dec_dir.startswith("deceptive"):
                final_key = key + "_deceptive"
            elif tru_dec_dir.startswith("truthful"):
                final_key = key + "_truthful"
            y = os.path.join(x, tru_dec_dir)
            folders = next(os.walk(y))[1]
            str_list = list()

            for fold in folders:
                z = os.path.join(y, fold)
                files = next(os.walk(z))[2]
                for fileRead in files:
                    fileop = open(os.path.join(z, fileRead), "r")
                    document = fileop.read().lower().rstrip('\n')
                    str_list.append(document)
                    count_dict[final_key]+=1
            fullStr = ' '.join(str_list)
            training_data_dict[final_key]=(fullStr)
    return training_data_dict

def preprocessing(training_data_dict):
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    for key, val in training_data_dict.items():
        training_data_dict[key]= re.sub(r'[^\w\s]', '', training_data_dict[key])
        training_data_dict[key]= re.sub('\\d', '', training_data_dict[key])
        training_data_dict[key] = " ".join([x for x in training_data_dict[key].split() if x not in stop])
    return training_data_dict

def word_count(string):
    counts = dict()
    words = string.split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts

def proba_given_word(word_count_dict):
    proba = dict()
    total_words = sum(word_count_dict.values())
    for key, val in word_count_dict.items():
        proba[key] = val/total_words
    return proba

def laplace_smoothing(word_count_dict, len_total_unique_words, word):
    total_words = sum(word_count_dict.values())
    val = 0
    if word_count_dict.get(word) is not None:
        val = word_count_dict[word]
    laplace = float(val+1)/float(total_words+len_total_unique_words)
    return laplace

def build_master_dict(comp_str, prior_prob_dict, total_unique_words, pos_dec, neg_dec, pos_tru, neg_tru):
    master_dict = dict()
    master_dict['prior_proba'] = prior_prob_dict
    words = comp_str.split()
    for word in words:
        master_dict[word] = dict()
        master_dict[word]['positive_deceptive'] = laplace_smoothing(pos_dec, total_unique_words, word)
        master_dict[word]['negative_deceptive'] = laplace_smoothing(neg_dec, total_unique_words, word)
        master_dict[word]['positive_truthful'] = laplace_smoothing(pos_tru, total_unique_words, word)
        master_dict[word]['negative_truthful'] = laplace_smoothing(neg_tru, total_unique_words, word)
    return master_dict

def naive_bayes(training_data_dict):

    pos_dec = training_data_dict['positive_deceptive']
    neg_dec = training_data_dict['negative_deceptive']
    pos_tru = training_data_dict['positive_truthful']
    neg_tru = training_data_dict['negative_truthful']
    complete_str = pos_dec + neg_dec + pos_tru + neg_tru

    ## Prior Probabilites
    prior_prob_dict = dict()
    prior_prob_dict['positive_deceptive_prior'] = count_dict['positive_deceptive']/sum(count_dict.values())
    prior_prob_dict['negative_deceptive_prior'] = count_dict['negative_deceptive']/sum(count_dict.values())
    prior_prob_dict['positive_truthful_prior'] = count_dict['positive_truthful']/sum(count_dict.values())
    prior_prob_dict['negative_truthful_prior'] = count_dict['negative_truthful']/sum(count_dict.values())
    # print(prior_prob_dict)
    word_count_pos_dec = word_count(pos_dec)
    word_count_neg_dec = word_count(neg_dec)
    word_count_pos_tru = word_count(pos_tru)
    word_count_neg_tru = word_count(neg_tru)

    word_count_comp_str = word_count(complete_str)
    total_unique_words = len(word_count_comp_str)
    master_dict = build_master_dict(complete_str, prior_prob_dict, total_unique_words, word_count_pos_dec, word_count_neg_dec, word_count_pos_tru, word_count_neg_tru)

    return master_dict

if __name__ == '__main__':

    model_file = "nbmodel.txt"
    input_path = str(sys.argv[1])
    training_data_dict= read_training_data(input_path)
    preprocessing(training_data_dict)
    master_dict = naive_bayes(training_data_dict)
    json = json.dumps(master_dict, indent=2)
    f = open(model_file, "w")
    f.write(json)
    f.close()