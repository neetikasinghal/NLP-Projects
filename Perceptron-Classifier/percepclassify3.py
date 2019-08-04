import sys
import os
import re
import json
import numpy as np

def read_testing_data(path_to_testing_data):
    testing_data_dict = dict()
    pos_neg_dir_list = next(os.walk(path_to_testing_data))[1]
    for pos_neg_dir in pos_neg_dir_list:
        x = os.path.join(path_to_testing_data, pos_neg_dir)
        tru_dec_dir_list = next(os.walk(x))[1]
        for tru_dec_dir in tru_dec_dir_list:
            y = os.path.join(x, tru_dec_dir)
            folders = next(os.walk(y))[1]
            for fold in folders:
                z = os.path.join(y, fold)
                files = next(os.walk(z))[2]
                for fileRead in files:
                    fileop = open(os.path.join(z, fileRead), "r")
                    document = fileop.read().lower().rstrip('\n')
                    document = preprocessing(document)
                    file_path = z+'/'+fileRead
                    testing_data_dict[file_path] = document
    return testing_data_dict


def preprocessing(doc):
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    doc= re.sub(r'[^\w\s]', '', doc)
    doc= re.sub('\\d', '', doc)
    doc = " ".join([x for x in doc.split() if x not in stop])
    return doc


def word_count(string):
    counts = dict()
    words = string.split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


def compute_feature_vector(feature_list, document):
    doc_words = word_count(document)
    feature_vector = np.zeros(len(feature_list))
    i=0
    for word in feature_list:
        if doc_words.get(word) is not None:
            feature_vector[i] = doc_words.get(word)
        i+=1
    return feature_vector


def predict_class(sentence, model_file):
    with open(model_file) as f:
        model_data = json.load(f)
    tru_dec_dict = model_data['tru_dec']
    tru_dec_bias = model_data['tru_dec_bias']
    tru_dec_features = list(tru_dec_dict.keys())
    tru_dec_weights = list(tru_dec_dict.values())
    tru_dec_weights = np.array(tru_dec_weights)
    tru_dec_feature_vector = compute_feature_vector(tru_dec_features, sentence)
    tru_dec_feature_vector = np.where(tru_dec_feature_vector > 0, 1, 0)
    activation = np.dot(tru_dec_weights, tru_dec_feature_vector)+tru_dec_bias
    if activation>0:
        result = 'truthful'
    else:
        result = 'deceptive'

    pos_neg_dict = model_data['pos_neg']
    pos_neg_bias = model_data['pos_neg_bias']
    pos_neg_features = list(pos_neg_dict.keys())
    pos_neg_weights = list(pos_neg_dict.values())
    pos_neg_weights = np.array(pos_neg_weights)
    pos_neg_feature_vector = compute_feature_vector(pos_neg_features, sentence)
    pos_neg_feature_vector = np.where(pos_neg_feature_vector > 0, 1, 0)
    activation = np.dot(pos_neg_weights, pos_neg_feature_vector)+pos_neg_bias
    if activation>0:
        result += ' positive'
    else:
        result += ' negative'
    f.close()
    return result

def evaluate(result, label_pos, label_neg):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for key, val in result.items():
        if label_pos in key and label_pos in val:
            true_positive+=1
        if label_neg in key and label_pos in val:
            false_positive+=1
        if label_neg in key and label_neg in val:
            true_negative+=1
        if label_pos in key and label_neg in val:
            false_negative+=1
    recall = true_positive / float(true_positive + false_negative)
    precision = true_positive / float(true_positive + false_positive)
    fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_positive+true_negative) / float(true_negative+true_positive+false_positive+false_negative)
    return fscore

def print_f1_scores():
    f1_truthful = evaluate(result, 'truthful', 'deceptive')
    print('truthful-f1: ' + str(f1_truthful))
    f1_deceptive = evaluate(result, 'deceptive', 'truthful')
    print('deceptive-f1: ' + str(f1_deceptive))
    f1_positive = evaluate(result, 'positive', 'negative')
    print('positive-f1: ' + str(f1_positive))
    f1_negative = evaluate(result, 'negative', 'positive')
    print('negative-f1: ' + str(f1_negative))

if __name__ == "__main__":
    model_file = str(sys.argv[1])
    output_file = "percepoutput.txt"
    input_path = str(sys.argv[2])

    testing_data_dict = read_testing_data(input_path)
    result = dict()
    for key, val in testing_data_dict.items():
        result[key] = predict_class(val, model_file)
    with open(output_file, 'w') as f:
        for key, value in result.items():
            f.write('%s %s\n' % (value, key))
    f.close()
    print_f1_scores()