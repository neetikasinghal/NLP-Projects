import json
import math
import os
import re
import sys

def preprocessing(testing_data_dict):
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    for key, val in testing_data_dict.items():
        testing_data_dict[key]= re.sub(r'[^\w\s]', '', testing_data_dict[key])
        testing_data_dict[key]= re.sub('\\d', '', testing_data_dict[key])
        testing_data_dict[key] = " ".join([x for x in testing_data_dict[key].split() if x not in stop])
    return testing_data_dict

def cal_proba(sentence, model_data, category):
    words = sentence.split()
    prob = math.log(model_data['prior_proba'][category+'_prior'])
    for word in words:
        ## ignore unseen words
        if model_data.get(word) is not None:
            val = model_data[word].get(category)
            prob = prob + math.log(val)
    return prob

def predict_class(sentence, model_file):
    predict_dict = dict()
    with open(model_file) as f:
        model_data = json.load(f)
    predict_dict['deceptive positive'] = cal_proba(sentence, model_data, 'positive_deceptive')
    predict_dict['deceptive negative'] = cal_proba(sentence, model_data, 'negative_deceptive')
    predict_dict['truthful positive'] = cal_proba(sentence, model_data, 'positive_truthful')
    predict_dict['truthful negative'] = cal_proba(sentence, model_data, 'negative_truthful')
    predicted_class = max(predict_dict, key=predict_dict.get)
    f.close()
    return predicted_class

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
                    file_path = z+'/'+fileRead
                    testing_data_dict[file_path] = document
    return testing_data_dict

if __name__ == '__main__':
    model_file = "nbmodel.txt"
    output_file = "nboutput.txt"
    input_path = str(sys.argv[1])
    testing_data_dict = read_testing_data(input_path)
    preprocessing(testing_data_dict)
    result = dict()
    for key, val in testing_data_dict.items():
        result[key] = predict_class(val, model_file)
    with open(output_file, 'w') as f:
        for key, value in result.items():
            f.write('%s %s\n' % (value, key))
    f.close()

