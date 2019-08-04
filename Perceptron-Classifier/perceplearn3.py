import os
import re
import sys
import math
import numpy as np
import time
import operator
import random
import json


def read_training_data(path_to_training_data):
    training_data_dict = dict()
    training_data_list_dict = dict()
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
            training_data_list_dict[final_key] = list()

            for fold in folders:
                z = os.path.join(y, fold)
                files = next(os.walk(z))[2]
                for fileRead in files:
                    fileop = open(os.path.join(z, fileRead), "r")
                    document = fileop.read().lower().rstrip('\n')
                    document = preprocessing(document)
                    str_list.append(document)
                    training_data_list_dict[final_key].append(document)
            fullStr = ' '.join(str_list)
            training_data_dict[final_key]=(fullStr)
    return training_data_dict, training_data_list_dict


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


def find_vocabulary(word_class):
    complete_str_pos_neg = word_class['positive'] + word_class['negative']
    complete_str_tru_dec = word_class['truthful'] + word_class['deceptive']
    word_count_pos_neg = word_count(complete_str_pos_neg)
    word_count_tru_dec = word_count(complete_str_tru_dec)
    return word_count_pos_neg, word_count_tru_dec


def compute_word_class(training_data_dict):
    word_class = dict()
    word_class['positive'] = training_data_dict['positive_deceptive'] + training_data_dict['positive_truthful']
    word_class['negative'] = training_data_dict['negative_deceptive'] + training_data_dict['negative_truthful']
    word_class['deceptive'] = training_data_dict['positive_deceptive'] + training_data_dict['negative_deceptive']
    word_class['truthful'] = training_data_dict['positive_truthful'] + training_data_dict['negative_truthful']
    return word_class


def compute_idf(word_dict):
    idf_dict = {}
    N = len(word_dict)
    for word, val in word_dict.items():
        idf_dict[word] = math.log10(N/float(val))
    return idf_dict


def compute_tf_idf(word_count, total_words, idf_dict, word):
    if word_count is None:
        word_count = 0
    tf = word_count / float(total_words)
    tf_idf = tf * idf_dict[word]
    return tf_idf


def feature_selection(word_dict, word_class, label, no_of_features):
    word_list = list(word_dict.keys())
    if label=='pos_neg':
        class_1 = word_count(word_class['positive'])
        class_2 = word_count(word_class['negative'])
    elif label=='tru_dec':
        class_1 = word_count(word_class['truthful'])
        class_2 = word_count(word_class['deceptive'])
    idf_dict = compute_idf(word_dict)
    result_list = [[0], [0]]
    tf_idf_mat = np.array(result_list)
    for word in word_list:
        tf_idf_class1 = compute_tf_idf(class_1.get(word), len(class_1), idf_dict, word)
        tf_idf_class2 = compute_tf_idf(class_2.get(word), len(class_2), idf_dict, word)
        array_list = [[tf_idf_class1], [tf_idf_class2]]
        array = np.array(array_list)
        tf_idf_mat = np.append(tf_idf_mat, array, axis=1)
    tf_idf_mat = tf_idf_mat[:, 1:]

    mean_tf_idf = tf_idf_mat.mean(axis=0)
    mean_tf_idf_list = mean_tf_idf.tolist()
    mean_tf_idf_dict = dict(zip(word_list, mean_tf_idf_list))
    sorted_list = sorted(mean_tf_idf_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_list = sorted_list[0:no_of_features]
    top_features = [i[0] for i in sorted_list]
    return top_features

def find_vocab_corpus(training_data_dict):
    pos_dec = training_data_dict['positive_deceptive']
    neg_dec = training_data_dict['negative_deceptive']
    pos_tru = training_data_dict['positive_truthful']
    neg_tru = training_data_dict['negative_truthful']
    complete_str = pos_dec + neg_dec + pos_tru + neg_tru
    word_count_comp_str = word_count(complete_str)
    return word_count_comp_str


def feature_selection2(word_dict, training_data_list_dict, no_of_features):
    total_docs = len(training_data_list_dict['positive_deceptive'])+len(training_data_list_dict['negative_deceptive'])+len(training_data_list_dict['positive_truthful'])+len(training_data_list_dict['negative_truthful'])
    total_words = len(word_dict)
    word_list = list(word_dict.keys())
    document_matrix = np.zeros(shape=(total_docs, total_words))
    for key, val in training_data_list_dict.items():
        list_of_doc = val
        for i in range(len(list_of_doc)):
            string_list = list_of_doc[i].split(' ')
            for word in string_list:
                document_matrix[i][word_list.index(word)]+=1
            document_matrix[i] = np.divide(document_matrix[i], len(string_list))
    count_matrix = np.where(document_matrix>0, 1,0)
    idf_matrix = np.sum(count_matrix, axis=0)
    idf_matrix = np.divide(math.log10(total_docs), idf_matrix)
    idf_matrix1=np.array([idf_matrix]*document_matrix.shape[0])
    tf_idf_mat=np.multiply(document_matrix, idf_matrix1)
    mean_tf_idf = np.mean(tf_idf_mat, axis=0)
    mean_tf_idf_list = mean_tf_idf.tolist()
    mean_tf_idf_dict = dict(zip(word_list, mean_tf_idf_list))
    sorted_list = sorted(mean_tf_idf_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_list = sorted_list[0:no_of_features]
    top_features = [i[0] for i in sorted_list]
    return top_features

def compute_feature_vector(feature_list, document):
    doc_words = word_count(document)
    feature_vector = np.zeros(len(feature_list))
    i=0
    for word in feature_list:
        if doc_words.get(word) is not None:
            feature_vector[i] = doc_words.get(word)
        i+=1
    return feature_vector

def compute_input_feature_vector(label1, label2, no_of_features, top_features, word_class_list):
    X_pos_neg = np.zeros(shape=(1, no_of_features))
    label = label1
    for j in range(0, 2):
        for i in range(len(word_class_list[label])):
            feature_vector = compute_feature_vector(top_features, word_class_list[label][i])
            feature_vector = feature_vector.reshape((1, no_of_features))
            X_pos_neg = np.concatenate((X_pos_neg, feature_vector), axis=0)
        label = label2
    X_pos_neg = X_pos_neg[1:, :]
    return X_pos_neg

def train_vanilla_perceptron(no_of_features, no_of_iterations, X, y):
    X=np.where(X>0,1,0)
    #initialize weights
    weight = np.zeros(shape=(no_of_features, ))
    bias = 0
    random.seed(50)
    for i in range(no_of_iterations):
        num = random.randint(0, len(X)-1)
        activation = np.dot(weight, X[num]) + bias
        if y[num] * activation <= 0:
            weight = np.sum((weight, y[num]*X[num]), axis=0)
            bias = bias + y[num]
    return weight, bias

def train_average_perceptron(no_of_features, no_of_iterations, X, y):
    X = np.where(X > 0, 1, 0)
    #initialize weights
    weight = np.zeros(shape=(no_of_features, ))
    cached_weight = np.zeros(shape=(no_of_features, ))
    bias = 0
    cached_bias = 0
    count = 1
    random.seed(50)
    for i in range(no_of_iterations):
        num = random.randint(0, len(X)-1)
        activation = np.dot(weight, X[num]) + bias
        if y[num] * activation <= 0:
            weight = np.sum((weight, y[num]*X[num]), axis=0)
            cached_weight = np.sum((cached_weight, count*y[num]*X[num]), axis=0)
            bias = bias + y[num]
            cached_bias = cached_bias + y[num]*count
        count+=1
    inverse_count = 1/count
    return np.subtract(weight, inverse_count*cached_weight), bias-(inverse_count*cached_bias)

def build_master_dict(label_1, label_2, bias_average_1, bias_average_2, weight_average_1, weight_average_2, top_features_1, top_features_2):
    master_dict = dict()
    master_dict[label_1 + '_bias'] = float(bias_average_1[0])
    master_dict[label_2 + '_bias'] = float(bias_average_2[0])
    top_features_val_1 = weight_average_1.tolist()
    top_features_dict_1 = dict(zip(top_features_1, top_features_val_1))
    master_dict[label_1] = top_features_dict_1
    top_features_val_2 = weight_average_2.tolist()
    top_features_dict_2 = dict(zip(top_features_2, top_features_val_2))
    master_dict[label_2] = top_features_dict_2
    return master_dict

def remove_low_freq_word(x_dict):
    word_dict = x_dict.copy()
    for word in x_dict:
        if x_dict[word] < 5:
            word_dict.pop(word)
    return word_dict

if __name__ == "__main__":
    model_file = "vanillamodel.txt"
    avg_model_file = "averagemodel.txt"

    input_path = str(sys.argv[1])
    start = time.time()
    label_1 = 'pos_neg'
    label_2 = 'tru_dec'
    no_of_features_pos_neg = 800
    no_of_features_tru_dec = 1200
    no_of_iterations_vanilla = 100*800
    no_of_iterations_average = 600*800

    training_data_dict, training_data_list_dict = read_training_data(input_path)
    word_class = compute_word_class(training_data_dict)

    word_dict_pos, word_dict_tru = find_vocabulary(word_class)
    top_features_pos_neg = feature_selection(word_dict_pos, word_class, label_1, no_of_features_pos_neg)
    top_features_tru_dec = feature_selection(word_dict_tru, word_class, label_2, no_of_features_tru_dec)
    #top_features_tru_dec = feature_selection(word_dict_pos, word_class, label_1, no_of_features_tru_dec)
    # top_features_tru_dec= ['ability', 'absolute', 'absolutely', 'ac', 'accessible', 'accommodating', 'accommodations', 'accomodating', 'accomodations', 'activities', 'activity', 'actual', 'addition', 'address', 'adequate', 'affina', 'affinia', 'afraid', 'age', 'allergies', 'alley', 'alone', 'alot', 'amazed', 'amazing', 'ambassador', 'amenity', 'ammenities', 'amount', 'annoying', 'answered', 'anybody', 'anytime', 'apart', 'apartment', 'apple', 'appliances', 'appointed', 'approached', 'aquarium', 'architecture', 'arent', 'aria', 'arrive', 'aside', 'ask', 'asleep', 'aspect', 'assume', 'attempted', 'attendant', 'attending', 'attention', 'attentive', 'avoid', 'awesome', 'awful', 'bag', 'band', 'bare', 'bartender', 'base', 'basement', 'basic', 'bathrobe', 'bathrobes', 'bathrooms', 'beach', 'beat', 'beatles', 'beautifully', 'bedbugs', 'beer', 'begin', 'beginning', 'bellboy', 'bellhop', 'bellman', 'bellmen', 'belongings', 'beside', 'beverage', 'beverages', 'beware', 'bigger', 'billed', 'birthday', 'bites', 'blankets', 'blew', 'block', 'blocks', 'board', 'bonus', 'booking', 'boring', 'bothered', 'bottom', 'boutique', 'bowl', 'boy', 'boys', 'brand', 'break', 'breathtaking', 'bridge', 'bring', 'bringing', 'brought', 'brown', 'bucks', 'budget', 'bug', 'bugs', 'bulb', 'burger', 'bus', 'cab', 'cabinet', 'cabs', 'cafe', 'cake', 'calls', 'cancel', 'canceled', 'cancellation', 'cannot', 'cap', 'cart', 'caters', 'ceiling', 'central', 'centre', 'chain', 'charm', 'charming', 'cheapest', 'chic', 'chicken', 'children', 'china', 'chocolate', 'choices', 'choose', 'choosing', 'chop', 'christmas', 'cities', 'cleanest', 'cleanliness', 'clerks', 'client', 'clients', 'clock', 'clothes', 'coast', 'cocktail', 'color', 'colors', 'comes', 'comfort', 'comfy', 'compared', 'compelled', 'complain', 'complaint', 'concern', 'concerned', 'condition', 'conditioner', 'conference', 'confirmed', 'connected', 'connection', 'conrad', 'consider', 'consisted', 'constantly', 'construction', 'contact', 'convenience', 'conveniently', 'conversation', 'correctly', 'costs', 'couch', 'couple', 'courteous', 'cover', 'covers', 'cozy', 'cramped', 'crappy', 'crew', 'crowded', 'culture', 'cups', 'current', 'curtains', 'date', 'dated', 'dealt', 'deco', 'decorations', 'definately', 'definetly', 'delayed', 'delightful', 'deluxe', 'designed', 'desired', 'die', 'difference', 'difficult', 'dine', 'directions', 'disappointing', 'dishes', 'dishwasher', 'display', 'dissatisfied', 'district', 'dock', 'docking', 'dog', 'doll', 'dollars', 'dozen', 'drake', 'dream', 'drip', 'dripped', 'drivers', 'dump', 'earned', 'east', 'eating', 'elegance', 'elegant', 'emergency', 'employees', 'empty', 'encountered', 'enjoyable', 'enjoyed', 'enter', 'entering', 'entertainment', 'equipped', 'establishment', 'europe', 'events', 'everybody', 'evident', 'exceptional', 'excited', 'exciting', 'exercise', 'exhausted', 'exorbitant', 'experienced', 'express', 'fabulous', 'face', 'facility', 'facing', 'fair', 'falling', 'falls', 'fan', 'fault', 'feature', 'feeling', 'feels', 'fees', 'festival', 'field', 'figured', 'filled', 'fit', 'fitzpatrick', 'fixtures', 'flatscreen', 'flowers', 'folks', 'foods', 'forget', 'forgot', 'form', 'frankly', 'freezing', 'fresh', 'friend', 'friendliness', 'friends', 'fruit', 'frustrated', 'frustrating', 'function', 'funny', 'furnished', 'garbage', 'giant', 'ginos', 'girl', 'girls', 'glad', 'gladly', 'glasses', 'god', 'goes', 'gold', 'gorgeous', 'gouge', 'grand', 'grant', 'granted', 'greet', 'grill', 'grocery', 'ground', 'guide', 'guys', 'gym', 'hairs', 'hall', 'hancock', 'handicapped', 'handle', 'hang', 'hardly', 'health', 'heart', 'heartbeat', 'heated', 'heavy', 'hello', 'helped', 'helping', 'heres', 'hi', 'hip', 'historic', 'history', 'hole', 'hop', 'hospital', 'hotwire', 'hundreds', 'ideal', 'imagine', 'immaculate', 'important', 'importantly', 'impression', 'impressions', 'impressive', 'inches', 'includes', 'incredible', 'indeed', 'indoor', 'inroom', 'insulation', 'intended', 'interesting', 'inviting', 'ipod', 'iron', 'isnt', 'john', 'joke', 'junior', 'kids', 'kitchen', 'knowing', 'knowledgeable', 'knows', 'larger', 'lead', 'leaked', 'lesser', 'lifetime', 'lift', 'linens', 'lit', 'lite', 'live', 'locations', 'lodging', 'logo', 'looks', 'lost', 'loud', 'lounge', 'love', 'loved', 'lower', 'lucky', 'lunch', 'magnificant', 'maid', 'maids', 'maintained', 'makes', 'mall', 'managed', 'manhattan', 'manner', 'marble', 'marks', 'married', 'martini', 'massages', 'mattresses', 'meals', 'means', 'meantime', 'mediocre', 'meeting', 'members', 'memorabilia', 'memorial', 'men', 'menu', 'mess', 'mignon', 'mildew', 'miles', 'millenium', 'min', 'mine', 'mini', 'minimum', 'mins', 'miss', 'moment', 'month', 'months', 'mostly', 'move', 'museum', 'museums', 'n', 'name', 'narrow', 'navy', 'needs', 'negative', 'neighbor', 'neighborhood', 'neighbors', 'nervous', 'nicely', 'nicer', 'nicest', 'noises', 'non', 'none', 'notch', 'nowhere', 'number', 'numerous', 'occasions', 'offers', 'often', 'ohare', 'older', 'olds', 'onsite', 'opinion', 'option', 'options', 'ordered', 'organized', 'others', 'ouch', 'overcharged', 'overlooking', 'paint', 'pair', 'palm', 'palms', 'pampered', 'pantry', 'parked', 'parks', 'partial', 'particular', 'passing', 'path', 'penny', 'perfect', 'perfectly', 'personnel', 'picked', 'picky', 'pier', 'pillow', 'pizza', 'planned', 'planning', 'plasma', 'plastic', 'pleasantly', 'please', 'pleased', 'pleasing', 'pleasure', 'pocket', 'policy', 'polite', 'positives', 'possibly', 'pot', 'prefer', 'premises', 'premium', 'prepaid', 'prepared', 'presidential', 'priced', 'priceline', 'prices', 'pricey', 'printed', 'produced', 'professional', 'promise', 'promptly', 'properly', 'property', 'pros', 'provide', 'provided', 'provides', 'pulled', 'pump', 'queen', 'quick', 'quiet', 'quieted', 'radio', 'rang', 'range', 'ratings', 'rd', 'reach', 'real', 'realized', 'reasonable', 'reccomend', 'receptionist', 'receptionists', 'recommended', 'redeeming', 'reeked', 'refreshing', 'refund', 'regardless', 'regency', 'regret', 'regular', 'relax', 'relaxed', 'relaxing', 'relieved', 'remember', 'remembered', 'reminded', 'remodeled', 'remote', 'renovated', 'renovations', 'repair', 'replaced', 'replenished', 'representative', 'requests', 'resist', 'resort', 'rest', 'rested', 'restroom', 'returning', 'rex', 'ring', 'roaches', 'rock', 'romance', 'romantic', 'royalty', 'rudely', 'rushed', 'safe', 'sales', 'satisfied', 'saved', 'says', 'scared', 'school', 'scum', 'seafood', 'sears', 'seasons', 'seat', 'secondly', 'section', 'security', 'seeing', 'seem', 'seems', 'select', 'self', 'sense', 'separate', 'served', 'servers', 'setting', 'shape', 'sharing', 'shedd', 'shoes', 'shop', 'shops', 'shows', 'shut', 'sights', 'sightseeing', 'single', 'sirens', 'sister', 'sit', 'site', 'sites', 'situated', 'sized', 'sleeper', 'slept', 'smile', 'smoking', 'snack', 'snacks', 'soda', 'soft', 'softest', 'sold', 'solid', 'somewhat', 'son', 'soon', 'sorts', 'sounded', 'spa', 'speaking', 'spectacular', 'spend', 'sports', 'spots', 'spray', 'spread', 'sq', 'stained', 'stains', 'stand', 'standard', 'standards', 'starbucks', 'stars', 'starwood', 'statement', 'station', 'stays', 'steak', 'steakhouse', 'steaks', 'step', 'stepped', 'stocked', 'stood', 'stops', 'style', 'suburbs', 'subway', 'suggested', 'suggestions', 'summer', 'super', 'superb', 'superior', 'suppose', 'surprise', 'surrounded', 'sushi', 'sweet', 'takes', 'talbot', 'talbott', 'taxes', 'taxi', 'tea', 'team', 'tech', 'teenage', 'teenager', 'televisions', 'telling', 'temp', 'terrace', 'terrible', 'thankfully', 'thanks', 'thanksgiving', 'thermostat', 'thier', 'thin', 'thoroughly', 'thumbs', 'tim', 'tip', 'together', 'toiletries', 'touches', 'tour', 'towards', 'towel', 'tower', 'towers', 'traditional', 'train', 'trained', 'training', 'transportation', 'trash', 'traveler', 'traveling', 'travelling', 'treadmill', 'treat', 'trendy', 'tribune', 'triple', 'turns', 'unable', 'underground', 'understaffed', 'understand', 'unfriendly', 'unhappy', 'unique', 'unlike', 'upcharge', 'upgraded', 'upper', 'upscale', 'upset', 'usual', 'value', 'variety', 'vegas', 'vending', 'vent', 'views', 'visits', 'voucher', 'vouchers', 'waiter', 'walkin', 'wanting', 'warm', 'warmly', 'waste', 'watch', 'web', 'wedding', 'week', 'weekends', 'welcomed', 'westin', 'whatsoever', 'whenever', 'whether', 'whilst', 'white', 'willing', 'windy', 'winter', 'wish', 'wished', 'wonder', 'wonderful', 'word', 'workout', 'world', 'worried', 'wow', 'writing', 'yet', 'york', 'young']

    # word_dict = find_vocab_corpus(training_data_dict)
    # #word_dict = remove_low_freq_word(word_dict)
    # top_features_pos_neg = feature_selection2(word_dict, training_data_list_dict, no_of_features)
    # top_features_tru_dec = top_features_pos_neg

    word_class_list = compute_word_class(training_data_list_dict)
    X_pos_neg = compute_input_feature_vector('positive', 'negative', no_of_features_pos_neg, top_features_pos_neg, word_class_list)
    X_tru_dec = compute_input_feature_vector('truthful', 'deceptive', no_of_features_tru_dec, top_features_tru_dec, word_class_list)

    length = len(X_pos_neg)
    y = list()
    y[:length] = [1] * length
    y = np.array(y)
    index = int(length/2)
    y[index:length] = -1
    y = y.reshape((length, 1))
    #run vanilla perceptron
    weight_vanilla_pos_neg, bias_vanilla_pos_neg = train_vanilla_perceptron(no_of_features_pos_neg, no_of_iterations_vanilla, X_pos_neg, y)
    weight_vanilla_tru_dec, bias_vanilla_tru_dec = train_vanilla_perceptron(no_of_features_tru_dec, no_of_iterations_vanilla, X_tru_dec, y)
    master_dict_vanilla = build_master_dict(label_1, label_2, bias_vanilla_pos_neg, bias_vanilla_tru_dec, weight_vanilla_pos_neg, weight_vanilla_tru_dec, top_features_pos_neg, top_features_tru_dec)
    json_vanilla = json.dumps(master_dict_vanilla, indent=2)
    f1 = open(model_file, "w")
    f1.write(json_vanilla)
    f1.close()

    #run average perceptron
    weight_average_pos_neg, bias_average_pos_neg = train_average_perceptron(no_of_features_pos_neg, no_of_iterations_average, X_pos_neg, y)
    weight_average_tru_dec, bias_average_tru_dec = train_average_perceptron(no_of_features_tru_dec, no_of_iterations_average, X_tru_dec, y)
    master_dict_average = build_master_dict(label_1, label_2, bias_average_pos_neg, bias_average_tru_dec, weight_average_pos_neg, weight_average_tru_dec, top_features_tru_dec, top_features_tru_dec)
    json_average = json.dumps(master_dict_average, indent=2)
    f2 = open(avg_model_file, "w")
    f2.write(json_average)
    f2.close()
    print(time.time()-start)
