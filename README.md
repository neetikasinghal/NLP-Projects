# NLP-Projects
All the project work done during coursework of Natural Language Processing during Masters at USC

## Naive Bayes Classifier
1. A Naive Bayes classifier to identify hotel reviews as either truthful or deceptive, and either positive or negative
with no external python libraries used.
2. Run nbclassify3.py and give input data as command line argument. It will create nbmodel file with words as features
and corresponding numbers giving probabilities.
3. Run nbclassify giving nbmodel and test data as input.
4. Got 90% accuracy on the data given.

## Perceptron Classifier
1. A Perceptron classifier(both vanilla model and average model) to identify hotel reviews as either truthful or deceptive, and either positive or negative
with no external python libraries used.
2. Run perceplearn.py and give input data as command line argument. It will create vanilla model file and average model file with words as features.
3. Run percepclassify giving vanilla/average model file path as first argument and test data path as second argument.

## Hidden Markov Model Part-of-Speech Tagger(HMM-POS Tagger)
1. A Hidden Markov Model part-of-speech tagger for Italian, Japanese, and a surprise language. The training data are provided tokenized and tagged.
2. Run hmmlearn3.py with giving input data as command line argument. It will create hmmmodel.txt with values of transition as well as emission states
3. Run hmmdecode3.py which takes path of model file as input and using viterbi decoding gets the most suitable tagger and produces the output.txt which contains tagged test data.
4. It has given the accuracy of 93% for japaneese, 91% for italian and 92% for surprise language.