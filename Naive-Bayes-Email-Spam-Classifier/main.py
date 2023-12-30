from collections import defaultdict
import os , string
from math import log

def parse_emails(directory_path):
    X = []
    Y = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)

            label = 0 if filename.startswith("HAM.") else 1

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                email_text = file.read()

            X.append(email_text)
            Y.append(label)

    return X, Y


def nb_train(x, y):
    ham_count = 0
    spam_count = 0
    ham_fd = defaultdict(int)
    spam_fd = defaultdict(int)

    for doc, label in zip(x, y):
        words = doc.split()
        if label == 0:  # HAM
            ham_count += 1
            for word in words:
                ham_fd[word.lower().translate(str.maketrans("", "", string.punctuation))] += 1
        elif label == 1:  # SPAM
            spam_count += 1
            for word in words:
                spam_fd[word.lower().translate(str.maketrans("", "", string.punctuation))] += 1

    model = {
        'ham_count': ham_count,
        'spam_count': spam_count,
        'ham_fd': dict(ham_fd),
        'spam_fd': dict(spam_fd)
    }

    return model


def nb_test(docs, model, use_log, smoothing):
    predictions = []
    smoothing1 = 1e-15 if smoothing == False else 1
    HAM_fd = sum(model['ham_fd'].values()) 
    SPAM_fd = sum(model['spam_fd'].values()) 
    vocabulary_size1 = len(set(model['ham_fd']).union(set(model['spam_fd'])))
    vocabulary_size = 0 if smoothing == False else vocabulary_size1

    for doc in docs:
        words = doc.split()
        ham_prob = 1 if use_log == False else 0
        spam_prob = 1 if use_log == False else 0
        
        for word in words:
            if use_log:
                ham_prob += log((model['ham_fd'].get(word.lower().translate(str.maketrans("", "", string.punctuation)), 0) + smoothing1) / (HAM_fd + vocabulary_size))
                spam_prob += log((model['spam_fd'].get(word.lower().translate(str.maketrans("", "", string.punctuation)), 0) + smoothing1) / (SPAM_fd + vocabulary_size))
            else:
                ham_prob *= (model['ham_fd'].get(word.lower().translate(str.maketrans("", "", string.punctuation)), 0)+ smoothing1) / (HAM_fd +  vocabulary_size)
                spam_prob *= (model['spam_fd'].get(word.lower().translate(str.maketrans("", "", string.punctuation)), 0)+ smoothing1) /(SPAM_fd +  vocabulary_size)
        ham_prob = (model['ham_count'] / (model['ham_count'] + model['spam_count'])) * ham_prob if use_log == False else log(model['ham_count'] / (model['ham_count'] + model['spam_count'])) + ham_prob
        spam_prob = (model['spam_count'] / (model['ham_count'] + model['spam_count'])) * spam_prob if use_log == False else log(model['spam_count'] / (model['ham_count'] + model['spam_count'])) + spam_prob
       
        prediction = 0 if ham_prob > spam_prob else 1
        predictions.append(prediction)
    return predictions

def f_score(y_true, y_pred):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    predicted_positives = sum(y_pred)
    actual_positives = sum(y_true)
    
    precision = true_positives / (predicted_positives + 1e-15)  
    recall = true_positives / (actual_positives + 1e-15)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)  

    return f1

def run_experiment(train_path, test_path):

    train_X, train_Y = parse_emails(train_path)
    trained_model = nb_train(train_X, train_Y)

    test_X, test_Y_true = parse_emails(test_path)
    configurations = [
        {'use_log': False, 'smoothing': False},
        {'use_log': False, 'smoothing': True},
        {'use_log': True, 'smoothing': False},
        {'use_log': True, 'smoothing': True}
    ]

    for config in configurations:
        test_Y_pred = nb_test(test_X, trained_model, use_log=config['use_log'], smoothing=config['smoothing'])
        f1 = f_score(test_Y_true, test_Y_pred)
        print(f"F1-Score (Log={config['use_log']}, Smoothing={config['smoothing']}): {f1}")


train_directory = "SPAM_training_set"
test_directory = "SPAM_test_set"
run_experiment(train_directory, test_directory)