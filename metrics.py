'''
Implement recall, f1, precision
'''

from keras import backend as K


def recall_score(y_val, y_pred):
    '''
    Implement recall score based on formula
    '''
    true_positives = K.sum(K.round(K.clip(y_val * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_val, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_val, y_pred):
    '''
    Implement precision based on formula
    '''
    true_positives = K.sum(K.round(K.clip(y_val * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (possible_positives + K.epsilon())
    return precision


def f1_score(y_val, y_pred):
    '''
    Implement f1_score based on formula
    '''
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
