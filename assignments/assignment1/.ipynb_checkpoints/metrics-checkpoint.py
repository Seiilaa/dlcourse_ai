def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = 0 # true positives
    fp = 0 # false positives
    fn = 0 # false negatives
    tn = 0 # true negatives
    
    for i in range(prediction.shape[0]):
        if prediction[i] == ground_truth[i] and ground_truth[i] == True:
            tp+=1
        elif prediction[i] != ground_truth[i] and ground_truth[i] == True:
            fn+=1
        elif prediction[i] != ground_truth[i] and ground_truth[i] == False:
            fp+=1
        else:
            tn+=1
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*precision*recall/(precision+recall)
    accuracy = (tp + tn) / (tp+fp+tn+fn)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    true_predicts = 0
    for i in range(prediction.shape[0]):
        if prediction[i] == ground_truth[i]:
            true_predicts += 1
    accuracy = true_predicts / prediction.shape[0]
    return accuracy
