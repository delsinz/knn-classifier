""" Remember to add our names, student IDs here """

import csv
from collections import Counter, defaultdict
from random import shuffle


"""
    Probably don't need a main in the final submission
"""
def main():
    # Data set which is a two tuple.
    data_set = preprocess_data('data.data', 3)
    partition_data(data_set)
    ''''
    instance = data_set[0][2156]
    neighbors = get_neighbors(instance, data_set, 1000, 'cos')
    print(predict_class(neighbors, 'ew'))
    print(predict_class(neighbors, 'ild'))
    print(predict_class(neighbors, 'id'))
    print(instance)
    print(partition_data(([1,2,2,2,3,3,4,5,5],[1])))'''


""" 
    preprocess_data in the specification takes only one argument 
    but our preprocess_data takes two arguments, the second one
    specifying the dataset we are going to be dealing with on
    a particular run. 
    Value of parameter abalone = 2 means we are dealing with
    abalone - 2, abalone = 3 means we are dealing with abalone -3. 
"""
# Returns ([lists of 8 attributes], [class_label])
def preprocess_data(filename, abalone = 2):
    # Load data
    with open(filename) as file:
        reader = csv.reader(file)

        # Construct instance list
        instances = []
        to_be_predicted = []
        for row in reader:
            instance = []
            for attribute in row:
                try:
                    instance.append(float(attribute))
                except ValueError:
                    instance.append(attribute)
            instances.append(instance[:len(instance) - 1])
            to_be_predicted.append(instance[len(instance) - 1])

        # Construct class labels
        class_labels = []
        if abalone == 3:
            for rings in to_be_predicted:
                label = ''
                if rings <= 8:
                    label = 'very-young'
                elif rings <= 10:
                    label = 'middle-age'
                else:
                    label = 'old'
                class_labels.append(label)
        elif abalone == 2:
            for rings in to_be_predicted:
                label = ''
                if rings <= 10:
                    label = 'young'
                else:
                    label = 'old'
                class_labels.append(label)
        else:
            return None

    # Construct data set
    data_set = (instances, class_labels)
    return data_set



def compare_instance(instance_0, instance_1, method = 'euclidean'):
    instance_0 = convert_categorical_attribute(instance_0)
    instance_1 = convert_categorical_attribute(instance_1)

    if method == 'euclidean':
        return euclidean_dist(instance_0, instance_1)
    elif method == 'cos':
        return cos_dist(instance_0, instance_1)
    elif method == 'manhattan':
        return manhattan_dist(instance_0, instance_1)
    else:
        return None



def get_neighbors(instance, training_data_set, k, method = 'euclidean'):
    # Get class labels and scores
    training_instances = training_data_set[0]
    class_labels = training_data_set[1]
    size = len(training_instances)
    scores = []
    for i in range(size):
        scores.append((class_labels[i], compare_instance(instance, training_instances[i], method)))
    # Sort result
    sorted_scores = sorted_scores = sorted(scores, key=lambda x:x[1])
    return sorted_scores[:k]


# Not necessarily this many metrics, but what the hell. Just pick a few maybe?
def evaluation(data_set, metric='accuracy'):
    if metric == 'accuracy':
        return 0
    elif metric == 'error': # Error rate
        return 0
    elif metric == 'precision':
        return 0
    elif metric == 'recall':
        return 0
    elif metric == 'specificity':
        return 0
    else:
        return None


def partition_data(data_set):
    partitioned_sets = []
    M = 10
    set_size = len(data_set[0])
    partition_size = set_size // M
    set_size_divider = set_size % M

    '''# List of random indices
    random_index = list(range(set_size))
    shuffle(random_index)'''

    data_list = [list(instance) for instance in zip(data_set[0], data_set[1])]
    shuffle(data_list)
    instances = []
    class_labels = []
    for row in data_list:

        instances.append(row[:len(row) - 1])
        class_labels.append(row[len(row)- 1])

    data_set = (instances, class_labels)


    start = 0
    for i in range(10):
        if i < set_size_divider:

            partitioned_sets.append((data_set[0][start:(start + partition_size + 1)],
            data_set[1][start:(start + partition_size + 1)]))
            start += partition_size + 1
        else:

            partitioned_sets.append((data_set[0][start:(start + partition_size)],
            data_set[1][start:(start + partition_size)]))
            start += partition_size

    '''for (me, you) in partitioned_sets:
        print((me, you))
        print("\n")'''

    return partitioned_sets




def predict_class(neighbors, method = 'ew'):
    if method == 'ew':
        return predict_equal_weight(neighbors)
    elif method == 'ild':
        return predict_inverse_linear_dist(neighbors)
    elif method == 'id':
        return predict_inverse_dist(neighbors)
    else:
        return None


def predict_equal_weight(neighbors):
    labels = [neighbor[0] for neighbor in neighbors]
    label_votes = dict(Counter(labels))
    return max(label_votes, key=label_votes.get)



def predict_inverse_linear_dist(neighbors):
    max_dist = max([neighbor[1] for neighbor in neighbors])
    min_dist = min([neighbor[1] for neighbor in neighbors])
    label_votes = defaultdict(float)
    for neighbor in neighbors:
        weight = 0
        if max_dist == min_dist:
            weight = 1
        else:
            weight = (max_dist - neighbor[1]) / (max_dist - min_dist)
        label_votes[neighbor[0]] += weight
    return max(label_votes, key=label_votes.get)



def predict_inverse_dist(neighbors):
    label_votes = defaultdict(float)
    offset = 2 # Offset > 1 so that weight for cos dis won't be negative due to the way cos dist is handled
    for neighbor in neighbors:
        weight = 1 / (neighbor[1] + offset)
        label_votes[neighbor[0]] += weight
    return max(label_votes, key=label_votes.get)



def euclidean_dist(instance_0, instance_1):
    # Assuming both instances have the same num of attributes
    length = len(instance_0)
    square_sum = 0
    for i in range(length):
        square_sum += (instance_0[i] - instance_1[i]) ** 2
    return square_sum ** 0.5



def accuracy(test_set, predicted_classes, class_name):

    length = len(test_set)
    correct_predictions = 0

    for i in length:
        if test_set[1][i] == class_name and test_set[1][i] == predicted_classes[i]:
            correct_predictions += 1
        
        elif test_set[1][i] != class_name and predicted_classes[i] != class_name:
            correct_predictions += 1
           
    return correct_predictions/length*100



def complete_accuracy(test_set, predicted_classes):
    
    classes = list(set(test_set[1]))
    sum_accuracy = 0
    for class_name in classes:
        sum_accuracy += accuracy(test_set, predicted_classes, class_name)

    return sum_accuracy/3



def precision(test_set, predicted_classes, class_name):
    
    length = len(test_set)
    true_positives = 0
    false_positives = 0

    for i in length:
        if test_set[1][i] == class_name and predicted_classes[i] == class_name:
            true_positives += 1
        elif test_set[1][i] != class_name  and predicted_classes[i] == class_name:
            false_positives += 1
    
    return true_positives/(true_positives + false_positives)


 
def complete_precision(test_set, predicted_classes):
    
    classes = list(set(test_set[1]))
    sum_precision = 0
    for class_name in classes:
        sum_precision += precision(test_set, predicted_classes, class_name)
    
    return sum_precision/3



def recall(test_set, predicted_classes, class_name):

    length = len(test_set)
    true_positives = 0
    false_negatives = 0

    for i in length:
        if test_set[1][i] == class_name and predicted_classes[i] == class_name:
            true_positives += 1
        elif test_set[1][i] == class_name  and predicted_classes[i] != class_name:
            false_negatives += 1
    
    return true_positives/(true_positives + false_negatives)



def total_recall(test_set, predicted_classes):

    classes = list(set(test_set[1]))
    sum_recall = 0 
    for class_name in classes:
        sum_recall += recall(test_set, predicted_classes, class_name)

    return sum_recall/3
    


def complete_error(test_set, predicted_classes):

    error = 100 - complete_accuracy(test_set, predicted_classes)

    return error


def cos_dist(instance_0, instance_1):
    mag_0 = 0
    mag_1 = 0
    dot_prod = 0
    for i in range(len(instance_0)):
        mag_0 += instance_0[i] ** 2
        mag_1 += instance_1[i] ** 2
        dot_prod += instance_0[i] * instance_1[i]
    mag_0 = mag_0 ** 0.5
    mag_1 = mag_1 ** 0.5
    #I feel like this should be dot_prod = 0
    if dot_prod == 0: #Orthogonal
        return 0
    else:
        # Give cos dist the same behavior (smaller == better) as euclidean and manhattan
        # similarity -> distance, thus the negation.
        return 1 - dot_prod / (mag_0 * mag_1)



def manhattan_dist(instance_0, instance_1):
    total = 0
    for i in range(len(instance_0)):
        total += abs(instance_0[i] - instance_1[i])
    return total



# Break categorical attribute Sex into 3 binary attributes: M, F, I
def convert_categorical_attribute(instance):
    # Assuming the attributes are provided in the given order
    val = instance[0]
    if val == 'M':
        return [1, 0, 0] + instance[1:]
    elif val == 'F':
        return [0, 1, 0] + instance[1:]
    elif val == 'I':
        return [0, 0, 1] + instance[1:]
    else:
        return [0, 0, 0] + instance[1:]


if __name__ == '__main__':
    main()
