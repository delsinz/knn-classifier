""" Remember to add our names, student IDs here """

import csv
from collections import Counter, defaultdict
from random import shuffle
from scipy.spatial import distance


"""
    Probably don't need a main in the final submission
"""
def main():
    # Data set which is a two tuple.
    data_set = preprocess_data('data.data', 3)

    ''''
    instance = data_set[0][2156]
    neighbors = get_neighbors(instance, data_set, 1000, 'cos')
    print(predict_class(neighbors, 'ew'))
    print(predict_class(neighbors, 'ild'))
    print(predict_class(neighbors, 'id'))
    print(instance)
    print(partition_data(([1,2,2,2,3,3,4,5,5],[1])))'''
    #print(len(data_set[0]))
    print(evaluation(data_set, metric='precision'))





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



def compare_instance(instance_0, instance_1, method):
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



def get_neighbors(instance, training_data_set, k, method):
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



'''
data_set: 2-tuple. ([list of instances], [list of class labels])
metric: accuracy || recall || precision || error
dist: euclidean || cos || manhattan
k: positive int
voting: ew || ild || id
'''
def evaluation(data_set, metric='accuracy', dist='euclidean', k=200, voting='ew'):
    score = 0
    partitioned_sets = partition_data(data_set)
    for i in range(len(partitioned_sets)):
        testing = partitioned_sets[i]
        training = combine_data_sets(partitioned_sets[:i], partitioned_sets[i+1:])
        score += single_pass_eval(training, testing, metric, dist, k, voting)
    return score / len(partitioned_sets)



# Combine two lists of data sets into one set
def combine_data_sets(list0, list1):
    instances = []
    class_labels = []
    for test_set in list0:
        for instance in test_set[0]:
            instances.append(instance)
        for label in test_set[1]:
            class_labels.append(label)
    for test_set in list1:
        for instance in test_set[0]:
            instances.append(instance)
        for label in test_set[1]:
            class_labels.append(label)
    return (instances, class_labels)



def single_pass_eval(training_set, test_set, metric, dist, k, voting):
    predicted_classes = []
    for instance in test_set[0]:
        neighbors = get_neighbors(instance, training_set, k, dist)
        predicted_classes.append(predict_class(neighbors, voting))
    if metric == 'accuracy':
        return complete_accuracy(test_set, predicted_classes)
    elif metric == 'recall':
        return complete_recall(test_set, predicted_classes)
    elif metric == 'precision':
        return complete_precision(test_set, predicted_classes)
    elif metric == 'error':
        return complete_error(test_set, predicted_classes)
    else:
        return None





def partition_data(data_set):
    partitioned_sets = []
    M = 10
    set_size = len(data_set[0])
    partition_size = set_size // M
    set_size_divider = set_size % M

    # Break down the data set
    data_list = [data_set[0][i]+[data_set[1][i]] for i in range(set_size)]
    # Random orderring
    shuffle(data_list)
    # Construct new randomly ordered data set
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

    return partitioned_sets




def predict_class(neighbors, method):
    if method == 'ew': # Equal weight
        return predict_equal_weight(neighbors)
    elif method == 'ild': # Inverse linear distance
        return predict_inverse_linear_dist(neighbors)
    elif method == 'id': # Inverse distance
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
    offset = 0.5 # Offset > 1 so that weight for cos dis won't be negative due to the way cos dist is handled
    for neighbor in neighbors:
        weight = 1 / (neighbor[1] + offset)
        label_votes[neighbor[0]] += weight
    return max(label_votes, key=label_votes.get)


def accuracy(test_set, predicted_classes, class_name):

    length = len(test_set)
    correct_predictions = 0

    for i in range(length):
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

    return sum_accuracy/len(classes)



def precision(test_set, predicted_classes, class_name):
    length = len(test_set)
    true_positives = 0
    false_positives = 0

    for i in range(length):
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

    return sum_precision/len(classes)



def recall(test_set, predicted_classes, class_name):

    length = len(test_set)
    true_positives = 0
    false_negatives = 0

    for i in range(length):
        if test_set[1][i] == class_name and predicted_classes[i] == class_name:
            true_positives += 1
        elif test_set[1][i] == class_name  and predicted_classes[i] != class_name:
            false_negatives += 1

    return true_positives/(true_positives + false_negatives)



def complete_recall(test_set, predicted_classes):

    classes = list(set(test_set[1]))
    sum_recall = 0
    for class_name in classes:
        sum_recall += recall(test_set, predicted_classes, class_name)

    return sum_recall/len(classes)



def complete_error(test_set, predicted_classes):

    error = 100 - complete_accuracy(test_set, predicted_classes)

    return error


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


# Tested.
def euclidean_dist(instance_0, instance_1):
    # Assuming both instances have the same num of attributes
    length = len(instance_0)
    square_sum = 0
    for i in range(length):
        square_sum += (instance_0[i] - instance_1[i]) ** 2
    return square_sum ** 0.5

def test_euclidean(data_set):
    count = 0
    # Test for euclidean correctness.
    for row1 in data_set[0]:
        row1 = convert_categorical_attribute(row1)
        print(count)
        count += 1
        for row2 in data_set[0]:
            row2 = convert_categorical_attribute(row2)
            if(euclidean_dist(row1, row2) - distance.euclidean(row1, row2) >= 0.001):
                print("Euclidean Distance is wrong\n")
        #our_dist = euclidean_dist(row1, row2)
        #correct_dist = numpy.linalg.norm(numpy.asarray(row1) - numpy.asarray(row2))

# Tested.
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


def test_cosine(data_set):
    count = 0
    # Test for cosine distance correctness.
    for row1 in data_set[0]:
        row1 = convert_categorical_attribute(row1)
        print(count)
        count += 1
        for row2 in data_set[0]:
            row2 = convert_categorical_attribute(row2)
            if(cos_dist(row1, row2) - (distance.cosine(row1, row2)) >= 0.001):
                print("Cosine Distance is wrong\n")

# Tested.
def manhattan_dist(instance_0, instance_1):
    total = 0
    for i in range(len(instance_0)):
        total += abs(instance_0[i] - instance_1[i])
    return total

def test_manhattan(data_set):
    count = 0
    # Test for manhattan correctness.
    for row1 in data_set[0]:
        row1 = convert_categorical_attribute(row1)
        print(count)
        count += 1
        for row2 in data_set[0]:
            row2 = convert_categorical_attribute(row2)
            if(manhattan_dist(row1, row2) - (distance.cityblock(row1, row2)) >= 0.001):
                print("Manhattan Distance is wrong\n")


if __name__ == '__main__':
    main()
