"""
Shreyash Patodia, Student ID: 767336
Username: spatodia
Mingyang Zhang, Student ID: 650242
Username: mingyangz
"""

import csv
from collections import Counter, defaultdict
from random import shuffle
from math import sqrt
# from scipy.spatial import distance
import random


def main():
    data_set = preprocess_data('data.data', 3)
    ans = evaluate(data_set, 'error', 'minkowski')
    evaluate(data_set, 'error')
    print(ans)

'''========= Core functions ========='''

# Returns ([lists of 8 attributes], [class_label])
'''
Usage: preprocess_data(filename) for abalone-3, 
preprocess_data(filename, 2) for abalone-2
'''
def preprocess_data(filename, abalone=3):
    '''
    Processes the input data before doing any sort of classification.
    Returns the data_set as a tuple: ([list with row of 8 attributes],
    [list of class_labels]).
    Arguments:
    filename: The name of the file
    abalone: NOT PART OF THE SPEC, but takes whether we need to use
    abalone-3 or abalone-2 for our calculations.
    Return:
    A 2-tuple made of a list of instances and a list of class labels.
    '''
    # Load data
    (instances, class_labels) = read_file(filename, abalone)

    # Process instances so that the M, F and I values make more sense,
    # see docstring of convert_categorical_attribute for more info.
    data_set = (instances, class_labels)

    processed_instances = []
    for instance in data_set[0]:
        processed_instance = process_instances(instance)
        processed_instances.append(processed_instance)

    processed_data_set = (processed_instances, data_set[1])
    return processed_data_set

'''
   Usage: evaluate(data_set) or evaluate(data_set, 'accuracy') for accuracy,
   can replace the second metric by 'precision', 'recall' or 'error'. 
   If something incorrect is passed as second parameter we just get the None. 
   Can pass dist metric otherwise euclidean is used, k can be anything between
   1 to 4177 and voting can be 'ew' -> equal weight, 'id' -> inverse distance
   and 'ild' -> inverse linear distance
'''
def evaluate(data_set, metric='accuracy', dist='euclidean', k=29, voting='ew'):
    '''
    Evaluate classifier based on the distance function, k value, and voting
    method.
    data_set: 2 tuple. ([list of instances], [list of class labels])
    metric: accuracy || recall || precision || error
    dist: euclidean || cos || manhattan
    k: positive integer
    ew: equal weight, ild: inverse linear distance, id: inverse distance.
    voting: ew || ild || id
    '''

    # Partition the data into 10-Folds. 
    partitioned_sets = partition_data(data_set)

    # Evaluation metrics. 
    sum_accuracy = 0
    sum_error = 0
    sum_precision = 0
    sum_recall = 0

    # Perform validation as many times as there are are datasets.
    for i in range(len(partitioned_sets)):
        test_data_set = partitioned_sets[i]

        training_data_set = combine_data_sets(partitioned_sets[:i],
                                              partitioned_sets[i+1:])
        # print("Length of the training set is:" + str(len(training_data_set[0])))
        (run_accuracy, run_error, run_precsion, run_recall) = single_pass_eval(
        training_data_set, test_data_set, dist, k, voting)
        sum_accuracy += run_accuracy
        sum_error += run_error
        sum_precision += run_precsion
        sum_recall += run_recall
    #return score / len(partitioned_sets)

    print("k=" + str(k) + ", Accuracy=" 
    + str(sum_accuracy/ len(partitioned_sets)) + ", Error="
    + str(sum_error/ len(partitioned_sets)) + ", Precision="
    + str(sum_precision/ len(partitioned_sets)) + ", Recall="
    + str(sum_recall/ len(partitioned_sets)))

    if(metric == "accuracy"):
        return sum_accuracy/len(partitioned_sets)
    elif(metric == "precision"):
        return sum_precision/len(partitioned_sets)
    elif(metric == "recall"):
        return sum_recall/len(partitioned_sets)
    elif(metric == "error"):
        return sum_error/len(partitioned_sets)
    else:
        return None



def get_neighbors(instance, training_data_set, k, method):
    '''
    Finds the k closest neighbours to the instance parameter
    based on the distance method provided as parameter
    '''
    # Get class labels and scores
    training_instances = training_data_set[0]
    class_labels = training_data_set[1]
    size = len(training_instances)
    scores = []
    for i in range(size):
        scores.append((class_labels[i], compare_instance(instance,
                       training_instances[i], method)))

    # Sort result
    sorted_scores = sorted(scores, key=lambda x: x[1])
    # Return the list of tuples. 
    return sorted_scores[:k]


'''
   Usage: can pass 'euclidean', 'manhattan' and 'minkowski'
   as the method, along with instance_0 and instance_1
'''
def compare_instance(instance_0, instance_1, method):
    '''
    Compares instances based on the name of the methods
    passed to it
    '''
    if method == 'euclidean':
        return euclidean_dist(instance_0, instance_1)
    elif method == 'minkowski':
        return minkowski_dist(instance_0, instance_1)
    elif method == 'manhattan':
        return manhattan_dist(instance_0, instance_1)
    else:
        return None

'''
    Usage: Voting method can be: 'ew' for equal weight, 'id' for inverse distane' and
   'ild' for inverse linear distance, neighbors is a list of tuples as specificed for
    the assignment. 
'''
def predict_class(neighbors, method):
    '''
    Takes neighbors (list of class labels) and method (string that specifies
    voting method) as arguments. Return the predicted class.
    '''

    if method == 'ew':
        return predict_equal_weight(neighbors)
    elif method == 'ild':
        return predict_inverse_linear_dist(neighbors)
    elif method == 'id':
        return predict_inverse_dist(neighbors)
    else:
        return None



'''========== Helper functions =========='''


'''====== Pre-processing function ======='''

def read_file(filename, abalone):
    '''
    Reads the file and returns our dataset
    '''
    with open(filename) as file:
        reader = csv.reader(file)
        # Construct instance list
        instances = []
        # to_be_predicted is the attribute to be predicted i.e. Rings
        # or age.
        to_be_predicted = []
        # Read each row and attribute and add to instances.
        for row in reader:
            instance = []
            for i in range(len(row)):
                attribute = row[i]
                try:
                    if i in [3, 4]:
                        instance.append(float(attribute)*2)
                    else:
                        instance.append(float(attribute))
                except ValueError:
                    instance.append(attribute)

            instances.append(instance[:len(instance) - 1])
            to_be_predicted.append(instance[len(instance) - 1])

    # Set class labels
    class_labels = assign_class_label(to_be_predicted, abalone)
    return (instances, class_labels)



def assign_class_label(to_be_predicted, abalone):
    '''
    Assigns labels based on numbers of rings and the value
    of abalone supplied
    '''
    class_labels = []

    # For abalone 3
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

    # For abalone 2
    elif abalone == 2:
        for rings in to_be_predicted:
            label = ''
            if rings <= 10:
                label = 'young'
            else:
                label = 'old'
            class_labels.append(label)

    return class_labels

# Break categorical attribute Sex into 3 binary attributes: M, F, I

def process_instances(instance):
    '''
    Removes the gender attribute and does feature vector normalisation
    '''
    # Assuming the attributes are provided in the given order

    instance = instance[1:]
    instance_square = [i ** 2 for i in instance]
    sum_squares = sum(instance_square)
    root_sum =  sum_squares ** 0.5
    instance = [x/root_sum for x in instance]
    return instance


'''====== Distance/Similarity Functions ======'''

def minkowski_dist(instance_0, instance_1):
    '''
    Computes and returns the Minkowski distance between
    instance_0 and instance_1, with p = 0.5
    '''
    p_val = float(0.5)
    # Assuming both instances have the same num of attributes
    length = len(instance_0)
    square_sum = 0
    for i in range(length):
        square_sum += abs(instance_0[i] - instance_1[i]) ** p_val
    return square_sum ** (1/p_val)

def euclidean_dist(instance_0, instance_1):
    '''
    Computes and returns the Euclidean distance between
    instance_0 and instance_1
    '''
    # Assuming both instances have the same num of attributes
    length = len(instance_0)
    square_sum = 0
    for i in range(length):
        square_sum += (instance_0[i] - instance_1[i]) ** 2
    return square_sum ** 0.5

def manhattan_dist(instance_0, instance_1):
    '''
    Computes and returns the manhattan distance
    between instance_0 and instance_1
    '''
    total = 0
    length = len(instance_0)
    for i in range(length):
        total += abs(instance_0[i] - instance_1[i])
    return total


''''
def cos_dist(instance_0, instance_1):
    # Computes and returns the Cosine Distance between
    # instance_0 and instance_1
    # The cosine distance = 1 - cosine similarity
    # To give it the same behaviour as euclidean and
    # manhattan distance
    mag_0 = 0
    mag_1 = 0
    dot_prod = 0
    length = len(instance_0)
    for i in range(length):
        mag_0 += instance_0[i] ** 2
        mag_1 += instance_1[i] ** 2
        dot_prod += instance_0[i] * instance_1[i]
    mag_0 = mag_0 ** 0.5
    mag_1 = mag_1 ** 0.5

    if dot_prod == 0: #Orthogonal
        return 0
    else:
        # Give cos dist the same behavior (smaller == better) as euclidean
        # and manhattan
        return 1 - dot_prod / (mag_0 * mag_1)
'''


'''========= Voting Method Functions =============='''

def predict_equal_weight(neighbors):
    '''
    Returns the majority class amongst the neightbours
    '''
    # print('In equal weight')
    labels = [neighbor[0] for neighbor in neighbors]
    label_votes = dict(Counter(labels))
    return max(label_votes, key=label_votes.get)



def predict_inverse_linear_dist(neighbors):
    '''
    Computes the inverse distance of neighbors of a certain type
    from the test data point and returns the maximum of the inverses
    i.e. the most similar class with label_votes
    '''
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
    '''
    Computes the inverse distance of neighbors of a certain type
    from the test data point and returns the maximum of the inverses
    i.e. the most similar class with label_votes
    '''
    label_votes = defaultdict(float)
    offset = 0.5
    for neighbor in neighbors:
        weight = 1 / (neighbor[1] + offset)
        label_votes[neighbor[0]] += weight
    return max(label_votes, key=label_votes.get)

'''======== Functions for the evalution metrics =========='''

def accuracy(test_set, predicted_classes, class_name):
    '''
    Finds the accuracy taking the class_name in the function arguments
    as the positive class and the rest as the negative class
    '''
    length = len(test_set[0])
    correct_predictions = 0

    for i in range(length):
        if test_set[1][i] == class_name and test_set[1][i] == predicted_classes[i]:
            correct_predictions += 1

        elif test_set[1][i] != class_name and predicted_classes[i] != class_name:
            correct_predictions += 1

    return correct_predictions/length



def complete_accuracy(test_set, predicted_classes):
    '''
    Finds the accuracy for all the classes and averages them
    '''
    classes = list(set(test_set[1]))
    #print(len(test_set[1]))
    sum_accuracy = 0
    for class_name in classes:
        sum_accuracy += accuracy(test_set, predicted_classes, class_name)

    return sum_accuracy/len(classes)



def precision(test_set, predicted_classes, class_name):
    '''
    Finds the precision taking the class_name in the function arguments
    as the positive class and the rest as the negative class
    '''
    length = len(test_set[0])
    true_positives = 0
    false_positives = 0

    for i in range(length):
        if test_set[1][i] == class_name and predicted_classes[i] == class_name:
            true_positives += 1
        elif test_set[1][i] != class_name and predicted_classes[i] == class_name:
            false_positives += 1
    if true_positives + false_positives == 0:
        return 1
    else:
        print(true_positives/(true_positives + false_positives))
        return true_positives/(true_positives + false_positives)



def complete_precision(test_set, predicted_classes):
    '''
    Finds the percision for all the classes and macro-averages them
    '''
    classes = list(set(test_set[1]))
    sum_precision = 0
    for class_name in classes:
        sum_precision += precision(test_set, predicted_classes, class_name)

    return sum_precision/len(classes)



def recall(test_set, predicted_classes, class_name):
    '''
    Finds the recall taking class_name in the function arguments
    as the positive class and the rest as the negative class
    '''
    length = len(test_set[0])
    true_positives = 0
    false_negatives = 0

    for i in range(length):
        if test_set[1][i] == class_name and predicted_classes[i] == class_name:
            true_positives += 1
        elif test_set[1][i] == class_name and predicted_classes[i] != class_name:
            false_negatives += 1

    if true_positives + false_negatives == 0:
        return 1
    else:
        return true_positives / (true_positives + false_negatives)



def complete_recall(test_set, predicted_classes):
    '''
    Finds the recall for all the classes and macro-averages them
    '''
    classes = list(set(test_set[1]))
    sum_recall = 0
    for class_name in classes:
        sum_recall += recall(test_set, predicted_classes, class_name)

    return sum_recall/len(classes)



def complete_error(test_set, predicted_classes):
    '''
    Finds the error as 100 - accuracy of the system
    '''
    error = 1 - complete_accuracy(test_set, predicted_classes)
    return error

'''========= Function for validation ========'''

def partition_data(data_set):
    '''
    Partitions the data into 10 folds for 10 fold cross 
    validation, change M to change the number of folds. 
    '''
    partitioned_sets = []
    M = 10
    set_size = len(data_set[0])
    # All partitions must be of this size.
    partition_size = set_size // M

    '''
       These instances are the remainder after making all the partitions of the
       same size and one element will be added to partitions (starting from the
       first) until we are out of the residual instances.
       So set size divider is the number of partitions (0th partition to
       (set_size_divider - 1)th partion) which will have one element more.
    '''
    set_size_divider = set_size % M

    # Break down the data set
    data_list = [data_set[0][i]+[data_set[1][i]] for i in range(set_size)]

    # Random orderring, for fair partitioning
    shuffle(data_list)
    # Construct new randomly ordered data set

    instances = []
    class_labels = []
    for row in data_list:
        instances.append(row[:len(row) - 1])
        class_labels.append(row[len(row)- 1])

    # The shuffled data set.
    shuffled_data_set = (instances, class_labels)
    start = 0

    # 10 fold cross validation.
    # Each elements of partitioned_sets will be used as test instance once.
    for i in range(M):
        if i < set_size_divider:
            partitioned_sets.append((
                shuffled_data_set[0][start:(start + partition_size + 1)],
                shuffled_data_set[1][start:(start + partition_size + 1)]))
            start += partition_size + 1
        else:
            partitioned_sets.append((
                shuffled_data_set[0][start:(start + partition_size)],
                shuffled_data_set[1][start:(start + partition_size)]))
            start += partition_size
    return partitioned_sets



# Combine two lists of data sets into one set
def combine_data_sets(training_subsets0, training_subsets1):
    '''
    Combines the M-1 partitions of the dataset into one
    set so that they can be used as training data for the
    current pass of the M-Fold Cross validation
    '''
    instances = []
    class_labels = []
    for subset in training_subsets0:
        for instance in subset[0]:
            instances.append(instance)
        for label in subset[1]:
            class_labels.append(label)
    for subset in training_subsets1:
        for instance in subset[0]:
            instances.append(instance)
        for label in subset[1]:
            class_labels.append(label)
    return (instances, class_labels)



def single_pass_eval(training_set, test_set, dist, k, voting):
    '''
    Evaluate the classifier based on test_set.
    The result will be averaged in M-fold cross-validation.
    '''
    predicted_classes = []
    for instance in test_set[0]:

        neighbors = get_neighbors(instance, training_set, k, dist)
        # print("My neighbors: " + str(neighbors))
        predicted_class = predict_class(neighbors, voting)
        # print("My predicted class: "  + str(predicted_class))
        predicted_classes.append(predicted_class)

    accuracy = complete_accuracy(test_set, predicted_classes)
    error = complete_error(test_set, predicted_classes)
    precision = complete_precision(test_set, predicted_classes)
    recall = complete_recall(test_set, predicted_classes)

    return (accuracy, error, precision, recall)


''' ========= Functions to help testing & comparison ========= '''
'''
def prime_finder():
    not_primes = set(j for i in range(2, 11) for j in range(i*2, 100, i))
    primes = [x for x in range(3, 100) if x not in not_primes]
    return primes



def holdout(data_set, split_ratio):

    instances = data_set[0]
    classes = data_set[1]
    training_instances = []
    training_classes = []
    test_instances = []
    test_classes = []
    for i in range(len(instances)):
        if random.random() < split_ratio:
            training_instances.append(instances[i])
            training_classes.append(classes[i])

        else:
            test_instances.append(instances[i])
            test_classes.append(classes[i])
    training_data_set = (training_instances, training_classes)
    test_data_set = (test_instances, test_classes)
    print(single_pass_eval(training_data_set, test_data_set,
                           dist='euclidean', k=29, voting='ew'))



def test_euclidean(data_set):
    count = 0
    # Test for euclidean correctness.
    for row1 in data_set[0]:

        print(count)
        count += 1
        for row2 in data_set[0]:
            if(euclidean_dist(row1, row2)
                - distance.euclidean(row1, row2) >= 0.001):
                print("Euclidean Distance is wrong\n")



def test_cosine(data_set):

    count = 0
    # Test for cosine distance correctness.
    for row1 in data_set[0]:

        print(count)
        count += 1
        for row2 in data_set[0]:

            if(cos_dist(row1, row2)
                - (distance.cosine(row1, row2)) >= 0.001):
                print("Cosine Distance is wrong\n")



def test_manhattan(data_set):
    count = 0
    # Test for manhattan correctness.
    for row1 in data_set[0]:

        print(count)
        count += 1
        for row2 in data_set[0]:

            if(manhattan_dist(row1, row2)
                - (distance.cityblock(row1, row2)) >= 0.001):
                print("Manhattan Distance is wrong\n")



def column_means(data_set):

    instances = data_set[0]
    means = [0 for i in range(len(instances[0]))]

    for i in range(1, len(instances[0])):
	    col_values = [row[i] for row in instances]
	    means[i] = sum(col_values) / float(len(instances))
    return means



def column_stdevs(data_set, means):
    instances = data_set[0]
    stdevs = [0 for i in range(len(instances[0]))]

    for i in range(1, len(instances[0])):
	    variance = [pow(row[i]-means[i], 2) for row in instances]
	    stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(instances)-1))) for x in stdevs]
    return stdevs


'''



if __name__ == '__main__':
    main()