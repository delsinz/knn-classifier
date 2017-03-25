""" Remember to add our names, student IDs here """

import csv
from collections import Counter, defaultdict
from random import shuffle
from scipy.spatial import distance

def main():
    '''
    REMOVE ME BEFORE SUBMITTING
    '''
    # Data set which is a two tuple.
    data_set = preprocess_data('data.data', 2)

    # Choose k = 5 for the 2 case.
    evaluation(data_set, dist='euclidean', k=31)

# Tested.
# Returns ([lists of 8 attributes], [class_label])
def preprocess_data(filename, abalone=3):
    '''
    Processes the input data before doing any sort of classification.
    Returns the data_set as a tuple: ([list with row of 8 attributes],
    [list of class_labels]).
    The function also takes the categorical gender attribute and turns
    it into a representation using numbers that plays well with our
    similarity metrics i.e it turns male into 0, infant into the
    mean numerical value of all the numerical attributes in the data set
    and female to 2 times the mean numerical value.

    Arguments:
    filename: The name of the file
    abalone: NOT PART OF THE SPEC, but takes whether we need to use
    abalone-3 or abalone-2 for our calculations.

    Return:
    A 2-tuple made of a list of instances and a list of class labels. 
    '''

    # Total value of the numerical values
    total_numerical = 0
    # Number of numerical values in the data set
    count_numerical = 0
    # Load data
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
            for attribute in row:
                try:
                    instance.append(float(attribute))
                    # Increment total_numerical for all numerical attributes.
                    total_numerical += float(attribute)
                    count_numerical += 1
                except ValueError:
                    instance.append(attribute)

            # Subtract the no. of rings as they are the value to be predicted.
            total_numerical -= instance[len(instance) - 1]
            count_numerical -= 1
            instances.append(instance[:len(instance) - 1])
            to_be_predicted.append(instance[len(instance) - 1])
    # Set class labels
    class_labels = assign_class_label(to_be_predicted, abalone)
    # Construct data set as a tuple of instances
    mean_numerical_value = total_numerical/count_numerical
    # Process instances so that the M, F and I values make more sense,
    # see docstring of convert_categorical_attribute for more info.
    processed_instances = []
    for instance in instances:
        processed_instance = convert_categorical_attribute(instance,
                             mean_numerical_value)
        processed_instances.append(processed_instance)

    data_set = (processed_instances, class_labels)
    return data_set

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
def convert_categorical_attribute(instance, mean_numerical_value):
    '''
    Converts Male ('M') to 0, Female ('F') to 2 times the
    mean_numerical_value of all the numerical values in the
    data set and Infant ('I') to mean_numerical_value. I
    fill in with mean_numerical_value so that the gender
    doesn't weigh too heavily. Also infant is between
    male and female, to signify that male and female are
    farthest apart.
    '''
    # Assuming the attributes are provided in the given order
    val = instance[0]
    if val == 'M':
        return [0] + instance[1:]
    elif val == 'F':
        return [2*mean_numerical_value] + instance[1:]
    elif val == 'I':
        return [mean_numerical_value] + instance[1:]

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
    return sorted_scores[:k]

def compare_instance(instance_0, instance_1, method):
    '''
    Compares instances based on the name of the methods
    passed to it
    '''
    if method == 'euclidean':
        return euclidean_dist(instance_0, instance_1)
    elif method == 'cos':
        return cos_dist(instance_0, instance_1)
    elif method == 'manhattan':
        return manhattan_dist(instance_0, instance_1)
    else:
        return None

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

# Tested.
def cos_dist(instance_0, instance_1):
    '''
    Computes and returns the Cosine Distance between
    instance_0 and instance_1
    The cosine distance = 1 - cosine similarity
    To give it the same behaviour as euclidean and
    manhattan distance
    '''
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


def evaluation(data_set, metric='accuracy', dist='euclidean', k=5, voting='ild'):
    
    score = 0
    partitioned_sets = partition_data(data_set)
    # Perform validation as many times as there are are datasets. 
    for i in range(len(partitioned_sets)):
        test_data_set = partitioned_sets[i]
        training_data_set = combine_data_sets(partitioned_sets[:i], partitioned_sets[i+1:])
        # print("Length of the training set is:" + str(len(training_data_set[0])))
        score += single_pass_eval(training_data_set, test_data_set, metric, dist, k, voting)
    print(str(k) + ", " + str(score / len(partitioned_sets)))


# Combine two lists of data sets into one set
def combine_data_sets(training_subsets0, training_subsets1):
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


def single_pass_eval(training_set, test_set, metric, dist, k, voting):
    predicted_classes = []
    for instance in test_set[0]:
        neighbors = get_neighbors(instance, training_set, k, dist)

        # print("My neighbors: " + str(neighbors))
        predicted_class = predict_class(neighbors, voting)
        # print("My predicted class: "  + str(predicted_class))
        predicted_classes.append(predicted_class)

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


def prime_finder():
    ''''
    Finds prime numbers in a certain range, the prime numbers were
    used as prospective k values in the k nearest neighbour classifier
    to find optimum values of key
    '''
    not_primes = set(j for i in range(2, 11) for j in range(i*2, 100, i))
    primes = [x for x in range(3, 100) if x not in not_primes]
    return primes

# Checked but not tested.
def partition_data(data_set):

    partitioned_sets = []
    M = 10
    set_size = len(data_set[0])
    # All partitions must be of this size. 
    partition_size = set_size // M
    
    '''
       These instances are the remainder after making all the partitions of the 
       same size and one element will be added to partitions (starting from the first) until
       we are out of the residual instances.
       So set size divider is the number of partitions (0th partition to (set_size_divider - 1)th partion)
       which will have one element more. 
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
    for i in range(10):
        if i < set_size_divider:
            partitioned_sets.append((shuffled_data_set[0][start:(start + partition_size + 1)],
            shuffled_data_set[1][start:(start + partition_size + 1)]))
            start += partition_size + 1
        else:
            partitioned_sets.append((shuffled_data_set[0][start:(start + partition_size)],
            shuffled_data_set[1][start:(start + partition_size)]))
            start += partition_size

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
    offset = 0.5 # Offset > 1 so that weight for cos dis won't be negative due to the way cos dist is handled
    for neighbor in neighbors:
        weight = 1 / (neighbor[1] + offset)
        label_votes[neighbor[0]] += weight
    return max(label_votes, key=label_votes.get)


def accuracy(test_set, predicted_classes, class_name):

    length = len(test_set[0])
    correct_predictions = 0

    for i in range(length):
        if test_set[1][i] == class_name and test_set[1][i] == predicted_classes[i]:
            correct_predictions += 1

        elif test_set[1][i] != class_name and predicted_classes[i] != class_name:
            correct_predictions += 1

    # print("Correctly predicted : " + str(correct_predictions) + " out of " + str(length))
    return correct_predictions/length*100



def complete_accuracy(test_set, predicted_classes):
    classes = list(set(test_set[1]))
    sum_accuracy = 0
    for class_name in classes:
        sum_accuracy += accuracy(test_set, predicted_classes, class_name)

    return sum_accuracy/len(classes)



def precision(test_set, predicted_classes, class_name):
    length = len(test_set[0])
    true_positives = 0
    false_positives = 0

    for i in range(length):
        if test_set[1][i] == class_name and predicted_classes[i] == class_name:
            true_positives += 1
        elif test_set[1][i] != class_name  and predicted_classes[i] == class_name:
            false_positives += 1
    if true_positives + false_positives == 0:
        return 1
    else:
        return true_positives/(true_positives + false_positives)

def complete_precision(test_set, predicted_classes):

    classes = list(set(test_set[1]))
    sum_precision = 0
    for class_name in classes:
        sum_precision += precision(test_set, predicted_classes, class_name)

    return sum_precision/len(classes)

def recall(test_set, predicted_classes, class_name):

    length = len(test_set[0])
    true_positives = 0
    false_negatives = 0

    for i in range(length):
        if test_set[1][i] == class_name and predicted_classes[i] == class_name:
            true_positives += 1
        elif test_set[1][i] == class_name  and predicted_classes[i] != class_name:
            false_negatives += 1

    if true_positives + false_negatives == 0:
        return 1
    else:
        return true_positives / (true_positives + false_negatives)



def complete_recall(test_set, predicted_classes):

    classes = list(set(test_set[1]))
    sum_recall = 0
    for class_name in classes:
        sum_recall += recall(test_set, predicted_classes, class_name)
        
    return sum_recall/len(classes)



def complete_error(test_set, predicted_classes):

    error = 100 - complete_accuracy(test_set, predicted_classes)

    return error

''' Tests for the distance metrics '''
'''
def test_euclidean(data_set):
    count = 0
    # Test for euclidean correctness.
    for row1 in data_set[0]:

        print(count)
        count += 1
        for row2 in data_set[0]:
            if(euclidean_dist(row1, row2) - distance.euclidean(row1, row2) >= 0.001):
                print("Euclidean Distance is wrong\n")
        #our_dist = euclidean_dist(row1, row2)
        #correct_dist = numpy.linalg.norm(numpy.asarray(row1) - numpy.asarray(row2))

def test_cosine(data_set):
    
    count = 0
    # Test for cosine distance correctness.
    for row1 in data_set[0]:

        print(count)
        count += 1
        for row2 in data_set[0]:

            if(cos_dist(row1, row2) - (distance.cosine(row1, row2)) >= 0.001):
                print("Cosine Distance is wrong\n")

def test_manhattan(data_set):
    count = 0
    # Test for manhattan correctness.
    for row1 in data_set[0]:

        print(count)
        count += 1
        for row2 in data_set[0]:

            if(manhattan_dist(row1, row2) - (distance.cityblock(row1, row2)) >= 0.001):
                print("Manhattan Distance is wrong\n")
'''

if __name__ == '__main__':
    main()
