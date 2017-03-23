import csv
from collections import Counter, defaultdict



def main():
    data_set = preprocess_data('data.data', 3)
    instance = data_set[0][16]
    neighbors = get_neighbors(instance, data_set, 20, 'cos')
    print(neighbors)
    print(predict_class(neighbors, 'ew'))
    print(predict_class(neighbors, 'ild'))



# Returns [[lists of 8 attributes], [class_label]]
def preprocess_data(filename, abalone = 2):
    # Load data
    file = open(filename)
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
    else:
        for rings in to_be_predicted:
            label = ''
            if rings <= 10:
                label = 'young'
            else:
                label = 'old'
            class_labels.append(label)

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
        return 0



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
    '''if method == 'cos': # For cos sim, larger values are better
        sorted_scores = list(reversed(sorted(scores, key=lambda x:x[1])))
    else: # For euclidean dist and manhattan dist, smaller values are better
        sorted_scores = sorted(scores, key=lambda x:x[1])'''
    return sorted_scores[:k]



def predict_class(neighbors, method = 'ew'):
    if method == 'ew':
        return predict_equal_weight(neighbors)
    elif method == 'ild':
        return predict_inverse_linear_dist(neighbors)
    elif method == 'id':
        return 0
    else:
        return 0



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
    return 0


def euclidean_dist(instance_0, instance_1):
    # Assuming both instances have the same num of attributes
    length = len(instance_0)
    square_sum = 0
    for i in range(length):
        square_sum += (instance_0[i] - instance_1[i]) ** 2
    return square_sum ** 0.5



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
    if(mag_0 * mag_1 == 0): # Orthogonal
        return 0
    else:
        # Give cos dist the same behavior (smaller == better) as euclidean and manhattan
        # similarity -> distance, thus the negation.
        return -dot_prod / (mag_0 * mag_1)



def manhattan_dist(instance_0, instance_1):
    total = 0
    for i in range(len(instance_0)):
        total += abs(instance_0[i] - instance_1[i])
    return total



# Break categorical attribute Sex into 3 binary attributes: M, F, I
def convert_categorical_attribute(instance):
    # Assuming the attributes are provided in the given order
    val = instance[0];
    if val == 'M':
        return [1, 0, 0] + instance[1:]
    elif val == 'F':
        return [0, 1, 0] + instance[1:]
    elif val == 'I':
        return [0, 0, 1] + instance[1:]
    else:
        return [0, 0, 0] + instance[1:]



if __name__ == '__main__':
    main();
