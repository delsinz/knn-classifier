import csv



def main():
    print(preprocess_data('data.data')[0][0])
    print(convert_categorical_attribute(preprocess_data('data.data')[0][0]))
    print(euclidean_dist([1,0], [1,0]))
    print(cos_sim([1, 0], [0,0]))
    print(manhattan_dist([1, 0], [0,0]))



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
    data_set = [instances, class_labels]
    return data_set



def compare_instance(instance_0, instance_1, method = 'euclidean'):
    instance_0 = convert_categorical_attribute(instance_0)
    instance_1 = convert_categorical_attribute(instance_1)

    if method == 'euclidean':
        return euclidean_dist(instance_0, instance_1)
    elif method == 'cos':
        return cos_sim(instance_0, instance_1)
    elif method == 'manhattan':
        return manhattan_dist(instance_0, instance_1)
    else:
        return 0



def euclidean_dist(instance_0, instance_1):
    # Assuming both instances have the same num of attributes
    length = len(instance_0)
    square_sum = 0
    for i in range(length):
        square_sum += (instance_0[i] - instance_1[i]) ** 2
    return square_sum ** 0.5



def cos_sim(instance_0, instance_1):
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
        return dot_prod / (mag_0 * mag_1)



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
