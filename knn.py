import csv



def main():
    print(preprocess_data('data.data')[0][0])
    print(convert_categorical_attribute(preprocess_data('data.data')[0][0]))



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



def compare_instance(instance_0, instance_1, method):
    return 0


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
