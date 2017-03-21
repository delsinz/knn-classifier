import csv



def main():
    print(preprocess_data('data.data'))



def preprocess_data(filename, abalone = 2):
    # Load data
    file = open(filename)
    reader = csv.reader(file)

    # Construct instance list
    instances = []
    for row in reader:
        instances.append(row)

    # Construct class labels
    class_labels = []
    if abalone == 3:
        for instance in instances:
            rings = int(instance[8])
            label = ''
            if rings <= 8:
                label = 'very-young'
            elif rings <= 10:
                label = 'middle-age'
            else:
                label = 'old'
            class_labels.append(label)
    else:
        for instance in instances:
            rings = int(instance[8])
            label = ''
            if rings <= 10:
                label = 'young'
            else:
                label = 'old'
            class_labels.append(label)

    # Construct data set
    data_set = [instances, class_labels]
    return data_set





if __name__ == '__main__':
    main();
