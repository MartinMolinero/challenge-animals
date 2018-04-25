from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np

predicate_matrix_file  = open('AwA-base/Animals_with_Attributes/predicate-matrix-binary.txt','r')
all_classes_file = open('AwA-base/Animals_with_Attributes/classes.txt', 'r')
train_classes_file = open('AwA-base/Animals_with_Attributes/trainclasses.txt', 'r')
test_classes_file = open('AwA-base/Animals_with_Attributes/testclasses.txt', 'r')
#train_predicates = open('AwA-base/Animals_with_Attributes/train_predicates.txt', 'r')

tuples = predicate_matrix_file.readlines()
#print(tuples)
classes = all_classes_file.readlines()
train_classes = train_classes_file.readlines()
test_classes = test_classes_file.readlines()
#print(classes)

'''
lo que vi en general es que cada metodo de clustering que estamos usando
para entrenarse usa el metodo fit, ese metodo recibe en x los data samples de training
y en y las diferentes clases en las que puede clasificar al parecer.
'''

def do_for_neural_networks(x_array, class_numbers):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(x_array, class_numbers)
    test = []#train_predicates.readlines()
    t_array = []
    for s in test:
        s =s.rstrip()
        s = s.split(' ')
        t_array.append(s)
    print(clf.predict(t_array))
    pass


def do_for_random_forest_tree():
    pass

def do_for_svm(x_array, class_numbers):
    clf = svm.SVC()
    print(clf.fit(x_array, class_numbers))
    test = []#train_predicates.readlines()
    t_array = []
    for s in test:
        s =s.rstrip()
        s = s.split(' ')
        t_array.append(s)
    print(clf.predict(np.array(t_array)))

def filter_by_class_name(matrix, class_names, dataset):
    result_array = []
    for d in dataset:
        i = class_names.index(d)
        result_array.append(matrix[i])

    return result_array


def separate_xs_ys(row, indexes):
    ys = []
    print("initial row")
    print(row)
    deleted = 0
    for i in indexes:
        ys.append(row[i-deleted])
        del row[i-deleted]
        deleted += 1
    print("final row")
    print(row)
    print("ys")
    print(ys)
    return [row, ys]

def fill_until_limit(low, high):
    index_array = []
    i = low
    while(i <= high):
        index_array.append(i)
        i += 1
    print(index_array)
    return index_array

def do_limits_for_case(group):
    low = 0
    high = 0
    if group == 1:
        low = 0
        high = 5
    if group == 2:
        low = 0
        high = 5
    if group == 3:
        low = 0
        high = 5
    if group == 4:
        low = 0
        high = 5
    if group == 5:
        low = 0
        high = 5
    if group == 6:
        low = 0
        high = 5
    if group == 7:
        low = 0
        high = 5
    if group == 8:
        low = 0
        high = 5
    if group == 9:
        low = 0
        high = 5
    if group == 10:
        low = 0
        high = 5
    if group == 11:
        low = 0
        high = 5
    if group == 12:
        low = 0
        high = 5
    if group == 13:
        low = 0
        high = 5
    if group == 14:
        low = 0
        high = 5
    if group == 15:
        low = 0
        high = 5
    if group == 16:
        low = 0
        high = 5
    if low != high:
        indexes = fill_until_limit(low, high)
        return indexes



def main():
    matrix = []
    for t in tuples:
        t = t.rstrip()
        t = t.split(' ')
        matrix.append(t)

    class_numbers = []
    class_names = []
    for c in classes:
        #print(c)
        c = c.rstrip()
        c = c.split(' ')
        #print(c)
        class_numbers.append(c[0])
        class_names.append(c[1])
    training_class_data = []
    test_class_data = []
    for tc in train_classes:
        tc = tc.rstrip()
        training_class_data.append(tc)

    for ts in test_classes:
        ts = ts.rstrip()
        test_class_data.append(ts)

    training_set = filter_by_class_name(matrix, class_names, training_class_data)
    print(len(training_set))
    test_set = filter_by_class_name(matrix, class_names, test_class_data)
    print(len(test_set))
    indexes = do_limits_for_case(1)
    result_train = []
    for t in training_set:
        result_train = separate_xs_ys(t, indexes)
        x_train = result_train[0]
        y_train = result_train[1]
    result_test = []
    for ts in test_set:
        result_test = separate_xs_ys(t, indexes)
        x_test = result_test[0]
        y_test = result_test[1]

    #do_for_svm(matrix, class_numbers)
    #do_for_neural_networks(matrix, class_numbers)

if __name__ == "__main__":
    # execute only if run as a script
    main()
