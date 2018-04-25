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
    test = []
    t_array = []
    for s in test:
        s =s.rstrip()
        s = s.split(' ')
        t_array.append(s)
    print(clf.predict(t_array))
    pass


def do_for_random_forest_tree():
    pass

def do_for_svm(X_train, Y_train, X_test, nu=True):
    clf = svm.LinearSVC()
    clf.fit(X_train, Y_train)
    print("Linear SVC: ", clf.predict(X_test))
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    print("SVC: ", clf.predict(X_test))
    if nu:
        clf = svm.NuSVC()
        clf.fit(X_train, Y_train)
        print("NuSVC: ", clf.predict(X_test))

def filter_by_class_name(matrix, class_names, dataset):
    result_array = []
    for d in dataset:
        i = class_names.index(d)
        result_array.append(matrix[i])

    return result_array


def separate_xs_ys(row, indexes):
    ys = []
    deleted = 0
    new_row = list(row)
    for i in indexes:
        ys.append(row[i])
        del new_row[i-deleted]
        deleted += 1
    return [new_row, ys]

def fill_until_limit(low, high):
    index_array = []
    i = low
    while(i <= high):
        index_array.append(i)
        i += 1
    # print(index_array)
    return index_array

def do_limits_for_case(group):
    low = 0
    high = 0
    if group == 1: # strong?
        low = 41
        high = 41
    if group == 2: # eats meat?
        low = 52
        high = 52
    if group == 3: # black?
        low = 0
        high = 0
    if group == 4: # ocean? 
        low = 73
        high = 73
    if group == 5: # smart?
        low = 80
        high = 80
    if group == 6: # domestic?
        low = 84
        high = 84
    if group == 7: # color
        low = 0
        high = 7 
    if group == 8: # fly?
        low = 34
        high = 34
    if group == 9: # fast?
        low = 39
        high = 39
    if group == 10: # thoughskin
        low = 13
        high = 13
    
    indexes = fill_until_limit(low, high)
    return indexes



def main():
    matrix = []
    for t in tuples:
        t = t.rstrip()
        t = t.split(' ')
        new_t = []
        for i in t:
            new_t.append(int(i))
        matrix.append(new_t)

    class_numbers = []
    class_names = []
    for c in classes:
        c = c.rstrip()
        c = c.split(' ')
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
    test_set = filter_by_class_name(matrix, class_names, test_class_data)
    indexes = do_limits_for_case(5)    # We should ask for this as user input

    matrix = []
    Y_train = []
    result_train = []
    for t in training_set:
        result_train = separate_xs_ys(t, indexes)
        matrix.append(result_train[0])
        if len(indexes) == 1:
            Y_train += result_train[1]
        else:
            Y_train.append(''.join(str(e) for e in result_train[1]))

    result_test = []
    X_test = []
    Y_test = []
    for ts in test_set:
        result_test = separate_xs_ys(ts, indexes)
        X_test.append(result_test[0])
        if len(indexes) == 1:
            Y_test += result_test[1]
        else:
            Y_test.append(''.join(str(e) for e in result_test[1]))

    print("Expected result:")
    print(Y_test)
    print("Actual result:")
    if len(indexes) == 1:
        do_for_svm(matrix, Y_train, X_test)
    else:
        do_for_svm(matrix, Y_train, X_test, False)

    #do_for_neural_networks(matrix, class_numbers)

if __name__ == "__main__":
    # execute only if run as a script
    main()
    predicate_matrix_file.close()  
    all_classes_file.close()
    train_classes_file.close()
    test_classes_file.close()