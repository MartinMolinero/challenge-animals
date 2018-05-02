from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
from prettytable import PrettyTable

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

def do_for_neural_networks(X_train, Y_train, X_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, Y_train)
    neural =  clf.predict(X_test)
    return neural



def do_for_random_forest_tree():
    pass

def do_for_svm(X_train, Y_train, X_test, nu=True):
    clf = svm.LinearSVC()
    clf.fit(X_train, Y_train)
    linearSvc = clf.predict(X_test)

    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    svc = clf.predict(X_test)

    if nu:
        clf = svm.NuSVC()
        clf.fit(X_train, Y_train)
        nuSvc = clf.predict(X_test)
        return linearSvc, svc, nuSvc
    else:
        return linearSvc, svc

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
    print("Group", group)
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
    if group == 8: # hunter?
        low = 58
        high = 58
    if group == 9: # small?
        low = 15
        high = 15
    if group == 10: # smelly
        low = 33
        high = 33

    indexes = fill_until_limit(low, high)
    return indexes

def get_percentage(expected, result):
    eq = 0
    for i in range(len(expected)):
        if expected[i] == result[i]:
            eq += 1

    per = eq * 100 / len(expected)
    return per

def get_percentage_each(expected, result):
    eq = 0
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            if expected[i][j] == result[i][j]:
                eq += 1

    per = eq * 100 / (len(expected)*len(expected[0]))
    return per

def convert_to_animals(arr, animals, indexes):
    if len(indexes) == 1:
        table = PrettyTable(['Animal', 'Predicate'])
        for i in range(len(arr)):
            row = []
            if arr[i] == 1:
                row.append(animals[i].rstrip())
                row.append("YES")
            else:
                row.append(animals[i].rstrip())
                row.append("NO")
            table.add_row(row)
    else:
        table = PrettyTable(['Animal', 'Predicate1', 'Predicate2', 'Predicate3', 'Predicate4', 'Predicate5', 'Predicate6', 'Predicate7', 'Predicate8'])

        for i in range(len(animals)):
            row = []
            row.append(animals[i].rstrip())
            for j in range(len(arr[i])):
                if arr[i][j] == '1':
                    row.append("YES")
                elif arr[i][j] == '0':
                    row.append("NO")
            table.add_row(row)
    print(table)

def convert_to_animals_results(animals, linearSvc, svc, nuSvc, neural, perLinear, perSvc, perNuSvc, perNeural):
    table = PrettyTable(['Animal', 'LinearSVC', 'SVC', 'NuSVC', 'Neural'])

    for i in range(len(animals)):
        row = []
        row.append(animals[i].rstrip())
        if linearSvc[i] == 1:
            row.append("YES")
        elif linearSvc[i] == 0:
            row.append("NO")
        if svc[i] == 1:
            row.append("YES")
        elif svc[i] == 0:
            row.append("NO")
        if nuSvc[i] == 1:
            row.append("YES")
        elif nuSvc[i] == 0:
            row.append("NO")
        if neural[i] == 1:
            row.append("YES")
        elif neural[i] == 0:
            row.append("NO")
        table.add_row(row)
    percentage = []
    percentage.extend(("PERCENTAGE", perLinear, perSvc, perNuSvc, perNeural))
    table.add_row(percentage)
    print(table)

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
    string = "1"
    print("Select the option you want to know from your animals")
    colors = "\n\t black\n\t white\n\t blue\n\t brown\n\t gray\n\t orange\n\t red\n\t yellow"
    indexes = do_limits_for_case(int(input("\n1 Are they strong? \n2 Do they eat meat? \n3 Are they black? \n4 Do they live in the ocean? \n5 Are they smart? \n6 Are they domestic?\n7 Do they have one of the following colors?"+ colors +  "\n8 Are they hunters ?\n9 Is it small?\n10 Is it smelly?\n\n")))    # We should ask for this as user input

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
    convert_to_animals(Y_test, test_classes, indexes)
    print("Actual results:\n")
    if len(indexes) == 1:
        linearSvc, svc, nuSvc = do_for_svm(matrix, Y_train, X_test)
        print("Linear SVC: ", linearSvc, get_percentage(Y_test, list(linearSvc)),"%")
        print("SVC: ",svc, get_percentage(Y_test, list(svc)),"%")
        print("NuSVC: ",nuSvc, get_percentage(Y_test, list(nuSvc)),"%")
        neural = do_for_neural_networks(matrix, Y_train, X_test)
        print("Neural Network (lbfgs): ", neural, get_percentage(Y_test, list(neural)),"%")
        convert_to_animals_results(test_classes, list(linearSvc), list(svc), list(nuSvc), list(neural), get_percentage(Y_test, list(linearSvc)), get_percentage(Y_test, list(svc)), get_percentage(Y_test, list(nuSvc)), get_percentage(Y_test, list(neural)))
    else:
        linearSvc, svc = do_for_svm(matrix, Y_train, X_test, False)
        print("Linear SVC: ", linearSvc, get_percentage_each(Y_test, list(linearSvc)),"%")
        print("SVC: ",svc, get_percentage_each(Y_test, list(svc)),"%")
        neural = do_for_neural_networks(matrix, Y_train, X_test)
        print("Neural Network (lbfgs): ", neural, get_percentage_each(Y_test, list(neural)),"%")
        print("Linear SVC")
        convert_to_animals(linearSvc, test_classes, indexes)
        print("SVC")
        convert_to_animals(svc, test_classes, indexes)
        print("Neural")
        convert_to_animals(neural, test_classes, indexes)


if __name__ == "__main__":
    # execute only if run as a script
    main()
    predicate_matrix_file.close()
    all_classes_file.close()
    train_classes_file.close()
    test_classes_file.close()
