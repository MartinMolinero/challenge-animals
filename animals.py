from sklearn import svm

predicate_matrix_file  = open('AwA-base/Animals_with_Attributes/predicate-matrix-binary.txt','r')
train_classes_file = open('AwA-base/Animals_with_Attributes/classes.txt', 'r')
train_predicates = open('AwA-base/Animals_with_Attributes/train_predicates.txt', 'r')

tuples = predicate_matrix_file.readlines()
#print(tuples)
classes = train_classes_file.readlines()
#print(classes)

def do_for_neural_networks(x_array, class_numbers):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(x_array, class_numbers)
    test = train_predicates.readlines()
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
    clf = svm.SVC(decision_function_shape='ovr')
    print(clf.fit(x_array, class_numbers))
    test = train_predicates.readlines()
    t_array = []
    for s in test:
        s =s.rstrip()
        s = s.split(' ')
        t_array.append(s)
    print(clf.predict(t_array))

def main():
    x_array = []
    for t in tuples:
        t = t.rstrip()
        t = t.split(' ')
        x_array.append(t)

    class_numbers = []
    class_names = []
    for c in classes:
        #print(c)
        c = c.rstrip()
        c = c.split(' ')
        #print(c)
        class_numbers.append(c[0])
        class_names.append(c[1])

    do_for_svm(x_array, class_numbers)
    do_for_neural_networks(x_array, class_numbers)
