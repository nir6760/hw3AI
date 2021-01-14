from ID3 import ID3 as ID3class
import utilis
from sklearn.model_selection import KFold


# evaluate the loss of specific classification, FN = 10*FP
def evaluate(real_classification, my_classification):
    if real_classification != my_classification:
        return 10 if real_classification == 'M' else 1  # he is sick
    return 0  # no False answer


# algorithem to decide if its better to prune tree
# for minimizing the loss as explained in the HW
def prune(tree, validation_group):
    if tree.children is None:  # it is a leave
        return tree
        # continues features, answers are binaries

    sub_validation_0 = validation_group[validation_group[tree.feature[0]] < tree.feature[1]]
    tree.children[0] = (0, prune(tree.children[0][1], sub_validation_0))

    sub_validation_1 = validation_group[validation_group[tree.feature[0]] >= tree.feature[1]]
    tree.children[1] = (1, prune(tree.children[1][1], sub_validation_1))

    err_prune = 0
    err_no_prune = 0

    for index, row in validation_group.iterrows():
        real_classification = row['diagnosis']
        err_prune += evaluate(real_classification, tree.class_of_node)
        err_no_prune += evaluate(real_classification, ID3class.classify(row, tree))

    if err_prune < err_no_prune:  # it is better to prune
        tree.f = None
        tree.children = None
    return tree


if __name__ == '__main__':
    E_train, F = utilis.createDF_train()
    E_test, F = utilis.createDF_test()
    best_predict = 0


    id3 = ID3class()
    id3.fit(E_train, F)
    print(id3.predict_loss(E_test))
    best_id3 = id3 #default
    epsilon = 0.05
    kf = KFold(n_splits=5, shuffle=True, random_state=123456789)  # todo: replace to 205467780
    for train_index, test_index in kf.split(E_train):
        id3_curr = ID3class()
        id3_curr.fit(E_train.loc[train_index], F)
        id3_curr.tree = prune(id3_curr.tree, E_train.loc[test_index])
        curr_predict = id3_curr.predict(E_train.loc[test_index])
        print(id3_curr.predict_loss(E_test), id3_curr.predict(E_test), curr_predict)
        if curr_predict >= best_predict and curr_predict < 1 - epsilon: # we don't want it will get too close to one,
            # because of overfitting
            best_predict = curr_predict
            best_id3 = id3_curr


    print()
    print(best_id3.predict_loss(E_test))
