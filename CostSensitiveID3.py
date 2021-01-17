from ID3 import ID3 as ID3class
import utilis


class CostSensitiveID3:
    def __init__(self):
        self.best_loss_id3 = None

    # train bestlossID3 , X- data:train, Y- features
    def fit(self, X, Y, frac_param=0.4):
        train = X.sample(frac=frac_param, random_state=1234)  # random state is a seed value
        valid_group = X.drop(train.index)
        id3_curr = ID3class()
        id3_curr.fit(train, F)
        id3_curr.tree = self.prune(id3_curr.tree, valid_group, evaluate_func=self.evaluate)
        self.best_loss_id3 = id3_curr

    # test ImprovedKNNForest and return the true_positive rate, X- data:test
    def predict(self, X, k=1):
        X = X.copy()
        return self.best_loss_id3.predict_loss(E_test)

    # evaluate the loss of specific classification, FN = 10*FP
    @staticmethod
    def evaluate(real_classification, my_classification):
        if real_classification != my_classification:
            return 10 if real_classification == 'M' else 1  # he is sick
        return 0  # no False answer

    # algorithem to decide if its better to prune tree
    # for minimizing the loss as explained in the HW
    @staticmethod
    def prune(tree, validation_group, evaluate_func):
        if tree.children is None:  # it is a leave
            return tree
        # continues features, answers are binaries

        sub_validation_0 = validation_group[validation_group[tree.feature[0]] < tree.feature[1]]
        tree.children[0] = (0, CostSensitiveID3.prune(tree.children[0][1], sub_validation_0, evaluate_func))

        sub_validation_1 = validation_group[validation_group[tree.feature[0]] >= tree.feature[1]]
        tree.children[1] = (1, CostSensitiveID3.prune(tree.children[1][1], sub_validation_1, evaluate_func))

        err_prune = 0
        err_no_prune = 0

        for index, row in validation_group.iterrows():
            real_classification = row['diagnosis']
            err_prune += evaluate_func(real_classification, tree.class_of_node)
            err_no_prune += evaluate_func(real_classification, ID3class.classify(row, tree))

        if err_prune < err_no_prune:  # it is better to prune
            tree.f = None
            tree.children = None
        return tree


if __name__ == '__main__':
    E_train, F = utilis.createDF_train()
    E_test, F_test = utilis.createDF_test()

    best_id3_loss = CostSensitiveID3()
    best_id3_loss.fit(E_train, F)
    print(best_id3_loss.predict(E_test))
