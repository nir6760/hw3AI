"""class of ID3 algo.
"""
import utilis
import math


class Tree:

    def __init__(self, f, childs, default_val):
        self.feature = f
        self.children = childs
        self.class_of_node = default_val


class ID3:
    def __init__(self):
        self.tree = None

    # train ID3 , X- data:train, Y- features
    def fit(self, X, Y):
        c = utilis.majority_class(X)
        self.tree = self.TDIDT(X, Y, c, ID3.MaxIG)

    # test ID3 and return the true_positive rate, X- data:test
    def predict(self, X):
        size_X = X.shape[0]
        if size_X == 0:
            raise Exception('There is no test, you can go home')
        cnt_true_positive = 0
        for index, row in X.iterrows():
            if ID3.classify(row, self.tree) == row['diagnosis']:
                cnt_true_positive += 1
        return cnt_true_positive / size_X

    # classify object
    @staticmethod
    def classify(object, tree):
        if tree is None:
            raise Exception("my friend, you need to train me first")
        if tree.children is None:
            return tree.class_of_node

        attribute = tree.feature[0]
        thres = tree.feature[1]
        #print('attribute ', attribute , ' - threshold ', thres)

        going_to = 0
        if object[attribute] >= thres:  # feature is 0
            going_to = 1

        # child[0] - val of feature, child[1] - subtree
        next_subtree = tree.children[going_to][1]
        return ID3.classify(object, next_subtree)

    # make the discrete functions, for dynamic partition
    @staticmethod
    def make_discrete(f, E):
        best_thres = -1
        best_ig = -1
        val_f_list = list(dict.fromkeys(E[f]))  # remove duplicates from list
        val_f_list.sort()
        for i in range(1, len(val_f_list)):
            thres = (val_f_list[i - 1] + val_f_list[i]) / 2
            f_disc = [f, thres]
            ig = ID3.IG(f_disc, E)
            if ig >= best_ig:
                best_thres = thres
                best_ig = ig

        return best_ig, best_thres

    # function which select the best feature for current node in ID3 Tree
    # select with dynamic partition
    @staticmethod
    def MaxIG(F, E):
        best_f = ['Default', -1]
        best_ig = -1
        for f in F:
            curr_f_best_ig, curr_f_best_thres = ID3.make_discrete(f, E)
            if curr_f_best_ig >= best_ig:
                best_f = [f, curr_f_best_thres]
                best_ig = curr_f_best_ig

        return best_f  # argmax

    # find the information gain of feature f in Examples
    @staticmethod
    def IG(f, Examples):
        examples0 = Examples[Examples[f[0]] >= f[1]]
        size_examples0 = examples0.shape[0]

        examples1 = Examples[Examples[f[0]] < f[1]]
        size_examples1 = examples1.shape[0]

        size_examples = Examples.shape[0]
        ig = ID3.H(Examples) - ((size_examples0 * ID3.H(examples0) +
                                 size_examples1 * ID3.H(examples1))
                                / size_examples)
        return ig

    # find the Entropy of E
    @staticmethod
    def H(E):
        size_examples = E.shape[0]
        # if size_examples == 0:
        #    return 0
        p_B = E[E['diagnosis'] == 'B'].shape[0] / size_examples
        p_M = E[E['diagnosis'] == 'M'].shape[0] / size_examples

        if p_B == 0 or p_M == 0:
            return 0

        h = -p_B * math.log(p_B, 2) - p_M * math.log(p_M, 2)
        return h

    # create TDIDT Tree from givev E and F with select feature function
    @staticmethod
    def TDIDT(E, F, Default, SelectFeature):
        if E.empty:  # there is no examples
            return Tree(None, None, Default)
        c = utilis.majority_class(E)
        if utilis.is_node(E, F, c):
            return Tree(None, None, c)
        f = SelectFeature(F, E)

        #F.remove(f[0])
        # F = delete_from_features(F, f) # on discrete cases

        subexamples0 = E[E[f[0]] < f[1]]
        subexamples1 = E[E[f[0]] >= f[1]]
        child0 = [0, ID3.TDIDT(subexamples0, F, c, SelectFeature)]
        child1 = [1, ID3.TDIDT(subexamples1, F, c, SelectFeature)]
        subtrees = [child0, child1]
        return Tree(f, subtrees, c)


if __name__ == '__main__':
    E_train, F = utilis.createDF_train()
    E_test, F = utilis.createDF_test()
    id3 = ID3()
    id3.fit(E_train, F)
    print(id3.predict(E_test))
