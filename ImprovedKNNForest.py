from CostSensitiveID3 import CostSensitiveID3 as costSensitiveClass
from ID3 import ID3 as ID3class
import utilis
import argparse
import numpy as np

# evaluate predict error (FN=1=FP)
from KNNForest import KNNForest


class ImprovedKNNForest:
    def __init__(self):
        self.improved_forest = None
        self.best_features = []
        self.best_features_with_diagnosis = []

    # train ImprovedKNNForest , X- data:train, Y- features
    def fit(self, E_training, F_training, evaluatae_func=None, p_param=0.3, N_param=15, num_feauters_param=11, ):
        E_training = E_training.copy()
        F_training = F_training.copy()
        self.best_features = findBestFeaturesByIG(num_feauters_param, E_training, F_training)
        self.best_features_with_diagnosis = self.best_features.copy()
        self.best_features_with_diagnosis.insert(0, 'diagnosis')
        E_train_best_with_diagnosis = E_train[self.best_features_with_diagnosis]

        train = E_train_best_with_diagnosis.sample(frac=0.8, random_state=1234)  # random state is a seed value
        valid_group = E_train_best_with_diagnosis.drop(train.index)

        my_knn_forest = KNNForest()
        my_knn_forest.fit(train, self.best_features, p_param, N_param)
        #my_knn_forest.fit(E_train_best_with_diagnosis, self.best_features, p_param, N_param)

        for it in range(my_knn_forest.N):
            my_knn_forest.id3_lst[it].tree = \
                costSensitiveClass.prune(my_knn_forest.id3_lst[it].tree, valid_group, evaluate_func=evaluatae_func)

        self.improved_forest = my_knn_forest


    # test ImprovedKNNForest and return the true_positive rate, X- data:test
    def predict(self, X, k_param=15):
        X = X.copy()
        E_test_best_improved = X[self.best_features_with_diagnosis]
        size_E_test_best_improved = E_test_best_improved.shape[0]
        if size_E_test_best_improved == 0:
            raise Exception('There is no test, you can go home')
        cnt_true_positive = 0
        for index, row in E_test_best_improved.iterrows():
            if self.classify_by_k(row, k_param) == row['diagnosis']:
                cnt_true_positive += 1
        return cnt_true_positive / size_E_test_best_improved

    # classify sample by k closest trees (default k=15) and return
    def classify_by_k(self, sample, k_param=6):
        sample_to_cenroid = self.improved_forest.minmax(sample.drop(['diagnosis']))  # we need only features
        self.improved_forest.K = k_param
        distance_np = np.array(
            [self.improved_forest.calc_distance(self.improved_forest.centroid_lst[it].to_numpy(),
                                                sample_to_cenroid.to_numpy())
             for it in range(self.improved_forest.N)])

        def get_indices_of_k_smallest(arr, k):  # finds the k indices of k smallest values (distance)
            if k == arr.size:  # all the trees can vote
                return range(k)
            #idx = np.argpartition(arr, k)
            idx = sorted(range(len(arr)), key=lambda sub: arr[sub])[:K]
            # return np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))].tolist()
            return idx[:k]

        conference_indices_lst = get_indices_of_k_smallest(distance_np, self.improved_forest.K)
        classify_func = lambda i: ID3class.classify(sample,
                                                    self.improved_forest.id3_lst[conference_indices_lst[i]].tree)

        dec_by = int(self.improved_forest.K / 3)
        votes_lst = [[classify_func(i)] * (3 - int(i /2)) for i in
                     range(self.improved_forest.K)]
        votes_lst_flatt = [item for sublist in votes_lst for item in sublist]

        if votes_lst_flatt.count('B') > len(votes_lst) / 2:  # B won the vote
            return 'B'
        return 'M'


# return 1 if wrong classification occurred
def evaluate_predict_err(real_classification, my_classification):
    if real_classification != my_classification:
        return 1
    return 0  # no False answer


# find the param best feauters which have the bigger IG
def findBestFeaturesByIG(param, E_train, F_intial):
    F_best_list = []
    for it in range(param):
        best_f = ID3class.MaxIG(F_intial, E_train)
        F_best_list.append(best_f[0])
        F_intial.remove(best_f[0])
    return F_best_list


if __name__ == '__main__':
    # params are N- total number of decision tree, K - votes number , p - between [0.3,0.7]
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', default=15, type=int,
                        help='N is the numbers of trees in the conference , (not all of them are voting).'
                             'N is bigger than zero', )

    parser.add_argument('-K', default=6, type=int,
                        help='K is the number of voting trees, K between 1 to N', )
    parser.add_argument('-p', default=0.3, type=float,
                        help='p is parameter between [0.3,0.7] for learning choice', )
    args = parser.parse_args()

    if args.N < 0 or args.K < 0 or args.K > args.N or args.p < 0.3 or args.p > 0.7:
        raise Exception("Invalid parameters, try again and call for help")

    N = args.N
    K = args.K
    p = args.p
    E_train, F = utilis.createDF_train()
    E_test, F_test = utilis.createDF_test()

    improved_knn_forest = ImprovedKNNForest()
    improved_knn_forest.fit(E_train, F, evaluate_predict_err, p, N)
    print(improved_knn_forest.predict(E_test, K))

'''
    regular = []
    improved = []

    for i in range(5):
        knn_forest1 = KNNForest()
        knn_forest1.fit(E_train, F, p, N)
        r = knn_forest1.predict(E_test, K)
        regular.append(r)

        knn_forest2 = ImprovedKNNForest()
        knn_forest2.fit(E_train, F, evaluate_predict_err, p, N)
        im = knn_forest2.predict(E_test, K)
        print(i, im)
        improved.append(im)

    x_lst = range(5)
    line1, = plt.plot(x_lst, regular, 'bo', )
    plt.plot(x_lst, regular, 'b')
    line2, = plt.plot(x_lst, improved, 'ro')
    plt.plot(x_lst, improved, 'r')
    line1.set_label('KnnForest - sec 6')
    line2.set_label('ImprovedKnnForest - sec 7')
    # naming the x axis
    plt.xlabel('experiment num - axis')
    # naming the y axis
    plt.ylabel('predication rate- axis')
    plt.title('Section 7')
    plt.legend()
    plt.show()
'''
