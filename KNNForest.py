from ID3 import ID3 as ID3class
import utilis
import random as rnd
import numpy as np
import argparse

class KNNForest:
    def __init__(self):
        self.id3_lst = []
        self.centroid_lst = []
        self.N = 1
        self.K = 1
        self.train_min = None
        self.train_max = None
        self.divide_by = None

    # train KNNForest , X- data:train, Y- features
    def fit(self, E_train, F_train, p_param=0.3, N_param=15):
        self.N = N_param
        n = E_train.shape[0]
        sub_sample_num = round(p_param * n)
        choose_lst = range(n)
        train_for_minmax = E_train.loc[:, E_train.columns != 'diagnosis']
        self.train_min = train_for_minmax.min()
        self.train_max = train_for_minmax.max()
        divide_by = self.train_max - self.train_min
        divide_by = divide_by.replace(0, 1)  # so we wont divide by zero
        self.divide_by = divide_by
        for it in range(self.N):
            sampling_indices = rnd.sample(choose_lst, k=sub_sample_num)  # randomized samples from training set
            curr_train = E_train.iloc[sampling_indices]
            id3_curr = ID3class()
            id3_curr.fit(curr_train, F_train)
            self.id3_lst.append(id3_curr)
            # calculate centroid
            curr_centroid_not_minmax = self.calc_centroid(curr_train)
            curr_centroid_minmax = self.minmax(curr_centroid_not_minmax)
            # append centroid
            self.centroid_lst.append(curr_centroid_minmax)

    # minmax the centroid
    def minmax(self, centroid_not_minmax):
        minmax_centroid = (centroid_not_minmax - self.train_min) / self.divide_by
        return minmax_centroid

    # test KNNForest and return the true_positive rate, X- data:test
    def predict(self, X, k_param=9):
        size_X = X.shape[0]
        if size_X == 0:
            raise Exception('There is no test, you can go home')
        cnt_true_positive = 0
        for index, row in X.iterrows():
            if self.classify_by_k(row, k_param) == row['diagnosis']:
                cnt_true_positive += 1
        return cnt_true_positive / size_X

    # calculate centroid
    @staticmethod
    def calc_centroid(df):
        df_relevant = df.loc[:, df.columns != 'diagnosis']  # we need only features
        centroid = df.mean()

        return centroid

    # calculate distance (Euclidean) between centroids
    @staticmethod
    def calc_distance(centroid1, centroid2):
        dist = np.linalg.norm(centroid1 - centroid2)
        return dist

    # classify sample by k closest trees (default k=15) and return
    def classify_by_k(self, sample, k_param=1):
        sample_to_cenroid = self.minmax(sample.drop(['diagnosis']))  # we need only features
        self.K = k_param
        distance_np = np.array([self.calc_distance(self.centroid_lst[it].to_numpy(), sample_to_cenroid.to_numpy())
                                for it in range(self.N)])

        def get_indices_of_k_smallest(arr, k):  # finds the k indices of k smallest values (distance)
            if k == arr.size:  # all the trees can vote
                return range(k)
            idx = np.argpartition(arr, k)
            # return np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))].tolist()
            return idx[:k]

        conference_indices_lst = get_indices_of_k_smallest(distance_np, self.K)
        classify_func = lambda i: ID3class.classify(sample, self.id3_lst[conference_indices_lst[i]].tree)
        votes_lst = [classify_func(i) for i in range(self.K)]
        if votes_lst.count('B') > len(votes_lst) / 2:  # B won the vote
            return 'B'
        return 'M'


if __name__ == '__main__':
    # params are N- total number of decision tree, K - votes number , p - between [0.3,0.7]
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', default=15, type=int,
                        help='N is the numbers of trees in the conference , (not all of them are voting).'
                             'N is bigger than zero', )

    parser.add_argument('-K', default=9, type=int,
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

    knn_forest = KNNForest()
    knn_forest.fit(E_train, F, p, N)
    print(knn_forest.predict(E_test, K))

