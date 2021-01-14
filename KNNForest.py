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

    # train KNNForest , X- data:train, Y- features
    def fit(self, E_train, F_train, p=0.5, N=1):
        self.N = N
        n = E_train.shape[0]
        sub_sample_num = round(p * n)
        choose_lst = range(n)
        for it in range(self.N):
            sampling_indices = rnd.choices(choose_lst, k=sub_sample_num)
            curr_train = E_train.loc[sampling_indices]
            id3_curr = ID3class()
            id3_curr.fit(curr_train, F_train)
            self.id3_lst.append(id3_curr)
            # calculate centroid
            # curr_train = curr_train.loc[:, curr_train.columns != 'diagnosis']  # don't need it in the centroid
            curr_centroid = self.calc_centroid(curr_train)
            # append centroid
            self.centroid_lst.append(curr_centroid)

    # test KNNForest and return the true_positive rate, X- data:test
    def predict(self, X, k=1):
        size_X = X.shape[0]
        if size_X == 0:
            raise Exception('There is no test, you can go home')
        cnt_true_positive = 0
        for index, row in X.iterrows():
            if self.classify_by_k(row, k) == row['diagnosis']:
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

    # classify sample by k closest trees (default k=1) and return
    def classify_by_k(self, sample, k=1):
        sample_to_cenroid = sample.drop(['diagnosis'])  # we need only features
        self.K = k
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
    parser.add_argument('-N', default=10, type=int,
                        help='N is the numbers of trees in the conference , (not all of them are voting).'
                             'N is bigger than zero', )

    parser.add_argument('-K', default=3, type=int,
                        help='K is the number of voting trees, K between 1 to N', )
    parser.add_argument('-p', default=0.5, type=float,
                        help='p is parameter between [0.3,0.7] for learning choice', )
    args = parser.parse_args()

    if args.N < 0 or args.K < 0 or args.K > args.N or args.p < 0.3 or args.p > 0.7:
        raise Exception("Invalid parameters, try again and call for help")

    N = args.N
    K = args.K
    p = args.p
    E_train, F = utilis.createDF_train()
    E_test, F = utilis.createDF_test()

    knn_forest = KNNForest()
    knn_forest.fit(E_train, F, p, N)
    print(knn_forest.predict(E_test, K))
