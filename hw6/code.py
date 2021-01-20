import numpy as np
import argparse
import torch
import random
from joblib import Parallel, delayed

'''Define Function'''


def get_data(path):
    X = []
    for x in open(path, 'r'):
        x = x.strip().split(' ')
        X.append([float(v) for v in x])
    X = np.array(X)
    X, Y = np.array(X[:, :-1]), np.array(X[:, -1])
    return X, Y


def get_01_err(Y_pre, Y_tar):
    if len(Y_pre.shape) == 3:
        return (Y_pre != Y_tar).mean(axis=-1).mean()
    return np.mean(Y_pre != Y_tar)


class ThetaNode:
    def __init__(self, idx=None, theta=None, value=None):
        self.child_1st = None
        self.child_2nd = None
        self.idx = idx
        self.theta = theta
        self.value = value


class DecisionTree:
    def __init__(self, use_cuda=False, data_idx=None):
        self.tree = None
        self.use_cuda = use_cuda
        self.train_idx_list = data_idx

    def check_X_the_same(self, X):
        return (X != X[0, :]).sum() == 0

    def check_Y_the_same(self, Y):
        return (Y != Y[0]).sum() == 0

    def cuda(self):
        self.use_cuda = True

    def decision_stump(self, X, Y):
        X_sort, _ = torch.sort(X, dim=0)
        All_theta_feat = (X_sort[1:] + X_sort[:-1]) / 2

        X_rep = X.unsqueeze(0).repeat(All_theta_feat.shape[0], 1, 1)
        Y_rep = Y.unsqueeze(0).repeat(All_theta_feat.shape[0], 1).unsqueeze(
            0).repeat(All_theta_feat.shape[1], 1, 1)
        All_theta_feat = All_theta_feat.permute(1, 0).unsqueeze(-1)
        X_rep = X_rep.permute(2, 0, 1)
       
        pos_idx = X_rep > All_theta_feat
        neg_idx = X_rep <= All_theta_feat

        Gini_pos = self.Gini(pos_idx * Y_rep)
        Gini_neg = self.Gini(neg_idx * Y_rep)
        Gini_value = Gini_pos * pos_idx.sum(-1) + Gini_neg * neg_idx.sum(-1)

        min_gini_idx = (Gini_value == Gini_value.min()).nonzero(as_tuple=False)[0]
        idx = min_gini_idx[0].item()
        theta = All_theta_feat[idx, min_gini_idx[1].item()].item()

        return idx, theta

    def branching(self, X, Y):
        if self.check_Y_the_same(Y) or self.check_Y_the_same(X):
            return ThetaNode(value=Y[0].item())

        idx, theta = self.decision_stump(X, Y)
        tree_node = ThetaNode(idx, theta)

        idx_1st = X[:, idx] > theta
        idx_2nd = X[:, idx] <= theta

        tree_node.child_1st = self.branching(X[idx_1st], Y[idx_1st])
        tree_node.child_2nd = self.branching(X[idx_2nd], Y[idx_2nd])
        return tree_node

    def Gini(self, Y):
        N = (Y != 0).sum(-1)
        N[N == 0] = 1
        return 1 - (((Y == -1).sum(-1) / N)**2 + ((Y == 1).sum(-1) / N)**2)

    def fit(self, X, Y):
        with torch.no_grad():
            X, Y = torch.from_numpy(X), torch.from_numpy(Y)
            if self.use_cuda:
                X, Y = X.cuda(), Y.cuda()
            self.tree = self.branching(X, Y)

    def get_label(self, x, tree=None):
        if tree is None:
            tree = self.tree

        if tree.value is not None:
            return tree.value

        elif x[tree.idx] > tree.theta:
            return self.get_label(x, tree.child_1st)
        else:
            return self.get_label(x, tree.child_2nd)

    def predict(self, X):
        with torch.no_grad():
            return np.array([self.get_label(x, self.tree) for x in X])


class RandomForest:
    def __init__(self, n_estimators=2000, n_jobs=0, use_cuda=False):
        self.use_cuda = use_cuda
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def bootstrap(self, X, Y, sr=0.5):
        random_list = [random.randint(0, X.shape[0]-1)
                       for _ in range(int(sr * len(Y)))]
        return X[random_list], Y[random_list], sorted(list(set(random_list)))

    def get_tree(self, D):
        X, Y, data_idx = D
        tree = DecisionTree(use_cuda=self.use_cuda, data_idx=data_idx)
        tree.fit(X, Y)
        return tree

    def get_label(self, tree, X):
        Y = tree.predict(X)
        return Y

    def fit(self, X, Y, sr=0.5):
        if self.n_jobs != 0:
            self.tree_list = Parallel(n_jobs=self.n_jobs)(delayed(self.get_tree)(
                self.bootstrap(X, Y, sr)) for _ in range(self.n_estimators))
        else:
            self.tree_list = []
            for _ in range(self.n_estimators):
                self.tree_list.append(self.get_tree(self.bootstrap(X, Y, sr)))

    def predict(self, X, output_all=False):
        Y_pre_list = np.array([self.get_label(tree, X)
                               for tree in self.tree_list])
        if output_all:
            return Y_pre_list
        else:
            Y_pre = np.ones(X.shape[0])
            Y_pre[Y_pre_list.sum(axis=0) <= 0] = -1
            return Y_pre


def main():
    '''Parsing'''
    parser = argparse.ArgumentParser(
        description='Argument Parser for ML HW6.')

    parser.add_argument('--tra_path', default='hw6_train.dat')
    parser.add_argument('--tst_path', default='hw6_test.dat')
    args = parser.parse_args()

    # load data
    tra_X, tra_Y = get_data(args.tra_path)
    tst_X, tst_Y = get_data(args.tst_path)

    '''Answer questions'''
    print('RUNNING Q14...')
    dt = DecisionTree(use_cuda=True)
    dt.fit(tra_X, tra_Y)
    tst_Y_pre = dt.predict(tst_X)
    print('Answer of Q14 : {:.4f}\n'.format(get_01_err(tst_Y_pre, tst_Y)))

    print('RUNNING Q15...')
    rf = RandomForest(n_estimators=2000, n_jobs=8, use_cuda=True)
    rf.fit(tra_X, tra_Y)
    tst_Y_pre = rf.predict(tst_X, output_all=True)
    print('Answer of Q15 : {:.4f}\n'.format(get_01_err(tst_Y_pre, tst_Y)))

    print('RUNNING Q16...')
    tra_Y_pre = rf.predict(tra_X, output_all=False)
    print('Answer of Q16 : {:.4f}\n'.format(get_01_err(tra_Y_pre, tra_Y)))

    print('RUNNING Q17...')
    tst_Y_pre = rf.predict(tst_X, output_all=False)
    print('Answer of Q17 : {:.4f}\n'.format(get_01_err(tst_Y_pre, tst_Y)))

    print('RUNNING Q18...')
    val_Y_pre = []
    for i in range(tra_X.shape[0]):
        tree_list = []
        for tree in rf.tree_list:
            if i not in tree.train_idx_list:
                tree_list.append(tree)

        if len(tree_list) == 0:
            y_pre = -1
        else:
            y_pre_list = []
            for tree in tree_list:
                y_pre_list.append(tree.get_label(tra_X[i]))
            y_pre = 1 if sum(y_pre_list) > 0 else -1

        val_Y_pre.append(y_pre)
    val_Y_pre = np.array(val_Y_pre)
    print('Answer of Q18 : {:.4f}\n'.format(get_01_err(val_Y_pre, tra_Y)))


if __name__ == "__main__":
    main()
