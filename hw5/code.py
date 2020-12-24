from svmutil import *
import numpy as np
import argparse

'''Define Function'''


def label_filtering(Y, label):
    Y_tmp = np.zeros(Y.shape)
    Y_tmp[Y == label] = 1
    Y_tmp[Y != label] = -1
    return Y_tmp


def feature_processing(X):
    X_tmp = []
    for x in X:
        x_tmp = np.zeros(36)
        for i in x:
            x_tmp[i - 1] = x[i]
        X_tmp.append(x_tmp)
    return np.array(X_tmp)


def get_data(path):
    Y, X = svm_read_problem(path)
    return np.array(Y), feature_processing(X)


def main():
    '''Parsing'''
    parser = argparse.ArgumentParser(
        description='Argument Parser for ML HW5.')

    parser.add_argument('--tra_path', default='satimage.scale')
    parser.add_argument('--tst_path', default='satimage.scale.t')
    args = parser.parse_args()

    # load data
    Y_tra_mul, X_tra = get_data(args.tra_path)
    Y_tst_mul, X_tst = get_data(args.tst_path)

    '''Answer questions'''
    print('RUNNING Q15...')
    Y_tra = label_filtering(Y_tra_mul, 3)
    model = svm_train(Y_tra, X_tra, '-s 0 -t 0 -c 10 -q')
    alpha = np.array([a[0] for a in model.get_sv_coef()]).reshape(-1, 1)
    SV = feature_processing(model.get_SV())
    w = np.dot(SV.T, alpha)
    print('Answer of Q15 : {:.4f}\n'.format(np.linalg.norm(w)))

    print('RUNNING Q16...')
    SV_num_list = []
    min_Ein = 1.0
    min_cls = 1
    for l in [1, 2, 3, 4, 5]:
        Y_tra = label_filtering(Y_tra_mul, l)
        model = svm_train(Y_tra, X_tra, '-s 0 -t 1 -g 1 -r 1 -d 2 -c 10 -q')
        SV_num_list.append(model.get_nr_sv())
        _, pre_acc, _ = svm_predict(Y_tra, X_tra, model, '-q')
        Ein = 1 - 0.01 * pre_acc[0]
        if Ein < min_Ein:
            min_Ein = Ein
            min_cls = l
    print('Answer of Q16 : “{:d}” versus “not {:d}”\n'.format(
        min_cls, min_cls))

    print('RUNNING Q17...')
    print('Answer of Q17 : {:d}\n'.format(max(SV_num_list)))

    print('RUNNING Q18...')
    Y_tra = label_filtering(Y_tra_mul, 6)
    Y_tst = label_filtering(Y_tst_mul, 6)
    gamma = 10
    min_Eout = 1.0
    min_C = 1e-2
    for C in [1e-2, 1e-1, 1e0, 1e1, 1e2]:
        model = svm_train(
            Y_tra, X_tra, '-s 0 -t 2 -g {:f} -c {:f} -q'.format(gamma, C))
        _, pre_acc, _ = svm_predict(Y_tst, X_tst, model, '-q')
        Eout = 1 - 0.01 * pre_acc[0]
        if Eout < min_Eout:
            min_Eout = Eout
            min_C = C
    print('Answer of Q18 : {:.2f}\n'.format(min_C))

    print('RUNNING Q19...')
    C = 0.1
    min_Eout = 1.0
    min_gamma = 1e-1
    for gamma in [1e-1, 1e0, 1e1, 1e2, 1e3]:
        model = svm_train(
            Y_tra, X_tra, '-s 0 -t 2 -g {:f} -c {:f} -q'.format(gamma, C))
        _, pre_acc, _ = svm_predict(Y_tst, X_tst, model, '-q')
        Eout = 1 - 0.01 * pre_acc[0]
        if Eout < min_Eout:
            min_Eout = Eout
            min_gamma = gamma
    print('Answer of Q19 : {:.2f}\n'.format(min_gamma))

    print('RUNNING Q20...')
    gamma_list = [1e-1, 1e0, 1e1, 1e2, 1e3]
    best_counter = np.zeros(len(gamma_list))
    C = 0.1
    for _ in range(1000):
        randomlist = np.random.permutation(len(X_tra))
        min_Eout = 1.0
        min_gamma_idx = 0
        for i in range(len(gamma_list)):
            gamma = gamma_list[i]
            model = svm_train(Y_tra[randomlist[200:]], X_tra[randomlist[200:]],
                              '-s 0 -t 2 -g {:f} -c {:f} -q'.format(gamma, C))
            _, pre_acc, _ = svm_predict(
                Y_tra[randomlist[:200]], X_tra[randomlist[:200]], model, '-q')
            Eout = 1 - 0.01 * pre_acc[0]
            if Eout < min_Eout:
                min_Eout = Eout
                min_gamma_idx = i
        best_counter[min_gamma_idx] += 1
    print('Answer of Q20 : {:.2f}\n'.format(
        gamma_list[np.argmax(best_counter)]))


if __name__ == "__main__":
    main()
