from liblinearutil import *
import numpy as np
import argparse

'''Define Function'''


def Q_transform(X):
    X_scnd = []
    for i in range(1, X.shape[1]):
        for j in range(i, X.shape[1]):
            X_scnd.append(X[:, i] * X[:, j])
    X_scnd = np.array(X_scnd).T
    return np.hstack((X, X_scnd))


def get_data(path, bias=1.0, transform=None):
    X = []
    for x in open(path, 'r'):
        x = x.strip().split(' ')
        x = [float(v) for v in x]
        X.append([bias] + x)

    X = np.array(X)
    X, Y = np.array(X[:, :-1]), np.array(X[:, -1])

    if transform is not None:
        X = transform(X)

    return X, Y


def main():
    '''Parsing'''
    parser = argparse.ArgumentParser(
        description='Argument Parser for MLF HW4.')

    parser.add_argument('--tra_path', default='hw4_train.dat')
    parser.add_argument('--tst_path', default='hw4_test.dat')
    args = parser.parse_args()

    # load data
    X_tra, Y_tra = get_data(path=args.tra_path, transform=Q_transform)
    X_tst, Y_tst = get_data(path=args.tst_path, transform=Q_transform)

    log10_lambda_choices = [-4, -2, 0, 2, 4]
    lambda_choices = [10**lmd for lmd in log10_lambda_choices]

    '''Answer questions'''
    print('RUNNING Q16...')
    best_log_lmd = 0
    max_acc = 0
    for i in range(len(lambda_choices)):
        lmd = lambda_choices[i]
        model = train(
            Y_tra, X_tra, '-s 0 -c {:f} -e 0.000001 -q'.format(1 / (2*lmd)))
        _, pre_acc, _ = predict(Y_tst, X_tst, model, '-q')
        if pre_acc[0] >= max_acc:
            best_log_lmd = log10_lambda_choices[i]
            max_acc = pre_acc[0]
    print('Answer of Q16 : {:2d}\n'.format(best_log_lmd))

    print('RUNNING Q17...')
    best_log_lmd = 0
    max_acc = 0
    for i in range(len(lambda_choices)):
        lmd = lambda_choices[i]
        model = train(
            Y_tra, X_tra, '-s 0 -c {:f} -e 0.000001 -q'.format(1 / (2*lmd)))
        _, pre_acc, _ = predict(Y_tra, X_tra, model, '-q')
        if pre_acc[0] >= max_acc:
            best_log_lmd = log10_lambda_choices[i]
            max_acc = pre_acc[0]
    print('Answer of Q17 : {:2d}\n'.format(best_log_lmd))

    print('RUNNING Q18...')
    best_lmd_idx = 0
    max_acc = 0
    for i in range(len(lambda_choices)):
        lmd = lambda_choices[i]
        model = train(Y_tra[:120], X_tra[:120],
                      '-s 0 -c {:f} -e 0.000001 -q'.format(1 / (2*lmd)))
        _, pre_acc, _ = predict(
            Y_tra[120:], X_tra[120:], model, '-q')
        if pre_acc[0] >= max_acc:
            best_lmd_idx = i
            max_acc = pre_acc[0]
    model = train(Y_tra[:120], X_tra[:120],
                  '-s 0 -c {:f} -e 0.000001 -q'.format(1 / (2*lambda_choices[best_lmd_idx])))
    _, pre_acc, _ = predict(Y_tst, X_tst, model, '-q')
    print('Answer of Q18 : {:.4f}\n'.format((100 - pre_acc[0]) * 0.01))

    print('RUNNING Q19...')
    model = train(Y_tra, X_tra,
                  '-s 0 -c {:f} -e 0.000001 -q'.format(1 / (2*lambda_choices[best_lmd_idx])))
    _, pre_acc, _ = predict(Y_tst, X_tst, model, '-q')
    print('Answer of Q19 : {:.4f}\n'.format((100 - pre_acc[0]) * 0.01))

    print('RUNNING Q20...')
    folds_num = 5
    X_folds = np.vsplit(X_tra, folds_num)
    Y_folds = np.hsplit(Y_tra, folds_num)
    best_lmd_idx = 0
    max_acc = 0
    for i in range(len(lambda_choices)):
        lmd = lambda_choices[i]
        acc_list = []
        for fold_idx in range(folds_num):
            X_tra_cv = np.vstack([X_folds[f_idx]
                                  for f_idx in range(folds_num) if fold_idx != f_idx])
            Y_tra_cv = np.hstack([Y_folds[f_idx]
                                  for f_idx in range(folds_num) if fold_idx != f_idx])
            X_val_cv = X_folds[fold_idx]
            Y_val_cv = Y_folds[fold_idx]
            model = train(Y_tra_cv, X_tra_cv,
                          '-s 0 -c {:f} -e 0.000001 -q'.format(1 / (2*lmd)))
            _, pre_acc, _ = predict(Y_val_cv, X_val_cv, model, '-q')
            acc_list.append(pre_acc[0])
        acc = np.mean(acc_list)
        if acc >= max_acc:
            best_lmd_idx = i
            max_acc = acc
    print('Answer of Q20 : {:.4f}\n'.format((100 - max_acc) * 0.01))


if __name__ == "__main__":
    main()
