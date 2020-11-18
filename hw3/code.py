import numpy as np
import random
import argparse

'''Define Function'''


def get_data(path, bias=1.0, transform=None):
    X = []
    for x in open(path, 'r'):
        x = x.strip().split('\t')
        x = [float(v) for v in x]
        X.append([bias] + x)

    X = np.array(X)
    X, Y = np.array(X[:, :-1]), np.array(X[:, -1])

    if transform is not None:
        X = transform(X)

    return X, Y


def get_wLIN(X, Y):
    X_plus = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
    return np.matmul(X_plus, Y)


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sign(s):
    s = np.sign(s)
    s[s == 0] = -1
    return s


def Q_transform(X, Q=3):
    return np.hstack([X]+[X[:, 1:]**q for q in range(2, Q+1)])


def err(w, X, Y, mode='sqr'):
    Y_pred = np.matmul(X, w)
    if mode == 'sqr':
        return ((Y_pred - Y)**2).mean()
    elif mode == 'ce':
        return -np.log(sigmoid(Y * Y_pred)).mean()
    elif mode == '0/1':
        Y_pred = sign(Y_pred)
        return (Y.astype(int) != Y_pred.astype(int)).mean()


def SGD(X, Y, lr, w_init=None, step_num=1000000, mode='sqr'):
    def random_pick(X, Y):
        idx = random.randint(0, X.shape[0] - 1)
        return X[idx:idx+1], Y[idx:idx+1]

    def grad_func(w, X, Y, mode):
        batch_size = X.shape[0]
        if mode == 'sqr':
            return -(2 / batch_size) * np.matmul(X.T, np.matmul(X, w) - Y)
        elif mode == 'ce':
            return np.mean(sigmoid(-Y * np.matmul(X, w)).reshape(-1, 1) * (Y.reshape(-1, 1) * X), axis=0)

    def update_w(w, x, y, lr):
        return w + lr * grad_func(w, x, y, mode)

    if mode == 'sqr':
        wLIN = get_wLIN(X, Y)
        E_in_sqr_LIN = err(wLIN, X, Y, mode='sqr')

    # initialization
    step = 0
    w = np.zeros(X.shape[1:]) if w_init is None else w_init

    # training
    while step < step_num:
        x, y = random_pick(X, Y)
        w = update_w(w, x, y, lr)
        step += 1

        # check early stopping
        if mode == 'sqr':
            E_in_sqr = err(w, X, Y, mode='sqr')
            if E_in_sqr <= 1.01 * E_in_sqr_LIN:
                break
    return w, step


def main():
    '''Parsing'''
    parser = argparse.ArgumentParser(
        description='Argument Parser for MLF HW3.')

    parser.add_argument('--tra_path', default='hw3_train.dat')
    parser.add_argument('--tst_path', default='hw3_test.dat')
    args = parser.parse_args()

    # load data
    X_tra, Y_tra = get_data(args.tra_path)
    X_tst, Y_tst = get_data(args.tst_path)

    '''Answer questions'''
    print('RUNNING Q14...')
    wLIN = get_wLIN(X_tra, Y_tra)
    print('Answer of Q14 : {:.4f}\n'.format(
        err(wLIN, X_tra, Y_tra, mode='sqr')))

    print('RUNNING Q15...')
    update_num_list = []
    for _ in range(1000):
        _, update_num = SGD(X_tra, Y_tra, lr=0.001)
        update_num_list.append(update_num)
    print('Answer of Q15 : {:.4f}\n'.format(np.mean(update_num_list)))

    print('RUNNING Q16...')
    ce_loss_list = []
    for _ in range(1000):
        w, _ = SGD(X_tra, Y_tra, lr=0.001, step_num=500, mode='ce')
        ce_loss = err(w, X_tra, Y_tra, mode='ce')
        ce_loss_list.append(ce_loss)
    print('Answer of Q16 : {:.4f}\n'.format(np.mean(ce_loss_list)))

    print('RUNNING Q17...')
    wLIN = get_wLIN(X_tra, Y_tra)
    ce_loss_list = []
    for _ in range(1000):
        w, _ = SGD(X_tra, Y_tra, lr=0.001, w_init=wLIN,
                   step_num=500, mode='ce')
        ce_loss = err(w, X_tra, Y_tra, mode='ce')
        ce_loss_list.append(ce_loss)
    print('Answer of Q17 : {:.4f}\n'.format(np.mean(ce_loss_list)))

    print('RUNNING Q18...')
    print('Answer of Q18 : {:.4f}\n'.format(
        abs(err(wLIN, X_tst, Y_tst, mode='0/1') - err(wLIN, X_tra, Y_tra, mode='0/1'))))

    print('RUNNING Q19...')
    X_tra_Q = Q_transform(X_tra, Q=3)
    X_tst_Q = Q_transform(X_tst, Q=3)
    wLIN_Q = get_wLIN(X_tra_Q, Y_tra)
    print('Answer of Q19 : {:.4f}\n'.format(abs(
        err(wLIN_Q, X_tst_Q, Y_tst, mode='0/1') - err(wLIN_Q, X_tra_Q, Y_tra, mode='0/1'))))

    print('RUNNING Q20...')
    X_tra_Q = Q_transform(X_tra, Q=10)
    X_tst_Q = Q_transform(X_tst, Q=10)
    wLIN_Q = get_wLIN(X_tra_Q, Y_tra)
    print('Answer of Q20 : {:.4f}\n'.format(abs(
        err(wLIN_Q, X_tst_Q, Y_tst, mode='0/1') - err(wLIN_Q, X_tra_Q, Y_tra, mode='0/1'))))


if __name__ == "__main__":
    main()
