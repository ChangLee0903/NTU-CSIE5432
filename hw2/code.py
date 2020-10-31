import numpy as np
import random
from scipy.stats import bernoulli
import argparse

'''
Define Function
'''


def generate_data(size, tau):
    x = np.sort(np.random.uniform(-1, 1, size))
    y = np.zeros(size).astype(int)
    y[x > 0] = 1
    y[x <= 0] = -1

    noisy_idx = bernoulli.rvs(tau, size=size) > 0
    y[noisy_idx] = -y[noisy_idx]

    return x, y


def get_g_Ein(x, y):
    theta_set = ([-1] + list(((x[1:] + x[:-1]) / 2))) * 2
    s_set = [-1] * len(x) + [1] * len(x)
    hypothesis_set = np.array(
        sorted(tuple(zip(s_set, theta_set)), key=lambda x: x[0] + x[1]))
    g, Ein = (-1, -1), 1

    for hypothesis in hypothesis_set:
        err = get_err(hypothesis, x, y)
        if Ein > err:
            g, Ein = hypothesis, err
        if Ein == 0:
            break
    return g, Ein


def get_err(hypothesis, x, y):
    s, theta = hypothesis
    pred = np.ones(len(y)).astype(int)

    if s > 0:
        pred[x <= theta] = -1
    else:
        pred[x > theta] = -1

    return (y != pred).sum() / len(y)


def get_Eout(hypothesis, tau, IsSimulate=False):
    if IsSimulate:
        x_tst, y_tst = generate_data(100000, tau)
        Eout = get_err(hypothesis, x_tst, y_tst)
        return Eout
    else:
        s, theta = hypothesis
        Eout = 0.5 * np.abs(theta) if s > 0 else 1 - 0.5 * np.abs(theta)
        return (1 - 2 * tau) * Eout + tau


def get_answer(exp_num, size, tau, IsSimulate=False):
    ans = []

    for _ in range(exp_num):
        random.seed(random.randint(1, 10000))
        x_tra, y_tra = generate_data(size, tau)

        g, Ein = get_g_Ein(x_tra, y_tra)
        Eout = get_Eout(g, tau, IsSimulate)
        ans.append(Eout - Ein)

    return np.mean(ans)


def main():
    '''
    Parsing
    '''
    parser = argparse.ArgumentParser(
        description='Argument Parser for MLF HW1.')
    parser.add_argument('--mode', default='closedform',
                        choices=['closedform', 'simulate'])
    args = parser.parse_args()
    if args.mode == 'simulate':
        is_simulate = True
        print("Tesing by simulation!")
    elif args.mode == 'closedform':
        is_simulate = False
        print("Tesing by closed form!")

    '''
    Answer questions
    '''

    print('RUNNING Q16...')
    print('Answer of Q16 : {:.4f}\n'.format(get_answer(
        exp_num=10000, size=2, tau=0, IsSimulate=is_simulate)))

    print('RUNNING Q17...')
    print('Answer of Q17 : {:.4f}\n'.format(get_answer(
        exp_num=10000, size=20, tau=0, IsSimulate=is_simulate)))

    print('RUNNING Q18...')
    print('Answer of Q18 : {:.4f}\n'.format(get_answer(
        exp_num=10000, size=2, tau=0.1, IsSimulate=is_simulate)))

    print('RUNNING Q19...')
    print('Answer of Q19 : {:.4f}\n'.format(get_answer(
        exp_num=10000, size=20, tau=0.1, IsSimulate=is_simulate)))

    print('RUNNING Q20...')
    print('Answer of Q20 : {:.4f}\n'.format(get_answer(
        exp_num=10000, size=200, tau=0.1, IsSimulate=is_simulate)))


if __name__ == "__main__":
    main()
