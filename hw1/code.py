import numpy as np
import random
import argparse

def main():
    '''
    Parsing
    '''
    parser = argparse.ArgumentParser(
            description='Argument Parser for MLF HW1.')
    parser.add_argument('--path', default='hw1_train.dat')
    args = parser.parse_args()

    '''
    Define Function
    '''
    def get_data(path, bias=1.0, scale=1.0):
        X = []
        for x in open(path, 'r'):
            x = x.strip().split('\t')
            x = [float(v) for v in x]
            X.append([bias] + x)

        X = np.array(X)
        X, Y = np.array(X[:, :-1]), np.array(X[:, -1])
        return X * scale, Y
        
    def update_w(w, x, y):
        result = np.sign(np.dot(w, x) * y)
        if result <= 0:
            return w + x * y, False
        else:
            return w, True

    def random_pick(X, Y):
        idx = random.randint(0, X.shape[0] - 1)
        return X[idx], Y[idx]
            
    def PLA(X, Y):
        # initialization
        counter = 0
        update_num = 0
        N = X.shape[0]
        w = np.zeros(X.shape[1:])
        
        # training
        while counter < 5 * N:
            x, y = random_pick(X, Y)
            w, Iscorrect = update_w(w, x, y)
            
            if Iscorrect:
                counter += 1
            else:
                counter = 0
                update_num += 1
                
        return w, update_num

    # load data
    X, Y = get_data(args.path)

    '''
    Answer questions
    '''
    # Q16. Repeat your experiment for 1000 times, each with a different random seed. 
    # What is the median number of updates before the algorithm returns wPLA? Choose the closest value.
    print('RUNNING Q16...')
    update_num_list = []
    for i in range(1000):
        _, update_num = PLA(X, Y)
        update_num_list.append(update_num)
    print('Answer of Q16 :', np.median(update_num_list))

    # Q17. Among all the w0 (the zero-th component of wPLA) obtained from the 1000 experiments above,
    # what is the median? Choose the closest value.
    print('RUNNING Q17...')
    w0_list = []
    for i in range(1000):
        w_PLA, _ = PLA(X, Y)
        w0_list.append(w_PLA[0])
    print('Answer of Q17 :', np.median(w0_list))

    # Q18. Set x0 = 10 to every xn instead of x0 = 1, and repeat the 1000 experiments above. 
    # What is the median number of updates before the algorithm returns wPLA? Choose the closest value.
    print('RUNNING Q18...')
    X, Y = get_data('hw1_train.dat', 10.0)
    update_num_list = []
    for i in range(1000):
        _, update_num = PLA(X, Y)
        update_num_list.append(update_num)
    print('Answer of Q18 :', np.median(update_num_list))

    # Q19. Set x0 = 0 to every xn instead of x0 = 1. This equivalently means not adding any x0, and you
    # will get a separating hyperplane that passes the origin. Repeat the 1000 experiments above. 
    # What is the median number of updates before the algorithm returns wPLA?
    print('RUNNING Q19...')
    X, Y = get_data('hw1_train.dat', 0.0)
    update_num_list = []
    for i in range(1000):
        _, update_num = PLA(X, Y)
        update_num_list.append(update_num)
    print('Answer of Q19 :', np.median(update_num_list))

    # Now, in addition to setting x0 = 0 to every xn, scale down each xn by 4. Repeat the 1000 experiments above. 
    # What is the median number of updates before the algorithm returns wPLA? Choose the closest value.
    print('RUNNING Q20...')
    X, Y = get_data('hw1_train.dat', bias=0.0, scale=0.25)
    update_num_list = []
    for i in range(1000):
        _, update_num = PLA(X, Y)
        update_num_list.append(update_num)
    print('Answer of Q20 :', np.median(update_num_list))

if __name__ == "__main__":
    main()
