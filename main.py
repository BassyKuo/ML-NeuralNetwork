"""
    Course:	EE6550, Machine Learning, Spring 2017
    Homework:	#5
    Implement:	Neural Network by the stochastic gradient descent (SGD) algorithm
    Python Version: python3 (3.5.0)
    Usage:
        python3 HW5_105062633_xinyukuo_python35.py
    Parameters:
        [Input]
            trainset
                - training data file (.csv)
            testset
                - testing data file (.csv)
            n {5 | 10}
                - n-fold cross-validation (choosing 5 or 10)
            lr
                - learning rate ETA
            lr_fixed {None | True}
                - learning rate method choices:
                  [*] fixed: fixed learning rate for each training iteration
                  [*] non-fixed: ETA = ALPHA/sqrt(i) where ALPHA > 0, i = 1 .. N iterations
        [output]
            depth
                - the best architecture of the feedforward neural network from depth (2~6)
            regularization
                - the best regularization parameter lambda in [0, 1]
            h_SGD
                - the hypothesis h_SGD returned by the SGD algorithm with the best performing w(i) on a validation set
            perf
                - performance of the returned hypothesis h_SGD on the labeled testing sample S kernel
    Student Name:	XIN-YU KUO
    Student ID:		105062633 (CS)
    Contact:		Bassy <aaammmyyy27@gmail.com>
"""

from data import *
from Hypothesis import *
#from sklearn.metrics.pairwise import *
import sys, os
import argparse
import numpy as np
import threading
import time

timer = time.strftime("%m%d%H%M", time.localtime())

depth_trace = 1		# the number of different depth sizes for training (between 1 ~ 5)
lam_trace	= 1		# the number of different lambda for training (between 1 ~ 5)

output_file = 'SGD_hypothesis_header.csv'

th_param = {
        'loss'  : [None] * depth_trace * lam_trace * 10,
        'test'  : {},
        'depth' : {},
        'lam'	: {},
        'W' : {},
        'b'	: {},
        }

def ProgParser ():
    """
    Command line:
        python3 <this_file> --trainset {train.csv} --testset {test.csv} --n 5 --lr 0.0002 --lr_fixed True
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
            description="Usage Example:\n"
                        "	python3 %(prog)s train --epoch 10000 --lr 0.02 > output.txt\n"
                        "	python3 %(prog)s test")
    parser.add_argument('-v', '--version',  action='version', version='%(prog)s 1.0')
    ### REQUIRED ###
    parser.add_argument('command', metavar='ACTION',	type=str, choices=['train', 'test'],
            help="train: training phase.\n"
                 "test: testing with parameters the latest training generated.")
    ### OPTIONAL ###
    parser.add_argument('--trainset', default='energy_efficiency_cooling_load_training.csv', help='a training file (.csv) [default: %(default)s]')
    parser.add_argument('--testset',  default='energy_efficiency_cooling_load_testing.csv',  help='a testing file (.csv) [default: %(default)s]')
    parser.add_argument('--n',		  default=5,		type=int,	 help='the number of fold (5 or 10) [default: %(default)s]', choices=[5,10])
    parser.add_argument('--lam',	  default=None,		type=float,
            help='the regularization parameter to make enough small of weight (0~1, None for random) [default: %(default)s]')
    parser.add_argument('--lr',		  default=0.01,	type=float,	 help='the learning rate ETA [default: %(default)s]')
    parser.add_argument('--lr_fixed', default=False,	type=bool, choices=[False, True],
            help='learning rate method choices: ("" or True) [default: %(default)s] \n'
                    '[*] fixed: fixed learning rate for each training iteration \n'
                    '[*] non-fixed: ETA = ALPHA/sqrt(i) where ALPHA > 0, i = 1 .. N iterations \n')
    parser.add_argument('--sample',	  default='purely',	type=str, choices=['purely', 'round-robin', 'hyper'],
            help='an algorithm to take a labeled item (x(i),c(x(i)) from the sample S [default: %(default)s]\n'
                 '[*] purely: selecting a labeled item form S at random in each iteration\n'
                 '[*] round-robin: first ordering the sample S at random and once the order is fixed, selecting a labeled item form S in this order and around.\n'
                 '[*] hyper: Round-robin once and then purely random.\n')
    parser.add_argument('--epoch',	  default=10000,	type=int, help='training iteration [default: %(default)s]')
    parser.add_argument('--depth',	  default=3,		type=int, help='the count of hidden layers + output layer [default: %(default)s]')
    parser.add_argument('--log_dir',  default='log/',	type=str, help='the folder saving your log files [default: %(default)s]')
    args = parser.parse_args()
    return args

################################################################################################################################################################

def main(args):
    if not os.path.isdir(args.log_dir):
        print("Create `{}` directory.".format(args.log_dir))
        os.makedirs(args.log_dir)

    xtr, ytr = load_data(args.trainset,0)
    xte, yte = load_data(args.testset,0)
    n = args.n
    lr	= args.lr
    lam = args.lam
    epoch = args.epoch
    depth = args.depth
    lr_fixed = args.lr_fixed

    xte, yte = shuffle_union(xte, yte)
    train_size = len(xtr)

    ###############################################
    ### Start to train
    ###############################################
    th = {}
    for fold in range(n):
        xtr, ytr = shuffle_union(xtr, ytr)
        xva, yva = xtr[:len(xtr)*2//3], ytr[:len(ytr)*2//3]
        xcv, ycv = xtr[len(xtr)*2//3+1:], ytr[len(ytr)*2//3+1:]
        for d in range(depth_trace):
            for l in range(lam_trace):
                options = {
                    'lr' : lr,
                    'lam' : lam,
                    'xcv' : xcv,
                    'ycv' : ycv,
                    'xva' : xva,
                    'yva' : yva,
                    'xte' : xte,
                    'yte' : yte,
                    'depth' : depth + d,
                    'epoch' : epoch,
                    'lr_fixed' : lr_fixed,
                    'train_size' : train_size,
                    }
                # train(options, fold)
                th_id = l * n * depth_trace + d * n + fold
                th[th_id] = threading.Thread(target=train, args=(options, th_id,))
                th[th_id].start()

    for th_id in range(n * depth_trace * lam_trace):
        th[th_id].join()
    # print("Loss: ", th_param['loss'])
    # print("Depth: ", th_param['depth'])
    # print("Lam: ", th_param['lam'])

    OPT = np.argmin(th_param['loss'][:n * depth_trace * lam_trace])
    h_test = Hypothesis(th_param['W'][OPT], th_param['b'][OPT])
    print("[OPTIMAL]")
    print("+ Val Loss: {}".format(th_param['loss'][OPT]))
    print("+ Depth: {}".format(th_param['depth'][OPT]))
    print("+ Lambda: {}".format(th_param['lam'][OPT]))
    print("Test Error: {}".format(h_test.error(xte[:], yte[:])))

    save_file(output_file, th_param['W'][OPT])
    save_log(os.path.join(args.log_dir, 'table-{}ep-{}.txt'.format(epoch,timer)), th_param, epoch)

def train(options, th_id):
    lr = options['lr']
    lam = options['lam']
    xcv = options['xcv']
    ycv = options['ycv']
    xva = options['xva']
    yva = options['yva']
    xte = options['xte']
    yte = options['yte']
    depth = options['depth']
    epoch = options['epoch']
    lr_fixed = options['lr_fixed']
    train_size = options['train_size']
    x_dim = xcv.shape[1]

    #########################################################
    ### Initialization of layer size, lambda, bias
    #########################################################

    #### LAYER SIZE ####
    K = {}
    K[0] = x_dim	# input layer
    for t in range(1, depth):
        # K[t] = int(train_size / 15 / depth)  -1	# for 1 neuron always output 1
        K[t] = int(4-t) if depth <= 3 else 2
    K[depth] = 1	# output layer

    ## RANDOM WEIGHTS ##
    W = {}
    for t in range(depth):
        #########################################################
        ## Weights choosen from uniform distribution in [-1/n, 1/n]
        ## - total number of weights < 1/15 * len(train)
        ## - input layer size	= K[t] + 1
        ## - output layer size	= K[t+1]
        ## - final output layer size = 1
        #########################################################
        # W[t] = np.random.uniform(-1/np.sqrt(x_dim), 1/np.sqrt(x_dim), [K[t] + 1, K[t+1]])
        W[t] = np.clip(np.random.normal(0., 1/x_dim, size=[K[t] + 1, K[t+1]]), -1/np.sqrt(x_dim), 1/np.sqrt(x_dim))

    ## LAMBDA & BIAS ##
    lam = np.random.random_sample() if lam is None else lam	# regularization parameter in [0,1]
    b	= np.ones((depth,1))			# bias: the neuron always output 1
    h	= Hypothesis(W,b)

    R_opt = np.inf
    alpha = lr
    for ep in range(epoch):
        if not lr_fixed:
            lr = alpha / np.sqrt(ep+1)
        ## PURELY RANDOM
        if args.sample == 'purely':
            i = np.random.randint(len(ycv))
        ## ROUND-ROBIN
        elif args.sample == 'round-robin':
            i = ep % len(ycv)
        ## HYPER
        else:
            i = (ep % 2) * (int(ep/2) % len(ycv)) + \
                (ep%2+1) * (np.random.randint(len(ycv)))

        h.train(xcv[i], ycv[i], lr, lam)

        R = h.error(xva[:], yva[:])
        if ep % 100:
            print("[Thread-{}][epoch {}]	Val Loss: {}".format(th_id, ep, R))
        if R < R_opt:
            R_opt = R
            opt = {
                    'W'		: h.W,
                    'lam'	: lam,
                    'lr'	: lr,
                    'depth'	: depth,
                    'loss'	: R,
                    'b'		: b,
                    }

    th_param['depth'][th_id] = opt['depth']
    th_param['loss'][th_id] = opt['loss']
    th_param['lam'][th_id] = opt['lam']
    th_param['W'][th_id] = opt['W']
    th_param['b'][th_id] = opt['b']
    h_test = Hypothesis(opt['W'], opt['b'])
    th_param['test'][th_id] = h_test.error(xte[:], yte[:])


def save_file(output_file, W):
    import pandas as pd
    import csv
    print("\n>> Record `depth`, `sizes`, `weight_{N}` for each layer in %s\n" % output_file)
    output = {
            'depth': [len(W)],
            'sizes': []
            }
    for t in range(len(W)):
        output['sizes'].append(W[t].shape[0])
        output['W_%s' % (t)] = W[t].reshape(-1).tolist()
    output['sizes'].append(1)
    # df = pd.DataFrame.from_dict(output, orient='index').fillna('')
    # df.to_csv(output_file, sep='\t', index=True, header=False)
    with open(output_file,'w',encoding='utf8') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerow(['depth'] + output['depth'])
        writer.writerow(['sizes'] + output['sizes'])
        for t in range(len(W)):
            name = 'W_%s' % (t)
            writer.writerow([name] + output[name])


def save_log(output_file, param, epoch):
    import csv
    print("\n>> Record `depth`, `lambda`, `val_loss`, `test_loss`, `epoch`, 'sizes' for each thread in %s\n" % output_file)
    header = ['depth', 'lambda', 'val_loss', 'test_loss', 'epoch', 'sizes']
    with open(output_file,'w',encoding='utf8') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerow(header)
        for th in range(len(param['depth'])):
            sizes = {}
            for i in range(len(param['W'][th])):
                sizes[i] = param['W'][th][i].shape[0]
            sizes[i] = param['W'][th][i].shape[1]
            row = [param['depth'][th], param['lam'][th], param['loss'][th], param['test'][th], epoch, sizes]
            writer.writerow(row)

def test(args):
    global output_file
    ## Load info file ##
    output_file = input("Load the hypothesis parameters file [%s]: " % output_file) or output_file
    info = {}
    with open(output_file, 'r') as f:
        for row in f.readlines():
            row = row.split()
            info[row[0]] = [float(i) for i in row[1:]]

    ## Load testing file ##
    testset = input("Load the testing file [%s]: " % args.testset) or args.testset
    xte, yte = load_data(testset,0)

    ## Hypothesis and train, print error ##
    W = {}
    for t in range(int(info['depth'][0])):
        input_size	= int(info['sizes'][t])
        output_size = int(info['sizes'][t+1]) - 1 if int(info['sizes'][t+1]) > 1 else 1
        W[t] = np.array(info['W_%s' % t]).reshape(input_size, output_size)

    b = np.ones((int(info['depth'][0]), 1))
    h_test = Hypothesis(W, b)
    print("[Performance]")
    print("Test Error: {}".format(h_test.error(xte, yte)))

if __name__ == '__main__':
    args = ProgParser()
    if args.command == 'train':
        main(args)
    elif args.command == 'test':
        test(args)
