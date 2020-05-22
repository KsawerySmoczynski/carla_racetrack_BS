#Script for pretraining reference net
import argparse
from net.a3c_lstm_net import A3C_LSTM


def main(args):

    # 1. initialize net
    # 2. Train loop for num of epochs
    #   2.1 load random indexes for epoch
    pass

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-s', '--num_steps',
        default=10000,
        type=int,
        dest='num_steps',
        help='Max number of steps per episode, if set to "None" episode will run as long as termiination conditions aren\'t satisfied')

    args = argparser.parse_known_args()
    if len(args) > 1:
        args = args[0]
    try:
        main(args)
    except KeyboardInterrupt:
        print('Interrupted by user! Bye.')