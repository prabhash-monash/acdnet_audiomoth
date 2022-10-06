import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description="Redoing ACDNET Code for Sound Classification by Prabhash")

    # General settings.
    parser.add_argument('--netType', default='ACDNet', required=False);
    parser.add_argument('--data', default='{}/datasets/'.format(os.getcwd()), required=False)
    parser.add_argument('--dataset', required=False, default='esc50', choices=['esc10', 'esc50', 'frog'])
    parser.add_argument('--BC', default=True, action='store_true', help='BC learning')
    parser.add_argument('--strongArgument', default=True, action='store_true', help='Add scale and gain augmentation');

    opt = parser.parse_args();
    print("opt = " + str(opt));

    # Learning settings.
    opt.batchSize = 64;
    opt.weightDecay = 5e-4;
    opt.momentum = 0.9;
    opt.nEpochs = 2000;
    opt.LR = 0.1;
    opt.schedule = [0.3, 0.6, 0.9];
    opt.warmup = 10;

    # Basic Net Settings.
    opt.nClasses = 50;
    opt.nFolds = 5;
    opt.splits = [i for i in range(1, opt.nFolds + 1)];
    opt.sr = 20000;
    opt.inputLength = 30225;

    # Test data.
    opt.nCrops = 10;

    return opt;


def display_info(opts):
    print('+-----------------------------------------------+');
    print('| {} Sound Classification              |'.format(opts.netType));
    print("+-----------------------------------------------+");
    print("| dataset = {}".format(opts.dataset));
    print("| nEpochs = {}".format(opts.nEpochs));
    print("| LRInit = {}".format(opts.LR));
    print("| schedule = {}".format(opts.schedule));
    print("| batchSize = {}".format(opts.batchSize));
    print("| weightDecay = {}".format(opts.weightDecay));
    print("| momentum = {}".format(opts.momentum));
    print("| warmup = {}".format(opts.warmup));
    print("|-----------------------------------------------+");
    print("|            Basic Net Settings:");
    print("+-----------------------------------------------+");
    print("| nClasses = {}".format(opts.nClasses));
    print("| nFolds = {}".format(opts.nFolds));
    print("| splits = {}".format(opts.splits));
    print("| sr = {}".format(opts.sr));
    print("| inputLength = {}".format(opts.inputLength));
    print("|-----------------------------------------------+");
    print("|           Test Data Settings:");
    print("+-----------------------------------------------+");
    print("| nCrops = {}".format(opts.nCrops));
    print("+_______________________________________________+")
