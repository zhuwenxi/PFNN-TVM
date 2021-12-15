import os
import json
import sys
import argparse

cacheBlock = {}
def getScheduleFromLog_interpA_xsmm(dim):
    tunedLog = './logs/logs_interpA_autoTVM_xsmm/tune_M{}_N{}_K{}.log'.format(dim[0], dim[1], dim[2])
    if os.path.exists(tunedLog):
        with open(tunedLog) as fp:
            context = json.loads(fp.readlines()[0])
            schedule = context['config']['entity']
            key = 'M' + str(dim[0]) + 'N' + str(dim[1]) + 'K' + str(dim[2])
            cacheBlock[key] = (schedule[1][2][-1], schedule[2][2][-1])

def getScheduleFromLog_interpA(dim):
    tunedLog = './logs/logs_interpA_autoTVM/tune_M{}_N{}_K{}.log'.format(dim[0], dim[1], dim[2])
    if os.path.exists(tunedLog):
        with open(tunedLog) as fp:
            context = json.loads(fp.readlines()[0])
            schedule = context['config']['entity']
            key = 'M' + str(dim[0]) + 'N' + str(dim[1]) + 'K' + str(dim[2])
            cacheBlock[key] = (schedule[1][2][-1], schedule[2][2][-1])

def getScheduleFromLog_interpB(dim):
    tunedLog = './logs/logs_interpB_autoTVM/tune_M{}_N{}_K{}.log'.format(dim[0], dim[1], dim[2])
    if os.path.exists(tunedLog):
        with open(tunedLog) as fp:
            context = json.loads(fp.readlines()[0])
            schedule = context['config']['entity']
            key = 'M' + str(dim[0]) + 'N' + str(dim[1]) + 'K' + str(dim[2])
            cacheBlock[key] = (schedule[0][2][-1], schedule[1][2][-1])

def getScheduleFromLog(dim):
    key = 'M' + str(dim[0]) + 'N' + str(dim[1]) + 'K' + str(dim[2])
    cacheBlock[key] = (16, dim[2])

def writeToFile():
    cacheBlockJson = './cache_block_size.json'
    with open(cacheBlockJson, 'w') as fp:
        fp.write(json.dumps(cacheBlock))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TVM compute library for PFNN with cubic optimization')
    parser.add_argument('-d', '--hidden', dest='hidden_len', default = 256, type=int,
            help='Hidden layer size.')
    parser.add_argument('--mode', dest='mode', default='manual', choices=['manual', 'autoTVM', 'ansor'], 
            help='select scheduler mode manual, autoTVM or ansor')
    parser.add_argument('--interp', dest='interp', default='B', choices=['A', 'B'],
            help='select interpolation: A or B')
    parser.add_argument('--extern', dest='extern', action="store_true", 
            help='whether to use extern micro kernel')
    args = parser.parse_args()
    K1, N3 = 1032, 912
    N1 = args.hidden_len
    N2 = args.hidden_len
    K2 = args.hidden_len
    K3 = args.hidden_len
    for batch in (1, 5, 8):
        layer_shapes = [(batch, N1, K1), (batch, N2, K2), (batch, N3, K3)]
        for dim in layer_shapes:
            if args.mode == 'autoTVM':
                if args.interp == 'B':
                    getScheduleFromLog_interpB(dim)
                else:
                    if args.extern:
                        getScheduleFromLog_interpA_xsmm(dim)
                    else:
                        getScheduleFromLog_interpA(dim)
            else:
                getScheduleFromLog(dim)
    writeToFile()



            


