import os
import sys
import struct
import random
import argparse

parser = argparse.ArgumentParser(description='TVM compute library for PFNN with cubic optimization')
parser.add_argument('-d', '--hidden', dest='hidden_len', default = 256, type=int,
            help='Hidden layer size.')
args = parser.parse_args()

batch = 8
W = 4

K1 = 1032
N1 = args.hidden_len
K2 = args.hidden_len
N2 = args.hidden_len
K3 = args.hidden_len
N3 = 908


file_path = "./models/Xmean.bin"
with open(file_path, 'wb+') as fp:
    for i in range(K1):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/Xstd.bin"
with open(file_path, 'wb+') as fp:
    for i in range(K1):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/Ymean.bin"
with open(file_path, 'wb+') as fp:
    for i in range(N3):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/Ystd.bin"
with open(file_path, 'wb+') as fp:
    for i in range(N3):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/Nor_W0.bin"
with open(file_path, 'wb+') as fp:
    for i in range(W * K1 * K2):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/Nor_W1.bin"
with open(file_path, 'wb+') as fp:
    for i in range(W * K2 * K3):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/Nor_W2.bin"
with open(file_path, 'wb+') as fp:
    for i in range(W * K3 * N3):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/Nor_b0.bin"
with open(file_path, 'wb+') as fp:
    for i in range(K2):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/Nor_b1.bin"
with open(file_path, 'wb+') as fp:
    for i in range(K3):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/Nor_b2.bin"
with open(file_path, 'wb+') as fp:
    for i in range(N3):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/inputs.bin"
with open(file_path, 'wb+') as fp:
    for i in range(batch * K1):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)

file_path = "./models/outputs.bin"
with open(file_path, 'wb+') as fp:
    for i in range(batch * N3):
        data=random.random()
        num = struct.pack("f", data)
        fp.write(num)
