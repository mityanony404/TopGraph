import numpy as np
import sys
import re

if __name__ == '__main__':
    fname = sys.argv[1]
    train_acc = []
    test_acc = []
    with open(fname, 'r') as f:
        for line in f:
            if 'Epoch' in line:
                acc = re.findall("\d+\.\d+", line)
                train_acc.append(float(acc[1]))
                test_acc.append(float(acc[2]))
    print(f'File: {fname} Train acc: {np.mean(train_acc):.4f}  Train std: {np.std(train_acc):.4f} '
          f'Test acc: {np.mean(test_acc):.4f} Test std: {np.std(test_acc):.4f}')

