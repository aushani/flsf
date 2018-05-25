import numpy as np
import matplotlib.pyplot as plt

path = '/home/aushani/data/eval_tsf++'
filenames = [
    #'%s/2011_09_26_drive_0001_sync/eval.csv' % (path),
    #'%s/2011_09_26_drive_0002_sync/eval.csv' % (path),
    #'%s/2011_09_26_drive_0005_sync/eval.csv' % (path),
    #'%s/2011_09_26_drive_0009_sync/eval.csv' % (path),
    #'%s/2011_09_26_drive_0011_sync/eval.csv' % (path),
    #'%s/2011_09_26_drive_0013_sync/eval.csv' % (path),
    #'%s/2011_09_26_drive_0014_sync/eval.csv' % (path),
    #'%s/2011_09_26_drive_0015_sync/eval.csv' % (path),
    #'%s/2011_09_26_drive_0017_sync/eval.csv' % (path),
    #'%s/2011_09_26_drive_0018_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0019_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0020_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0022_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0023_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0027_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0028_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0029_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0032_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0035_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0036_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0039_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0046_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0048_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0051_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0052_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0056_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0057_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0059_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0060_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0061_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0064_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0070_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0079_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0084_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0086_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0087_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0091_sync/eval.csv' % (path),
    '%s/2011_09_26_drive_0093_sync/eval.csv' % (path),
    ]

def load(fn):
    return np.loadtxt(fn)

errs = None
for f in filenames:
    res = load(f)

    if errs is None:
        errs = res
    else:
        errs = np.append(errs, res)

print 'Count: ', len(errs)

n_30 = np.sum(errs < 0.3) + 0.0
print 'Within 30 cm', n_30 / len(errs)

print 'Mean', np.mean(errs)
print 'Median', np.median(errs)

plt.hist(errs, weights=(1.0/len(errs),)*len(errs), range=(0, 1.0), bins=20)
plt.plot([0.3, 0.3], [0, 0.35], 'k-')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 0.35)
plt.show()
