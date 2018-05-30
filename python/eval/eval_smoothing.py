import numpy as np
import matplotlib.pyplot as plt
import os.path

path = '/home/aushani/data/eval_tsf++'
dirnames = [
    #'%s/2011_09_26_drive_0001_sync' % (path),
    #'%s/2011_09_26_drive_0002_sync' % (path),
    #'%s/2011_09_26_drive_0005_sync' % (path),
    #'%s/2011_09_26_drive_0009_sync' % (path),
    #'%s/2011_09_26_drive_0011_sync' % (path),
    #'%s/2011_09_26_drive_0013_sync' % (path),
    #'%s/2011_09_26_drive_0014_sync' % (path),
    #'%s/2011_09_26_drive_0015_sync' % (path),
    #'%s/2011_09_26_drive_0017_sync' % (path),
    #'%s/2011_09_26_drive_0018_sync' % (path),
    '%s/2011_09_26_drive_0019_sync' % (path),
    '%s/2011_09_26_drive_0020_sync' % (path),
    '%s/2011_09_26_drive_0022_sync' % (path),
    '%s/2011_09_26_drive_0023_sync' % (path),
    '%s/2011_09_26_drive_0027_sync' % (path),
    '%s/2011_09_26_drive_0028_sync' % (path),
    '%s/2011_09_26_drive_0029_sync' % (path),
    '%s/2011_09_26_drive_0032_sync' % (path),
    '%s/2011_09_26_drive_0035_sync' % (path),
    '%s/2011_09_26_drive_0036_sync' % (path),
    '%s/2011_09_26_drive_0039_sync' % (path),
    '%s/2011_09_26_drive_0046_sync' % (path),
    '%s/2011_09_26_drive_0048_sync' % (path),
    '%s/2011_09_26_drive_0051_sync' % (path),
    '%s/2011_09_26_drive_0052_sync' % (path),
    '%s/2011_09_26_drive_0056_sync' % (path),
    '%s/2011_09_26_drive_0057_sync' % (path),
    '%s/2011_09_26_drive_0059_sync' % (path),
    '%s/2011_09_26_drive_0060_sync' % (path),
    '%s/2011_09_26_drive_0061_sync' % (path),
    '%s/2011_09_26_drive_0064_sync' % (path),
    '%s/2011_09_26_drive_0070_sync' % (path),
    '%s/2011_09_26_drive_0079_sync' % (path),
    '%s/2011_09_26_drive_0084_sync' % (path),
    '%s/2011_09_26_drive_0086_sync' % (path),
    '%s/2011_09_26_drive_0087_sync' % (path),
    '%s/2011_09_26_drive_0091_sync' % (path),
    '%s/2011_09_26_drive_0093_sync' % (path),
    ]

def load(fn):
    return np.loadtxt(fn)

def get_within_30cm(classname, smoothing):
    errs = None
    for d in dirnames:
        fn  = '%s/%s/%s.csv' % (d, smoothing, classname)
        if not os.path.isfile(fn):
            continue

        res = load(fn)

        if errs is None:
            errs = res
        else:
            errs = np.append(errs, res)

    if errs is None:
        return None

    n_30 = np.sum(errs < 0.3) + 0.0
    return n_30 / len(errs)

def make_smoothing_plot(classname):
    print classname
    xs = []
    ys = []
    for smoothing in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        p_30 = get_within_30cm(classname, smoothing)

        if not p_30 is None:
            xs.append(smoothing)
            ys.append(p_30)

    plt.semilogx(xs, ys)
    plt.xlabel('Smoothing param')
    plt.ylabel('Percent within 30 cm')
    plt.title(classname)

plt.subplot(2, 2, 1)
make_smoothing_plot('Car')

plt.subplot(2, 2, 2)
make_smoothing_plot('NoObject')

plt.subplot(2, 2, 3)
make_smoothing_plot('Cyclist')

plt.subplot(2, 2, 4)
make_smoothing_plot('Pedestrian')

plt.show()
