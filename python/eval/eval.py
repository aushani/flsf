import numpy as np
import matplotlib.pyplot as plt
import os.path

path = '/home/aushani/data/eval_tsf++/model.ckpt-2616000'
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

def get_errors(classname, smoothing):
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

    return errs

def process_labeled_tracklets(smoothing):
    classes = ['Car', 'Cyclist', 'Misc', 'Pedestrian', 'Tram', 'Truck', 'Van', 'Person (sitting)']

    errs = []

    for object_class in classes:
        errs = np.append(errs, get_errors(object_class, smoothing))

    print 'All Tracklets'
    print '\tCount: ', len(errs)

    n_30 = np.sum(errs < 0.3) + 0.0
    print '\tWithin 30 cm', n_30 / len(errs)

    print '\tMean', np.mean(errs)

    plt.hist(errs, weights=(1.0/len(errs),)*len(errs), range=(0, 1.0), bins=20)
    plt.plot([0.3, 0.3], [0, 0.35], 'k-')
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 0.35)
    plt.title('All Labeled Tracklets')


def process_class(classname, smoothing):
    errs = get_errors(classname, smoothing)

    print ''
    print classname
    print '\tCount: ', len(errs)

    n_30 = np.sum(errs < 0.3) + 0.0
    print '\tWithin 30 cm', n_30 / len(errs)

    print '\tMean', np.mean(errs)

    plt.hist(errs, weights=(1.0/len(errs),)*len(errs), range=(0, 1.0), bins=20)
    plt.plot([0.3, 0.3], [0, 0.35], 'k-')
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 0.35)
    plt.title(classname)

smoothing = 0.07

plt.subplot(2, 2, 1)
process_labeled_tracklets(smoothing)

plt.subplot(2, 2, 2)
process_class('Car', smoothing)

plt.subplot(2, 2, 3)
process_class('Cyclist', smoothing)

plt.subplot(2, 2, 4)
process_class('Pedestrian', smoothing)

plt.show()
