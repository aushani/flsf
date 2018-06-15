from flow_sample import *
import os
import numpy as np

import matplotlib.pyplot as plt

class FlowDataManager:

    def __init__(self, path, shuffle=True, evaluation=False):
        self.filenames = [
            '%s/2011_09_26_drive_0001_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0002_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0005_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0009_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0011_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0013_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0014_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0015_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0017_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0018_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0019_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0020_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0022_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0023_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0027_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0028_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0029_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0032_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0035_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0036_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0039_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0046_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0048_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0051_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0052_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0056_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0057_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0059_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0060_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0061_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0064_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0070_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0079_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0084_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0086_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0087_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0091_sync/matches.bin' % (path),
            '%s/2011_09_26_drive_0093_sync/matches.bin' % (path),
            ]

        # only use first 10 for training
        if evaluation:
            self.filenames = self.filenames[10:]
        else:
            self.filenames = self.filenames[:10]

        self.f_ptrs = {}
        self.file_ranges = {}

        self.width = 25
        self.length = 25
        self.height = 13

        self.size_occ = self.width * self.length * self.height
        self.size_err2 = 1
        self.size_label = 1
        self.size_match = 1

        self.sample_size_bytes = (self.size_occ*2 + self.size_label + self.size_err2 + self.size_match)*4

        count = 0
        for fn in self.filenames:
            self.f_ptrs[fn] = open(fn, 'rb')

            statinfo = os.stat(fn)
            num_samples = statinfo.st_size / (self.sample_size_bytes + 0.0)
            print '%s is %10d MB with %10f samples' % (fn, statinfo.st_size/(1024*1024), num_samples)

            self.file_ranges[fn] = (count, count + num_samples)
            count += num_samples

        self.num_samples = int(count)

        self.idx_at = 0
        self.idxs = np.arange(0, count)

        if shuffle:
            np.random.shuffle(self.idxs)

        self.reserved = [False,] * self.num_samples

    def make_validation(self, n = 1000):
        self.validation_set = self.get_next_samples(n, reserve=True, augment = False)

    def get_next_samples(self, n, reserve = False, augment = True):
        samples = []

        for i in range(n):
            samples.append(self.get_next_sample(reserve=reserve, augment=augment))

        return FlowSampleSet(samples)

    def get_next_sample(self, reserve = False, augment = True):
        res = self.get_sample(idx = self.idxs[self.idx_at], augment = augment)

        if reserve:
            self.reserved[self.idx_at] = True

        self.idx_at = (self.idx_at + 1) % self.num_samples

        while self.reserved[self.idx_at]:
            self.idx_at = (self.idx_at + 1) % self.num_samples

        return res

    def get_sample(self, idx, augment = False):
        dt = np.dtype([  ('occ1', (np.float32, self.size_occ)),
                         ('occ2', (np.float32, self.size_occ)),
                         ('err2', (np.float32, self.size_err2)),
                         ('filter', (np.int32, self.size_label)),
                         ('match', (np.int32, self.size_match))])

        # Figure out which file
        fn_idx = None
        for fn in self.filenames:
            start, finish = self.file_ranges[fn]
            if idx >= start and idx < finish:
                fn_idx = fn
                break

        if fn_idx is None:
            raise ValueError('Error, idx out of range')

        # Seek to sample
        idx_f = idx - self.file_ranges[fn_idx][0]
        self.f_ptrs[fn_idx].seek(self.sample_size_bytes * idx_f, os.SEEK_SET)

        x = np.fromfile(self.f_ptrs[fn_idx], dtype=dt, count=1)

        if augment:
            rotate = np.random.randint(0, 4)
            flip = np.random.randint(0, 2) == 0

            return FlowSample(x, rotate=rotate, flip=flip)
        else:
            return FlowSample(x)

if __name__ == '__main__':
    d = FlowDataManager()

    print 'Press enter to continue'
    raw_input()

    d.make_validation(1000)

    valid_set = d.validation_set

    print 'Filter'
    print valid_set.filter

    print 'Match'
    print valid_set.match

    print 'Err2'
    print valid_set.err2

    plt.clf()

    plt.subplot(3, 1, 1)
    plt.hist(np.sqrt(valid_set.err2[valid_set.match == 1]))

    plt.subplot(3, 1, 2)
    plt.hist(np.sqrt(valid_set.err2[valid_set.match == 0]))

    plt.subplot(3, 1, 3)
    #plt.hist(np.sqrt(valid_set.err2))
    plt.hist(valid_set.err2)

    plt.show()

    while True:
        sample = d.get_next_sample(augment = False)

        print 'occ1', np.min(sample.occ1[:]), np.max(sample.occ1[:])
        print 'occ2', np.min(sample.occ2[:]), np.max(sample.occ2[:])

        print 'err2', sample.err2
        print 'filter', sample.filter
        print 'match', sample.match

        raw_input()
