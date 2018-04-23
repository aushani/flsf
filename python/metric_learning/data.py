from sample import *
import os
import numpy as np

import sklearn.metrics
import matplotlib.pyplot as plt

class DataManager:

    def __init__(self, path='/home/aushani/data/matches_v2/'):
        # only use first 10 for training
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
            #'%s/2011_09_26_drive_0019_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0020_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0022_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0023_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0027_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0028_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0029_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0032_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0035_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0036_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0039_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0046_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0048_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0051_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0052_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0056_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0057_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0059_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0060_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0061_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0064_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0070_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0079_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0084_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0086_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0087_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0091_sync/matches.bin' % (path),
            #'%s/2011_09_26_drive_0093_sync/matches.bin' % (path),
            ]

        self.f_ptrs = {}
        self.file_ranges = {}

        self.width = 7
        self.length = 7
        self.height = 12

        self.size_occ = self.width * self.length * self.height
        self.size_label = 1

        self.size_match = 1

        self.sample_size_bytes = (self.size_occ*2 + self.size_label + self.size_match)*4

        count = 0
        for fn in self.filenames:
            self.f_ptrs[fn] = open(fn, 'rb')

            statinfo = os.stat(fn)
            num_samples = statinfo.st_size / (self.sample_size_bytes)
            print '%s is %10d MB with %10f samples' % (fn, statinfo.st_size/(1024*1024), num_samples)

            self.file_ranges[fn] = (count, count + num_samples)
            count += num_samples

        self.num_samples = count

        self.idx_at = 0
        self.idxs = np.arange(0, count)
        np.random.shuffle(self.idxs)

        self.reserved = [False,] * self.num_samples

    def make_validation(self, n = 1000):
        self.validation_set = self.get_next_samples(n, reserve=True)

    def get_next_samples(self, n, reserve = False):
        samples = []

        for i in range(n):
            samples.append(self.get_next_sample(reserve=reserve))

        return SampleSet(samples)

    def get_next_sample(self, reserve = False):
        res = self.get_sample(idx = self.idxs[self.idx_at])

        if reserve:
            self.reserved[self.idx_at] = True

        self.idx_at = (self.idx_at + 1) % self.num_samples

        while self.reserved[self.idx_at]:
            self.idx_at = (self.idx_at + 1) % self.num_samples

        return res

    def get_sample(self, idx=0):
        dt = np.dtype([  ('occ1', (np.float32, self.size_occ)),
                         ('occ2', (np.float32, self.size_occ)),
                         ('label1', (np.int32, self.size_label)),
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

        return Sample(x)

if __name__ == '__main__':
    d = DataManager()
    d.make_validation(100)

    valid_set = d.validation_set
