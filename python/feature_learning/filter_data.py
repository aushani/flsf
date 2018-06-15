from filter_sample import *
import os
import numpy as np

import matplotlib.pyplot as plt

class FilterDataManager:

    def __init__(self, path, shuffle=True, evaluation=False):
        # only use first 10 for training
        self.filenames = [
            '%s/2011_09_26_drive_0001_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0002_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0005_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0009_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0011_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0013_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0014_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0015_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0017_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0018_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0019_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0020_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0022_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0023_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0027_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0028_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0029_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0032_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0035_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0036_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0039_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0046_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0048_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0051_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0052_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0056_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0057_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0059_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0060_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0061_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0064_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0070_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0079_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0084_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0086_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0087_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0091_sync/filter.bin' % (path),
            '%s/2011_09_26_drive_0093_sync/filter.bin' % (path),
            ]

        # only use first 10 for training
        if evaluation:
            self.filenames = self.filenames[10:]
        else:
            self.filenames = self.filenames[:10]

        self.f_ptrs = {}
        self.file_ranges = {}

        self.padding = 24

        self.output_width = 167
        self.output_length = 167
        self.input_width = 167 + self.padding
        self.input_length = 167 + self.padding
        self.height = 13

        self.size_occ = self.input_width * self.input_length * self.height
        self.size_filter = self.output_width * self.output_length

        self.sample_size_bytes = (self.size_occ + self.size_filter)*4

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

        return FilterSampleSet(samples)

    def get_next_sample(self, reserve = False, augment = True):
        res = self.get_sample(idx = self.idxs[self.idx_at], augment = augment)

        if reserve:
            self.reserved[self.idx_at] = True

        self.idx_at = (self.idx_at + 1) % self.num_samples

        while self.reserved[self.idx_at]:
            self.idx_at = (self.idx_at + 1) % self.num_samples

        return res

    def get_sample(self, idx, augment = False):
        dt = np.dtype([  ('occ', (np.float32, self.size_occ)),
                         ('filter', (np.int32, self.size_filter))])

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

            return FilterSample(x, rotate=rotate, flip=flip)
        else:
            return FilterSample(x)

if __name__ == '__main__':
    d = FilterDataManager()

    print 'Have %d samples' % (d.num_samples)

    num_neg = 0
    num_pos = 0

    for i in range(d.num_samples):
        sample = d.get_next_sample()

        num_neg += np.sum(sample.filter == 0)
        num_pos += np.sum(sample.filter == 1)

        print '% 5d / % 5d Have %d positive samples, %d negative samples' % (i, d.num_samples, num_pos, num_neg)

    print 'Have %d positive samples, %d negative samples' % (num_pos, num_neg)

    while True:
        sample = d.get_next_sample(augment = False)

        print 'occ', np.min(sample.occ[:]), np.max(sample.occ[:])
        print 'filter', np.min(sample.filter[:]), np.max(sample.filter[:])

        plt.clf()

        plt.pcolor(sample.filter)
        plt.colorbar()
        plt.title('Filter')

        plt.show()
