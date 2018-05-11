import numpy as np

class FilterSample:

    def __init__(self, data, rotate=0, flip=False):
        self.width = 167
        self.length = 167
        self.height = 13

        self.occ = data['occ'].reshape((self.width, self.length, self.height))
        self.filter = data['filter'].reshape((self.width, self.length))

        # Rescale range from 0-1 to -0.5 - +0.5
        self.occ -= 0.5

        # Data augmentation
        self.occ = np.rot90(self.occ, k=rotate)
        self.filter = np.rot90(self.filter, k=rotate)

        if flip:
            self.occ = self.occ.transpose(1, 0, 2)
            self.filter = self.filter.transpose()

class FilterSampleSet:

    def __init__(self, samples):
        n = len(samples)
        shape = (n, samples[0].occ.shape[0], samples[0].occ.shape[1], samples[0].occ.shape[2])

        self.occ = np.zeros(shape)
        self.filter = np.zeros((n, samples[0].filter.shape[0], samples[0].filter.shape[1]))

        for i, sample in enumerate(samples):
            self.occ[i, :, :, :] = sample.occ
            self.filter[i, :, :] = sample.filter
