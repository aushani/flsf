import numpy as np

class Sample:

    def __init__(self, data, rotate=0, flip=False):
        self.width = 167
        self.length = 167
        self.height = 13

        self.occ1 = data['occ1'].reshape((self.width, self.length, self.height))
        self.occ2 = data['occ2'].reshape((self.width, self.length, self.height))

        self.filter = data['filter'].reshape((self.width, self.length))
        self.flow = data['flow'].reshape((self.width, self.length, 2))

        # Rescale range from 0-1 to -0.5 - +0.5
        self.occ1 -= 0.5
        self.occ2 -= 0.5

        # Data augmentation
        self.occ1 = np.rot90(self.occ1, k=rotate)
        self.occ2 = np.rot90(self.occ2, k=rotate)
        self.filter = np.rot90(self.filter, k=rotate)
        self.flow = np.rot90(self.flow, k=rotate)

        if flip:
            self.occ1 = self.occ1.transpose(1, 0, 2)
            self.occ2 = self.occ2.transpose(1, 0, 2)

            self.filter = self.filter.transpose()
            self.flow = self.flow.transpose(1, 0, 2)

class SampleSet:

    def __init__(self, samples):
        n = len(samples)
        shape = (n, samples[0].occ1.shape[0], samples[0].occ1.shape[1], samples[0].occ1.shape[2])

        self.occ1 = np.zeros(shape)
        self.occ2 = np.zeros(shape)

        self.filter = np.zeros((n, samples[0].filter.shape[0], samples[0].filter.shape[1]))
        self.flow = np.zeros((n, samples[0].flow.shape[0], samples[0].flow.shape[1], samples[0].flow.shape[2]))

        for i, sample in enumerate(samples):
            self.occ1[i, :, :, :] = sample.occ1
            self.occ2[i, :, :, :] = sample.occ2

            self.filter[i, :, :] = sample.filter
            self.flow[i, :, :, :] = sample.flow
