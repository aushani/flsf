import numpy as np

class FlowSample:

    def __init__(self, data, rotate=0, flip=False):
        self.width = 27
        self.length = 27
        self.height = 13

        self.occ1 = data['occ1'].reshape((self.width, self.length, self.height))
        self.occ2 = data['occ2'].reshape((self.width, self.length, self.height))

        self.err2 = data['err2']
        self.filter = data['filter']
        self.match = data['match']

        # Rescale range from 0-1 to -0.5 - +0.5
        self.occ1 -= 0.5
        self.occ2 -= 0.5

        # Data augmentation
        self.occ1 = np.rot90(self.occ1, k=rotate)
        self.occ2 = np.rot90(self.occ2, k=rotate)

        if flip:
            self.occ1 = self.occ1.transpose(1, 0, 2)
            self.occ2 = self.occ2.transpose(1, 0, 2)

class FlowSampleSet:

    def __init__(self, samples):
        n = len(samples)
        shape = (n, samples[0].occ1.shape[0], samples[0].occ1.shape[1], samples[0].occ1.shape[2])

        self.occ1 = np.zeros(shape)
        self.occ2 = np.zeros(shape)

        self.err2 = np.zeros((n,))
        self.filter = np.zeros((n,))
        self.match = np.zeros((n,))

        for i, sample in enumerate(samples):
            self.occ1[i, :, :, :] = sample.occ1
            self.occ2[i, :, :, :] = sample.occ2

            self.err2[i] = sample.err2
            self.filter[i] = sample.filter
            self.match[i] = sample.match
