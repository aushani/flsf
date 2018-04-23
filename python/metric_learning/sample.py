import numpy as np

idx_to_classes = ['Car', 'Cycl', 'Misc', 'NO_O', 'Pede', 'Tram', 'Truc', 'Van']
classes_to_idx = {}
for i, cn in enumerate(idx_to_classes):
    classes_to_idx[cn] = i
n_classes = len(idx_to_classes)

class Sample:

    def __init__(self, data):
        self.width = 7
        self.length = 7
        self.height = 12

        self.occ1 = data['occ1'].reshape((self.width, self.length, self.height))
        self.occ2 = data['occ2'].reshape((self.width, self.length, self.height))

        self.label1 = data['label1']
        self.match = data['match']

class SampleSet:

    def __init__(self, samples):
        n = len(samples)
        shape = (n, samples[0].occ1.shape[0], samples[0].occ1.shape[1], samples[0].occ1.shape[2])

        self.occ1 = np.zeros(shape)
        self.occ2 = np.zeros(shape)
        self.match = np.zeros((n,))

        self.label1 = np.zeros((n,))

        self.samples = samples

        for i, sample in enumerate(samples):
            self.occ1[i, :, :, :] = sample.occ1
            self.occ2[i, :, :, :] = sample.occ2

            self.label1[i] = sample.label1
            self.match[i] = sample.match

    def get_scores(self):
        scores = []

        for sample in self.samples:
            scores.append(sample.get_score())

        return np.array(scores)
