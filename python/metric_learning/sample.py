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

        # Rescale range from 0-1 to -0.5 - +0.5
        self.occ1 -= 0.5
        self.occ2 -= 0.5


        self.feature_vector = self.get_icra2017_feature_vector()

        self.label1 = data['label1']
        self.match = data['match']

    def get_icra2017_feature_vector(self):
        feature_vector = np.zeros((3*self.height,))

        i_mid = self.width / 2
        j_mid = self.length / 2
        for i in range(self.height):
            p1 = self.occ1[i_mid, j_mid, i] + 0.5
            p2 = self.occ2[i_mid, j_mid, i] + 0.5

            idx_offset = 0

            if p1 < 0.5 and p2 < 0.5:
                idx_offset = 0
            elif p1 > 0.5 and p2 > 0.5:
                idx_offset = self.height
            elif p1 < 0.5 and p2 > 0.5:
                idx_offset = 2*self.height
            elif p1 > 0.5 and p2 < 0.5:
                idx_offset = 2*self.height

            feature_vector[i + idx_offset] = 1

        return feature_vector

class SampleSet:

    def __init__(self, samples):
        n = len(samples)
        shape = (n, samples[0].occ1.shape[0], samples[0].occ1.shape[1], samples[0].occ1.shape[2])

        self.occ1 = np.zeros(shape)
        self.occ2 = np.zeros(shape)
        self.match = np.zeros((n,))

        self.label1 = np.zeros((n,))

        self.feature_vector = np.zeros((n, samples[0].feature_vector.shape[0]))

        self.samples = samples

        for i, sample in enumerate(samples):
            self.occ1[i, :, :, :] = sample.occ1
            self.occ2[i, :, :, :] = sample.occ2

            self.label1[i] = sample.label1
            self.match[i] = sample.match

            self.feature_vector[i, :] = sample.feature_vector

    def get_scores(self):
        scores = []

        for sample in self.samples:
            scores.append(sample.get_score())

        return np.array(scores)
