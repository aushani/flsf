import numpy as np

class OldFlowSample:

    def __init__(self, flow_sample=None, occ1=None, occ2=None, match=None, foreground=None):
        if flow_sample:
            self.make_feature_vector(flow_sample.occ1, flow_sample.occ2)
            self.match = flow_sample.match
        else:
            assert not occ1 is None
            assert not occ2 is None
            assert not match is None
            assert not foreground is None

            self.make_feature_vector(occ1, occ2)
            self.match = match
            self.foreground = foreground

    def make_feature_vector(self, occ1, occ2):
        height = occ1.shape[2]

        self.feature_vector = np.zeros((height * 3,))

        # Center position
        idx_i = occ1.shape[0] / 2 + 1
        idx_j = occ1.shape[1] / 2 + 1

        for idx_k in range(height):
            x1 = occ1[idx_i, idx_j, idx_k]
            x2 = occ2[idx_i, idx_j, idx_k]

            free1 = x1 < -0.01
            free2 = x2 < -0.01

            occu1 = x1 > 0.01
            occu2 = x2 > 0.01

            if free1 and free2:
                idx_fv = 0
            elif occu1 and occu2:
                idx_fv = 0
            elif free1 and occu2:
                idx_fv = 2
            elif occu1 and free2:
                idx_fv = 2
            else:
                continue

            self.feature_vector[3*idx_k + idx_fv] = 1

class OldFlowSampleSet:

    def __init__(self, flow_samples):
        shape = flow_samples.occ1.shape

        n = shape[0]

        # Center position
        idx_i = shape[1] / 2 + 1
        idx_j = shape[2] / 2 + 1

        height = shape[3]
        self.feature_vector = np.zeros((n, height*3))

        self.match = np.zeros((n,))
        self.foreground = np.zeros((n,))

        for i in range(n):
            occ1 = flow_samples.occ1[i, :, :, :]
            occ2 = flow_samples.occ2[i, :, :, :]

            match = flow_samples.match[i]
            foreground = flow_samples.filter[i]
            sample = OldFlowSample(occ1 = occ1, occ2 = occ2, match = match, foreground=foreground)

            self.feature_vector[i, :] = sample.feature_vector
            self.match[i] = sample.match
            self.foreground[i] = sample.foreground
