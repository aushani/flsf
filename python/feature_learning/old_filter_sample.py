import numpy as np

class OldFilterSample:

    def __init__(self, filter_sample=None, occ=None, filter_res=None):
        if not filter_sample is None:
            self.make_feature_vector(filter_sample.occ)
            self.make_filter(filter_sample.filter)
        else:
            assert not occ is None
            assert not filter_res is None

            self.make_feature_vector(occ)
            self.make_filter(filter_res)

    def make_feature_vector(self, occ):
        height = occ.shape[2]

        self.feature_vector = np.zeros((occ.shape[0], occ.shape[1], height*2))

        is_occu = occ > 0.01
        is_free = occ < -0.01

        self.feature_vector[:, :, :height] = is_occu
        self.feature_vector[:, :, height:] = is_free

    def make_filter(self, filter_res):
        self.filter = np.zeros(filter_res.shape)

        is_foreground = filter_res == 1
        is_background = filter_res == 0
        is_invalid = filter_res == -1

        #print np.sum(is_foreground)
        #print np.sum(is_background)
        #print np.sum(is_invalid)
        #print np.size(filter_res)
        #assert np.sum(is_foreground) + np.sum(is_background) + np.sum(is_invalid) == np.size(filter_res)

        self.filter[is_foreground] = 1
        self.filter[is_background] = -1

class OldFilterSampleSet:

    def __init__(self, filter_samples):
        shape = filter_samples.occ.shape

        n = shape[0]

        self.feature_vector = np.zeros((n, shape[1], shape[2], shape[3]*2))

        self.filter = np.zeros(filter_samples.filter.shape)

        for i in range(n):
            occ = filter_samples.occ[i, :, :, :]
            filter_res = filter_samples.filter[i, :, :]

            sample = OldFilterSample(occ=occ, filter_res=filter_res)

            self.feature_vector[i, :, :, :] = sample.feature_vector
            self.filter[i, :, :] = sample.filter
