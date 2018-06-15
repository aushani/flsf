from occupancy_constancy import *
from feature_learning import *

def evaluate_matches(model_file, n=5000, oc=False, fl=False):
    np.random.seed(0)

    flow_dm = FlowDataManager(evaluation=False)
    flow_dm.make_validation(n)

    ss = flow_dm.validation_set
    old_ss = OldFlowSampleSet(ss)

    match = old_ss.match == 1
    is_foreground = old_ss.foreground == 1

    if ol:
        print 'Loading occupancy constancy', model_file
        ol = OccupancyConstancy()
        ol.restore(model_file)
        scores = ol.eval_scores(old_ss.feature_vector)

        return match[is_foreground], scores[is_foreground]

    if fl:
        print 'Loading metric learning', model_file
        fl = FeatureLearning()
        fl.restore(model_file)
        dists = fl.eval_dist(ss.occ1, ss.occ2)

        return match[is_foreground], -dists[is_foreground]
