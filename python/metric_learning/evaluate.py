from icra2017_learning import *
from metric_learning import *

def evaluate(model_file, n=100000, il=False, ml=False):
    np.random.seed(0)

    flow_dm = FlowDataManager(evaluation=False)
    flow_dm.make_validation(10000)

    ss = flow_dm.validation_set
    old_ss = OldFlowSampleSet(ss)

    match = old_ss.match == 1
    is_foreground = old_ss.foreground == 1

    if il:
        print 'Loading icra 2017', model_file
        il = Icra2017Learning()
        il.restore(model_file)
        scores = il.eval_scores(old_ss.feature_vector)

        return match[is_foreground], scores[is_foreground]

    if ml:
        print 'Loading metric learning', model_file
        ml = MetricLearning()
        ml.restore(model_file)
        dists = ml.eval_dist(ss.occ1, ss.occ2)

        return match[is_foreground], -dists[is_foreground]
