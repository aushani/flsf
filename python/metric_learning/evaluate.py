from icra2017_learning import *
from icra2017_background import *
from metric_learning import *

def evaluate_matches(model_file, n=5000, il=False, ml=False):
    np.random.seed(0)

    flow_dm = FlowDataManager(evaluation=False)
    flow_dm.make_validation(n)

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

def evaluate_filter(model_file, n=100, ib=False, ml=False):
    np.random.seed(0)

    filter_dm = FilterDataManager(evaluation=False)
    filter_dm.make_validation(n)

    ss = filter_dm.validation_set
    old_ss = OldFilterSampleSet(ss)

    filter_res = ss.filter
    is_valid = filter_res >= 0

    if ib:
        print 'Loading icra 2017', model_file
        ib = Icra2017Background()
        ib.restore(model_file)
        scores = ib.eval_scores(old_ss.feature_vector)

        return filter_res[is_valid], scores[is_valid]

    if ml:
        print 'Loading metric learning', model_file
        ml = MetricLearning()
        ml.restore(model_file)
        prob = ml.eval_filter_prob(ss.occ)
        prob = prob[:, :, :, 1]

        return filter_res[is_valid], prob[is_valid]
