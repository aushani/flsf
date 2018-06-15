from feature_learning import *
from filter_data import *
import matplotlib.pyplot as plt
import sys
import time

if len(sys.argv) < 2:
    print 'Please specify tensorflow model file'
    sys.exit(1)

model_file = sys.argv[1]

filter_data = FilterDataManager(shuffle=False)
flow_data = FlowDataManager(shuffle=False)

fl = FeatureLearning()
fl.restore(model_file)

while True:
    sample = filter_data.get_next_sample(augment = False)

    tic = time.time()
    probs = fl.eval_filter_prob(sample.occ)
    toc = time.time()
    t_ms = (toc - tic) * 1e3

    print 'Took %5.3f ms to eval prob' % (t_ms)

    probs = probs[0, :, :, :]

    plt.clf()

    plt.subplot(3, 1, 1)
    plt.pcolor(sample.filter)
    plt.colorbar()

    plt.subplot(3, 1, 2)
    plt.pcolor(probs[:, :, 0])
    plt.colorbar()

    plt.subplot(3, 1, 3)
    plt.pcolor(probs[:, :, 1])
    plt.colorbar()

    print 'showing'

    plt.show()
