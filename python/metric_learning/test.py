from metric_learning import *
from data import *
import matplotlib.pyplot as plt
import sys
import time

plt.switch_backend('GTKAgg')

if len(sys.argv) < 2:
    print 'Please specify tensorflow model file'
    sys.exit(1)

model_file = sys.argv[1]

dm = DataManager()

ml = MetricLearning(dm)
ml.restore(model_file)

while True:
    sample = dm.get_next_sample(augment = False)

    print sample.occ1

    tic = time.time()
    probs = ml.eval_filter_prob(sample.occ1)
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
