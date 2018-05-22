import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

def make_pr_curve(true_label, pred_label, label):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(true_label, pred_label)

    plt.plot(recall, precision, label=label)

    plt.legend(loc='lower left')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)

ml_match = np.loadtxt('ml_match.csv')
ml_score = np.loadtxt('ml_score.csv')

il_match = np.loadtxt('il_match.csv')
il_score = np.loadtxt('il_score.csv')

plt.clf()

make_pr_curve(il_match, il_score, 'Occupancy Constancy (ICRA 2017)')
make_pr_curve(ml_match, ml_score, 'Metric Learning')

plt.title('Precision-Recall for Constancy Metrics')

plt.show()

