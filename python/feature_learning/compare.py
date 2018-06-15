import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

def make_pr_curve(true_label, pred_label, label, color=None):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(true_label, pred_label)

    plt.plot(recall, precision, color, label=label, linewidth=4)

    plt.legend(loc='lower left', fontsize=16)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)

def make_constancy_plots():
    fl_match = np.loadtxt('fl_match.csv')
    fl_score = np.loadtxt('fl_score.csv')

    il_match = np.loadtxt('il_match.csv')
    il_score = np.loadtxt('il_score.csv')

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    make_pr_curve(il_match, il_score, 'Occupancy Constancy', 'r')
    make_pr_curve(fl_match, fl_score, 'Feature Learning', 'b')

    #plt.title('Location Matching')

    plt.savefig('pr_curve_constancy.png')

def make_filter_plots():
    fl_filter = np.loadtxt('fl_filter.csv')
    fl_prob   = np.loadtxt('fl_filter_prob.csv')

    ib_filter = np.loadtxt('ib_filter.csv')
    ib_score  = np.loadtxt('ib_filter_score.csv')

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)

    make_pr_curve(ib_filter, ib_score, 'Ushani et. al 2017')
    make_pr_curve(fl_filter, fl_prob,  'Proposed')

    plt.title('Precision-Recall for Background Filter')

    plt.savefig('pr_curve_filter.png')

plt.switch_backend('agg')

make_constancy_plots()
#make_filter_plots()
