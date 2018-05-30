import tensorflow as tf
from flow_data import *
from old_flow_sample import *
from old_batch_manager import *
import time
import matplotlib.pyplot as plt
import sklearn.metrics
import sys

class Icra2017Learning:

    def __init__(self, flow_data=None, exp_name='old_const'):
        self.exp_name = exp_name

        if flow_data:
            self.validation_set = OldFlowSampleSet(flow_data.validation_set)
            self.batches = OldBatchManager(flow_data)

        self.height = 13

        # Inputs
        self.fv    = tf.placeholder(tf.float32, shape=[None, 3*self.height])

        # Output
        self.match       = tf.placeholder(tf.float32, shape=[None,])
        y_ = 2*self.match - 1

        # Train LR
        w = tf.Variable(tf.zeros([1, 3*self.height]))
        b = tf.Variable(tf.zeros([1]))

        self.score = tf.matmul(w, self.fv, transpose_b = True) + b
        cost = tf.reduce_mean(tf.log(1 + tf.exp(-y_ * self.score)))
        self.loss = cost

        self.opt = tf.train.FtrlOptimizer(0.01)
        self.train_step = self.opt.minimize(self.loss)

        y = tf.sign(self.score)
        correct_prediction = tf.equal(y, tf.sign(y_))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Optimizer
        self.opt = tf.train.AdamOptimizer(1e-6)

        var_list = tf.trainable_variables()
        self.train_step = self.opt.minimize(self.loss, var_list = var_list)

        # Session
        self.sess = tf.Session()

        # Summaries
        total_loss_sum = tf.summary.scalar('total loss', self.loss)

        self.summaries = tf.summary.merge_all()

    def restore(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        print 'Restored model from', filename

    def train(self, start_iter = 0):
        if start_iter == 0:
            # Initialize variables
            self.sess.run(tf.global_variables_initializer())

        # Set up writer
        self.writer = tf.summary.FileWriter('./%s/logs' % (self.exp_name), self.sess.graph)

        # Save checkpoints
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        fd_valid = {
                        self.fv :      self.validation_set.feature_vector,
                        self.match :   self.validation_set.match,
                   }

        iteration = start_iter + 1

        it_save = 100000
        it_plot = 100000
        it_summ = 100

        t_sum = 0
        t_save = 0
        t_plots = 0
        t_train = 0
        t_data = 0

        while True:

            if iteration % it_summ == 0:
                tic = time.time()
                summary = self.sess.run(self.summaries, feed_dict=fd_valid)
                self.writer.add_summary(summary, iteration)
                toc = time.time()

                #print '\tTook %5.3f sec to generate summaries' % (toc - tic)

            if iteration % it_save == 0:
                tic = time.time()
                save_path = saver.save(self.sess, "./%s/model.ckpt" % (self.exp_name), global_step = iteration)
                toc = time.time()

                #print '\tTook %5.3f sec to save checkpoint' % (toc - tic)

            if iteration % it_plot == 0:
                print 'Iteration %d' % (iteration)

                self.make_plots(save='%s/res_%010d.png' % (self.exp_name, iteration / it_plot))
                #self.make_match_plot(valid_set, save='%s/metric_%010d.png' % iteration)

                print '  Match accuracy = %5.3f %%' % (100.0 * self.accuracy.eval(session = self.sess, feed_dict = fd_valid))

                if iteration > 0:
                    print '  Loading data at %5.3f ms / iteration' % (t_data*1000.0/iteration)
                    print '  Training at %5.3f ms / iteration' % (t_train*1000.0/iteration)

                print ''

            tic = time.time()
            samples = self.batches.get_next_batch()
            toc = time.time()

            t_data += (toc - tic)

            tic = time.time()
            fd = {
                   self.fv:     samples.feature_vector,
                   self.match:  samples.match,
                 }
            self.train_step.run(session = self.sess, feed_dict = fd)
            toc = time.time()

            t_train += (toc - tic)

            iteration += 1

    def eval_scores(self, feature_vector):
        fd = {self.fv: feature_vector}
        scores = self.score.eval(session = self.sess, feed_dict = fd)
        scores = np.squeeze(scores)

        return scores

    def make_plots(self, save=None):
        # Get data

        scores = self.eval_scores(self.validation_set.feature_vector)

        is_background = self.validation_set.foreground == 0
        is_foreground = self.validation_set.foreground == 1

        match = self.validation_set.match

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(20, 20)

        plt.subplot(3, 2, 1)
        self.make_pr_curve(match[is_background], scores[is_background], 'Background')
        self.make_pr_curve(match[is_foreground], scores[is_foreground], 'Foreground')
        self.make_pr_curve(match, scores, 'All')

        for i, cumul in enumerate([False, True]):
            if i == 0:
                plot_type = 'Histogram'
            if i == 1:
                plot_type = 'CDF'

            plt.subplot(3, 2, 3 + 2*i)
            plt.title('Occupancy Constancy %s' % plot_type)
            idx_match = match == 1
            idx_nonmatch = match == 0

            plt.hist(scores[idx_match], bins=100, range=(-1.0, 1.0), cumulative=cumul,
                    normed=cumul, histtype='step', label='Matching (%d)' % np.sum(idx_match))
            plt.hist(scores[idx_nonmatch], bins=100, range=(-1.0, 1.0), cumulative=cumul,
                    normed=cumul, histtype='step', label='Non-Matching (%d)' % np.sum(idx_nonmatch))
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('Score')

            plt.subplot(3, 2, 4 + 2*i)
            plt.title('Occupancy Constancy %s (Foreground only)' % plot_type)
            idx_match = (match == 1) & (is_foreground)
            idx_nonmatch = (match == 0) & (is_foreground)

            plt.hist(scores[idx_match], bins=100, range=(-1.0, 1.0), cumulative=cumul,
                    normed=cumul, histtype='step', label='Matching (%d)' % np.sum(idx_match))
            plt.hist(scores[idx_nonmatch], bins=100, range=(-1.0, 1.0), cumulative=cumul,
                    normed=cumul, histtype='step', label='Non-Matching (%d)' % np.sum(idx_nonmatch))
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('Distance')

        if save:
            plt.savefig(save)

        plt.clf()

    def make_pr_curve(self, true_label, pred_label, label):
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(true_label, pred_label)

        plt.plot(recall, precision, label=label)

        #plt.step(recall, precision, color='b', alpha=0.2, where='post')
        #plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.legend(loc='lower left')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid()
        #plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

if __name__ == '__main__':
    plt.switch_backend('agg')

    flow_dm = FlowDataManager()
    flow_dm.make_validation(10000)

    il = Icra2017Learning(flow_dm)

    if len(sys.argv) > 1:
        load_iter = int(sys.argv[1])
        print 'Loading from iteration %d' % (load_iter)

        il.restore('model.ckpt-%d' % (load_iter))
        il.train(start_iter = load_iter+1)
    else:
        il.train()
