import tensorflow as tf
from filter_data import *
from old_filter_sample import *
from old_filter_batch_manager import *
import time
import matplotlib.pyplot as plt
import sklearn.metrics
import sys

class Icra2017Background:

    def __init__(self, filter_data=None, exp_name='old_background'):
        self.exp_name = exp_name

        self.full_input_width  = 167 + 24
        self.full_input_length = 167 + 24
        self.full_output_width  = 167
        self.full_output_length = 167
        self.full_height = 13

        if filter_data:
            self.validation_set = OldFilterSampleSet(filter_data.validation_set)
            self.batches = OldFilterBatchManager(filter_data)

        self.height = 13

        # Inputs
        self.fv = tf.placeholder(tf.float32, shape=[None, self.full_input_width, self.full_input_length, 2*self.height])

        # Output
        self.filter = tf.placeholder(tf.float32, shape=[None, self.full_output_width, self.full_output_length], name='true_label')

        self.score = tf.contrib.layers.conv2d(self.fv, num_outputs = 1, kernel_size = 5, activation_fn = None, padding = 'VALID')

        print self.score.shape
        i0 = 10
        i1 = self.score.shape[1] - 10

        self.score = self.score[:, i0:i1, i0:i1, 0]
        print self.score.shape

        assert self.score.shape[1] == self.full_output_width
        assert self.score.shape[2] == self.full_output_length

        # Filter weighting
        num_neg = 12119773.0
        num_pos = 336639.0
        denom = 1.0 / num_neg + 1.0 / num_pos
        self.neg_weight = (1.0 / num_neg) / denom
        self.pos_weight = (1.0 / num_pos) / denom

        cost = tf.log(1 + tf.exp(-self.filter * self.score))

        neg_loss = self.neg_weight * cost
        pos_loss = self.pos_weight * cost

        is_background = self.filter < 0
        is_foreground = self.filter > 0

        neg_filter_loss = tf.where(is_background, neg_loss, tf.zeros_like(neg_loss))
        pos_filter_loss = tf.where(is_foreground, pos_loss, tf.zeros_like(pos_loss))
        weighted_filter_loss = neg_filter_loss + pos_filter_loss

        cost = tf.reduce_mean(weighted_filter_loss)
        self.loss = cost

        self.opt = tf.train.FtrlOptimizer(0.01)
        self.train_step = self.opt.minimize(self.loss)

        y = tf.sign(self.score)
        valid_prediction = tf.not_equal(self.filter, 0)
        correct_prediction = tf.equal(y, tf.sign(self.filter))
        correct_prediction = correct_prediction & valid_prediction

        self.valid_prediction = tf.cast(valid_prediction, tf.float32)
        self.correct_prediction = tf.cast(correct_prediction, tf.float32)

        self.accuracy = tf.reduce_sum(self.correct_prediction) / tf.reduce_sum(self.valid_prediction)

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
                        self.fv     :   self.validation_set.feature_vector,
                        self.filter :   self.validation_set.filter,
                   }

        iteration = start_iter + 1

        it_save = 10000
        it_plot = 10000
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

                corr, valid, acc = self.sess.run([self.correct_prediction, self.valid_prediction, self.accuracy],
                                                 feed_dict = fd_valid)

                #print ' %f / %f ' % (np.sum(corr), np.sum(valid))
                print '  Filter accuracy = %5.3f %%' % (100.0 * acc)

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
                   self.filter: samples.filter,
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
        labels = np.copy(self.validation_set.filter)

        scores = scores[:]
        labels = labels[:]

        is_background = labels < 0
        is_foreground = labels > 0
        is_invalid    = labels ==  0
        is_valid      = ~is_invalid

        labels[is_background] = 0
        labels[is_foreground] = 1 #redundant

        scores = scores[is_valid]
        labels = labels[is_valid]

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(20, 20)

        self.make_pr_curve(labels, scores, 'Background Filter')

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

    filter_dm = FilterDataManager()
    filter_dm.make_validation(50)

    ib = Icra2017Background(filter_dm)

    if len(sys.argv) > 1:
        load_iter = int(sys.argv[1])
        print 'Loading from iteration %d' % (load_iter)

        ib.restore('model.ckpt-%d' % (load_iter))
        ib.train(start_iter = load_iter+1)
    else:
        ib.train()
