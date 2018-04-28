import tensorflow as tf
from data import *
import time
import sample
import sys

plt.switch_backend('agg')

class MetricLearning:

    def __init__(self, data_manager):

        self.dm = data_manager

        self.width = self.dm.width
        self.length = self.dm.length
        self.height = self.dm.height
        self.dim_data = self.width * self.length * self.height

        self.n_classes = sample.n_classes

        self.latent_dim = 25

        self.default_keep_prob = 0.8

        # Inputs
        self.occ1    = tf.placeholder(tf.float32, shape=[None, self.width, self.length, self.height], name='occ1')
        self.occ2    = tf.placeholder(tf.float32, shape=[None, self.width, self.length, self.height], name='occ2')

        # Dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.match       = tf.placeholder(tf.float32, shape=[None,], name='match')
        self.true_filter = tf.placeholder(tf.int32, shape=[None,], name='true_filter')

        # Filter
        self.pred_filter = self.make_filter(self.occ1)
        self.filter_probs = tf.nn.softmax(logits = self.pred_filter)

        # Metric distance loss
        self.dist, metric_loss = self.make_metric_distance(self.occ1, self.occ2, self.match)

        # Filter loss
        filter_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.true_filter, logits=self.pred_filter)

        correct_filter = tf.equal(self.true_filter, tf.cast(tf.argmax(self.pred_filter, 1), tf.int32))
        self.filter_accuracy = tf.reduce_mean(tf.cast(correct_filter, tf.float32))

        #self.loss = metric_loss + 1e-3 * filter_loss
        filter_pass_prob = self.filter_probs[:,1]
        filtered_metric_loss = tf.multiply(filter_pass_prob, metric_loss)

        loss_all = filtered_metric_loss + filter_loss
        self.loss = tf.reduce_mean(loss_all)

        # Optimizer
        self.opt = tf.train.AdamOptimizer(1e-4)

        var_list = tf.trainable_variables()
        self.train_step = self.opt.minimize(self.loss,  var_list = var_list)

        # Session
        self.sess = tf.Session()

        # Summaries
        metric_loss_sum = tf.summary.scalar('metric distance loss', tf.reduce_mean(metric_loss))
        filtered_metric_loss_sum = tf.summary.scalar('filtered metric distance loss', tf.reduce_mean(filtered_metric_loss))

        filter_loss_sum = tf.summary.scalar('filter loss', tf.reduce_mean(filter_loss))

        filter_acc_sum = tf.summary.scalar('filter accuracy', self.filter_accuracy)

        total_loss_sum = tf.summary.scalar('total loss', self.loss)

        self.summaries = tf.summary.merge_all()


    def make_encoder(self, occ):
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            flatten = tf.contrib.layers.flatten(occ)
            flatten_do = tf.nn.dropout(flatten, self.keep_prob)

            l1 = tf.contrib.layers.fully_connected(flatten_do, 200, activation_fn = tf.nn.leaky_relu, scope='l1')
            l1_do = tf.nn.dropout(l1, self.keep_prob)

            l2 = tf.contrib.layers.fully_connected(l1_do, 100, activation_fn = tf.nn.leaky_relu, scope='l2')
            l2_do = tf.nn.dropout(l2, self.keep_prob)

            l3 = tf.contrib.layers.fully_connected(l2_do, 50, activation_fn = tf.nn.leaky_relu, scope='l3')
            l3_do = tf.nn.dropout(l3, self.keep_prob)

            latent = tf.contrib.layers.fully_connected(l3_do, self.latent_dim, activation_fn = tf.nn.leaky_relu, scope='latent')
            latent_do = tf.nn.dropout(latent, self.keep_prob)

        return latent_do

    def make_filter(self, occ):
        latent = self.make_encoder(occ)

        output = tf.contrib.layers.fully_connected(latent, 2, activation_fn = tf.nn.leaky_relu, scope="filter")

        return output

    def make_metric_distance(self, occ1, occ2, match):
        latent1 = self.make_encoder(occ1)
        latent2 = self.make_encoder(occ2)

        dist = tf.reduce_sum(tf.squared_difference(latent1, latent2), axis=1)

        #loss = match * tf.nn.sigmoid(dist) + (1 - match) * tf.nn.sigmoid(-dist)
        #loss = match * dist + (1 - match) *(-dist)

        match_loss = tf.nn.sigmoid(dist - 10)
        not_match_loss = 1 - tf.nn.sigmoid(dist - 10)

        loss = match * match_loss + (1 - match) * not_match_loss

        #return dist, tf.reduce_mean(loss)
        return dist, loss

    def eval_dist(self, occ1, occ2):
        fd = {self.occ1: occ1, self.occ2: occ2, self.keep_prob: 1}
        dist = self.dist.eval(session = self.sess, feed_dict = fd)

        return dist

    def eval_filter_prob(self, occ):
        fd = {self.occ1: occ, self.keep_prob: 1}

        probs = self.filter_probs.eval(session = self.sess, feed_dict = fd)

        return probs

    def restore(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        print 'Restored model from', filename

    def train(self, start_iter = 0):
        if start_iter == 0:
            # Initialize variables
            self.sess.run(tf.global_variables_initializer())

        # Set up writer
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

        # Save checkpoints
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        valid_set = self.dm.validation_set

        fd_valid = { self.occ1: valid_set.occ1,
                     self.occ2: valid_set.occ2,
                     self.match: valid_set.match,
                     self.true_filter: valid_set.filter,
                     self.keep_prob : 1.0,
                   }

        iteration = start_iter

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
                save_path = saver.save(self.sess, "./model.ckpt", global_step = iteration)
                toc = time.time()

                #print '\tTook %5.3f sec to save checkpoint' % (toc - tic)

            if iteration % it_plot == 0:
                print 'Iteration %d' % (iteration)

                self.make_metric_plot(valid_set, save='metric_%010d.png' % iteration)
                self.make_filter_plot(valid_set, save='filter_%010d.png' % iteration)

                print '  Filter accuracy = %5.3f %%' % (100.0 * self.filter_accuracy.eval(session = self.sess, feed_dict = fd_valid))

                if iteration > 0:
                    print '  Loading data at %5.3f ms / iteration' % (t_data*1000.0/iteration)
                    print '  Training at %5.3f ms / iteration' % (t_train*1000.0/iteration)

                print ''

            tic = time.time()
            samples = self.dm.get_next_samples(1)
            toc = time.time()

            t_data += (toc - tic)

            tic = time.time()
            fd = { self.occ1: samples.occ1,
                   self.occ2: samples.occ2,
                   self.match: samples.match,
                   self.true_filter: samples.filter,
                   self.keep_prob: self.default_keep_prob,
                 }
            self.train_step.run(session = self.sess, feed_dict = fd)
            toc = time.time()

            t_train += (toc - tic)

            iteration += 1

    def make_metric_plot(self, dataset, save=None, show=False):
        distances = self.eval_dist(dataset.occ1, dataset.occ2)
        probs = self.eval_filter_prob(dataset.occ1)

        plt.clf()

        for cutoff in [0.2, 0.4, 0.6, 0.8]:
            idx = probs[:, 0] > cutoff

            if np.sum(idx) == 0:
                continue

            dists_c = distances[idx]
            match_c = dataset.match[idx]

            self.make_pr_curve(match_c, -dists_c, 'Cutoff = %5.3f' % cutoff)

        self.make_pr_curve(dataset.match, -distances, 'All')

        if save:
            plt.savefig(save)

    def make_filter_plot(self, dataset, save=None, show=False):
        probs = self.eval_filter_prob(dataset.occ1)

        true_label = dataset.filter
        pred_label = probs[:, 1]

        plt.clf()
        self.make_pr_curve(true_label, pred_label, 'Filter')

        if save:
            plt.savefig(save)

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
    dm = DataManager()
    dm.make_validation(10000)

    validation = dm.validation_set

    ml = MetricLearning(dm)

    if len(sys.argv) > 1:
        load_iter = int(sys.argv[1])
        print 'Loading from iteration %d' % (load_iter)

        ml.restore('model.ckpt-%d' % (load_iter))
        ml.train(start_iter = load_iter+1)
    else:
        ml.train()
