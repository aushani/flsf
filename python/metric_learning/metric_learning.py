import tensorflow as tf
from data import *
from batch_manager import *
import time
import sample
import sys

class MetricLearning:

    def __init__(self, data_manager, exp_name="exp"):
        self.exp_name = exp_name

        self.width  = data_manager.width
        self.length = data_manager.length
        self.height = data_manager.height
        self.dim_data = self.width * self.length * self.height

        self.validation_set = data_manager.validation_set

        self.latent_dim = 10

        self.default_keep_prob = 0.8

        # Inputs
        self.occ    = tf.placeholder(tf.float32, shape=[None, self.width, self.length, self.height], name='occ')

        self.occ1    = tf.placeholder(tf.float32, shape=[None, self.width, self.length, self.height], name='occ1')
        self.occ2    = tf.placeholder(tf.float32, shape=[None, self.width, self.length, self.height], name='occ2')

        self.filter  = tf.placeholder(tf.int32, shape=[None, self.width, self.length], name='filter')

        self.flow  = tf.placeholder(tf.float32, shape=[None, self.width, self.length, 2], name='flow')

        # Dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Encoder
        self.encoding = self.get_encoding(self.occ)

        # Metric distance
        metric_loss = self.make_metric_distance(self.occ1, self.occ2, self.flow, self.filter)

        # Filter
        self.make_filter(self.occ1)

        # Loss
        self.loss = self.normalized_filter_loss + metric_loss

        # Optimizer
        self.opt = tf.train.AdamOptimizer(1e-4)

        var_list = tf.trainable_variables()
        self.train_step = self.opt.minimize(self.loss,  var_list = var_list)

        # Session
        self.sess = tf.Session()

        # Summaries
        metric_loss_sum = tf.summary.scalar('metric distance loss', metric_loss)
        #filtered_metric_loss_sum = tf.summary.scalar('filtered metric distance loss', tf.reduce_mean(filtered_metric_loss))

        filter_loss_sum = tf.summary.scalar('filter loss', tf.reduce_mean(self.normalized_filter_loss))
        filter_acc_sum = tf.summary.scalar('filter accuracy', self.filter_accuracy)

        bg_acc_sum = tf.summary.scalar('filter accuracy background', self.bg_accuracy)
        fg_acc_sum = tf.summary.scalar('filter accuracy foreground', self.fg_accuracy)

        total_loss_sum = tf.summary.scalar('total loss', self.loss)

        self.summaries = tf.summary.merge_all()

    def get_encoding(self, occ):
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            occ_do = tf.nn.dropout(occ, self.keep_prob)

            l1 = tf.contrib.layers.conv2d(occ_do, num_outputs = 100, kernel_size = 9, activation_fn = tf.nn.leaky_relu, scope='l1')
            l1_do = tf.nn.dropout(l1, self.keep_prob)

            l2 = tf.contrib.layers.conv2d(l1_do, num_outputs = 50, kernel_size = 3, activation_fn = tf.nn.leaky_relu, scope='l2')
            l2_do = tf.nn.dropout(l2, self.keep_prob)

            l3 = tf.contrib.layers.conv2d(l2_do, num_outputs = 25, kernel_size = 3, activation_fn = tf.nn.leaky_relu, scope='l3')
            l3_do = tf.nn.dropout(l3, self.keep_prob)

            latent = tf.contrib.layers.conv2d(l3_do, num_outputs = self.latent_dim, kernel_size = 1, activation_fn = tf.nn.leaky_relu, scope='latent')
            latent_do = tf.nn.dropout(latent, self.keep_prob)

        return latent_do

    def make_filter(self, occ):
        encoding = self.get_encoding(occ)

        self.pred_filter = tf.contrib.layers.conv2d(encoding, num_outputs = 2, kernel_size = 3, activation_fn = tf.nn.leaky_relu, scope='filter')

        self.filter_probs = tf.nn.softmax(logits = self.pred_filter)

        # Make filter loss
        invalid_filter = self.filter < 0
        num_valid = tf.reduce_sum(tf.cast(tf.logical_not(invalid_filter), tf.float32))

        masked_filter = tf.where(invalid_filter, tf.zeros_like(self.filter), self.filter)

        filter_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_filter,
                                                                     logits=self.pred_filter)

        masked_filter_loss = tf.where(invalid_filter, tf.zeros_like(filter_loss), filter_loss)
        total_filter_loss = tf.reduce_sum(masked_filter_loss)
        self.normalized_filter_loss = total_filter_loss / num_valid

        # Filter accuracy
        ml_filter = tf.argmax(self.pred_filter, 3)
        correct_filter = tf.equal(self.filter, tf.cast(ml_filter, tf.int32))

        num_correct = tf.reduce_sum(tf.cast(correct_filter, tf.float32))

        self.filter_accuracy = num_correct / num_valid

        is_background = tf.equal(self.filter, 0)
        correct_background = tf.logical_and(tf.equal(tf.cast(ml_filter, tf.int32), 0), is_background)
        num_background = tf.reduce_sum(tf.cast(is_background, tf.float32))
        num_background_correct = tf.reduce_sum(tf.cast(correct_background, tf.float32))
        self.bg_accuracy = num_background_correct / num_background

        is_foreground = tf.equal(self.filter, 1)
        correct_foreground = tf.logical_and(tf.equal(tf.cast(ml_filter, tf.int32), 1), is_foreground)
        num_foreground = tf.reduce_sum(tf.cast(is_foreground, tf.float32))
        num_foreground_correct = tf.reduce_sum(tf.cast(correct_foreground, tf.float32))
        self.fg_accuracy = num_foreground_correct / num_foreground

    def make_metric_distance(self, occ1, occ2, true_flow, true_filter):
        latent1 = self.get_encoding(occ1)
        latent2 = self.get_encoding(occ2)

        invalid_filter = self.filter < 0
        valid_filter = tf.cast(tf.logical_not(invalid_filter), tf.float32)

        print 'Latent encodings'
        print latent1.shape
        print latent2.shape

        total_loss = 0
        count = 0

        ds = [-15, -8, -4, -2, -1, 0, 1, 2, 4, 8, 15]
        decimation = 2

        for di in ds:
            i0 = -di
            i1 = self.width - di
            i0 = np.clip(i0, a_min=0, a_max=None)
            i1 = np.clip(i1, a_min=None, a_max=self.width)

            for dj in ds:
                j0 = -dj
                j1 = self.length - dj
                j0 = np.clip(j0, a_min=0, a_max=None)
                j1 = np.clip(j1, a_min=None, a_max=self.length)

                l1 = latent1[:, i0:i1:decimation, j0:j1:decimation, :]
                l2 = latent2[:, (i0+di):(i1+di):decimation, (j0+dj):(j1+dj):decimation, :]
                dist_latent = tf.reduce_sum(tf.squared_difference(l1, l2), axis=3, name='dist_%d_%d' % (di, dj))

                err_x = true_flow[:, i0:i1:decimation, j0:j1:decimation, 0] - di
                err_y = true_flow[:, i0:i1:decimation, j0:j1:decimation, 1] - dj
                err2 = tf.multiply(err_x, err_x) + tf.multiply(err_y, err_y)

                vf = valid_filter[:, i0:i1:decimation, j0:j1:decimation]

                #print 'Dist shape'
                #print dist_latent.shape
                #print dist_latent

                #print 'Err shape'
                #print err2.shape

                norm_err2 = tf.clip_by_value(err2 - 1.0, -1, 3)
                val = norm_err2 * dist_latent

                #print 'Val shape'
                #print val.shape
                #print vf.shape

                total_loss += tf.reduce_sum(vf*tf.sigmoid(val))
                count += tf.reduce_sum(val)

        return total_loss / tf.cast(count, tf.float32)

    def eval_encoder(self, occ):
        fd = {self.occ: occ, self.keep_prob: 1}
        encoding = self.encoding.eval(session = self.sess, feed_dict = fd)

        return encoding

    def eval_filter_prob(self, occ):
        if len(occ.shape) == 3:
            occ = np.expand_dims(occ, 0)
        fd = {self.occ1: occ, self.keep_prob: 1}

        probs = self.filter_probs.eval(session = self.sess, feed_dict = fd)

        return probs

    def restore(self, filename):
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        print 'Restored model from', filename

    def train(self, batch_manager, start_iter = 0):
        if start_iter == 0:
            # Initialize variables
            self.sess.run(tf.global_variables_initializer())

        # Set up writer
        self.writer = tf.summary.FileWriter('./%s/logs' % (self.exp_name), self.sess.graph)

        # Save checkpoints
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        fd_valid = { self.occ1:        self.validation_set.occ1,
                     self.occ2:        self.validation_set.occ2,
                     self.filter:      self.validation_set.filter,
                     self.flow:        self.validation_set.flow,
                     self.keep_prob:   1.0,
                   }

        iteration = start_iter

        it_save = 1000
        it_plot = 1000
        it_summ = 100
        it_stat = 1000

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

                print '\tTook %5.3f sec to generate summaries' % (toc - tic)

            if iteration % it_save == 0:
                tic = time.time()
                save_path = saver.save(self.sess, "./%s/model.ckpt" % (self.exp_name), global_step = iteration)
                toc = time.time()

                #print '\tTook %5.3f sec to save checkpoint' % (toc - tic)

            if iteration % it_plot == 0:
                tic = time.time()
                #self.make_metric_plot(self.validation_set, save='%s/metric_%010d.png' % (self.exp_name, iteration / it_plot))
                self.make_filter_plot(self.validation_set, save='%s/filter_%010d.png' % (self.exp_name, iteration / it_plot))
                toc = time.time()

                print '\tTook %5.3f sec to make plots' % (toc - tic)


            if iteration % it_stat == 0 and iteration > 0:
                tot_acc, bg_acc, fg_acc = self.sess.run([self.filter_accuracy, self.bg_accuracy, self.fg_accuracy], feed_dict = fd_valid)

                print 'Iteration %d' % (iteration)
                print '  Filter accuracy = %5.3f %%' % (100.0 * tot_acc)
                print '      BG accuracy = %5.3f %%' % (100.0 * bg_acc)
                print '      FG accuracy = %5.3f %%' % (100.0 * fg_acc)
                print ''
                print '  Loading data at %5.3f ms / iteration' % (t_data*1000.0/it_stat)
                print '  Training at %5.3f ms / iteration' % (t_train*1000.0/it_stat)
                print ''

                t_data = 0
                t_train = 0


            tic = time.time()
            samples = batch_manager.get_next_batch()
            toc = time.time()

            t_data += (toc - tic)

            tic = time.time()
            fd = { self.occ1:        samples.occ1,
                   self.occ2:        samples.occ2,
                   self.filter:      samples.filter,
                   self.flow:        samples.flow,
                   self.keep_prob:   self.default_keep_prob,
                 }
            self.train_step.run(session = self.sess, feed_dict = fd)
            toc = time.time()

            t_train += (toc - tic)

            iteration += 1

    def make_metric_plot(self, dataset, save=None, show=False):
        encodings1 = self.eval_encoding(dataset.occ1)
        encodings2 = self.eval_encoding(dataset.occ2)

        n = encodings1.shape[0]

        for i in n:
            e1 = encodings1[i, :, :, :]
            e2 = encodings2[i, :, :, :]

            f1 = dataset.filter[i, :, :]

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

        prob_background = probs[:, :, :, 0]
        prob_foreground = probs[:, :, :, 1]

        true_label = true_label.flatten()
        prob_background = prob_background.flatten()
        prob_foreground = prob_foreground.flatten()

        is_background = (true_label == 0)
        is_foreground = (true_label == 1)

        valid = np.logical_or(is_background, is_foreground)

        plt.clf()
        self.make_pr_curve(is_background[valid], prob_background[valid], 'Background')
        self.make_pr_curve(is_foreground[valid], prob_foreground[valid], 'Foreground')

        plt.grid()

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
        #plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

if __name__ == '__main__':
    plt.switch_backend('agg')

    dm = DataManager()
    dm.make_validation(50)

    ml = MetricLearning(dm)

    bm = BatchManager(dm)

    if len(sys.argv) > 1:
        load_iter = int(sys.argv[1])
        print 'Loading from iteration %d' % (load_iter)

        ml.restore('model.ckpt-%d' % (load_iter))
        ml.train(bm, start_iter = load_iter+1)
    else:
        ml.train(bm)
