import tensorflow as tf
from flow_data import *
from filter_data import *
from batch_manager import *
import time
import sys
import sklearn

class MetricLearning:

    def __init__(self, filter_data=None, flow_data=None, exp_name="exp"):
        self.exp_name = exp_name

        self.full_width  = 167
        self.full_length = 167
        self.full_height = 13

        self.patch_width  = 15
        self.patch_length = 15
        self.patch_height = 13

        if filter_data:
            self.filter_validation_set = filter_data.validation_set
            self.filter_batches = BatchManager(filter_data)

        if flow_data:
            self.flow_validation_set = flow_data.validation_set
            self.flow_batches = BatchManager(flow_data)

        self.latent_dim = 25

        self.default_keep_prob = 0.5

        # Filter weighting
        num_neg = 12119773.0
        num_pos = 336639.0
        denom = 1.0 / num_neg + 1.0 / num_pos
        self.neg_weight = (1.0 / num_neg) / denom
        self.pos_weight = (1.0 / num_pos) / denom

        # Inputs
        self.full_occ    = tf.placeholder(tf.float32, shape=[None, self.full_width, self.full_length, self.full_height],
                                          name='full_occ')

        self.patch1      = tf.placeholder(tf.float32, shape=[None, self.patch_width, self.patch_length, self.patch_height], name='patch1')
        self.patch2      = tf.placeholder(tf.float32, shape=[None, self.patch_width, self.patch_length, self.patch_height], name='patch2')

        self.err2        = tf.placeholder(tf.float32, shape=[None,], name='err2')

        # Ground truth outputs
        self.filter  = tf.placeholder(tf.int32, shape=[None, self.full_width, self.full_length], name='filter')
        self.match   = tf.placeholder(tf.int32, shape=[None,], name='match')

        self.patch1_filter = tf.placeholder(tf.int32, shape=[None,], name='patch1_filter')

        # Dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Encoder
        self.encoding = self.get_encoding(self.full_occ)

        # Filter
        self.make_filter(self.full_occ)

        # Metric distance
        self.metric_loss, self.dist, self.pred_patch_prob_fg = self.make_metric_distance(self.patch1, self.patch2,
                                                                                    self.match, self.err2, self.patch1_filter)

        # Loss
        self.total_loss = self.normalized_filter_loss + self.metric_loss

        # Optimizers
        filter_opt = tf.train.AdamOptimizer(1e-6)
        all_vars = tf.trainable_variables()
        self.filter_train_step = filter_opt.minimize(self.normalized_filter_loss, var_list = all_vars)

        metric_dist_opt = tf.train.AdamOptimizer(1e-6)
        encoder_vars = tf.trainable_variables(scope='Encoder')
        self.metric_dist_train_step = metric_dist_opt.minimize(self.metric_loss, var_list = encoder_vars)

        # Session
        self.sess = tf.Session()

        # Summaries
        metric_loss_sum = tf.summary.scalar('metric distance loss', self.metric_loss)

        filter_loss_sum = tf.summary.scalar('filter loss', self.normalized_filter_loss)
        filter_acc_sum = tf.summary.scalar('filter accuracy', self.filter_accuracy)

        bg_acc_sum = tf.summary.scalar('filter accuracy background', self.bg_accuracy)
        fg_acc_sum = tf.summary.scalar('filter accuracy foreground', self.fg_accuracy)

        total_loss_sum = tf.summary.scalar('total loss', self.total_loss)

        self.summaries = tf.summary.merge_all()

    def get_encoding(self, occ, padding = 'SAME'):
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            occ_do = tf.nn.dropout(occ, self.keep_prob)

            l1 = tf.contrib.layers.conv2d(occ_do, num_outputs = 200, kernel_size = 9,
                    activation_fn = tf.nn.leaky_relu, padding = padding, scope='l1')
            l1_do = tf.nn.dropout(l1, self.keep_prob)

            l2 = tf.contrib.layers.conv2d(l1_do, num_outputs = 100, kernel_size = 3,
                    activation_fn = tf.nn.leaky_relu, padding = padding, scope='l2')
            l2_do = tf.nn.dropout(l2, self.keep_prob)

            l3 = tf.contrib.layers.conv2d(l2_do, num_outputs = 50, kernel_size = 3,
                    activation_fn = tf.nn.leaky_relu, padding = padding, scope='l3')
            l3_do = tf.nn.dropout(l3, self.keep_prob)

            latent = tf.contrib.layers.conv2d(l3_do, num_outputs = self.latent_dim, kernel_size = 1,
                    activation_fn = tf.nn.leaky_relu, padding = padding, scope='latent')
            latent_do = tf.nn.dropout(latent, self.keep_prob)

        return latent_do

    def get_filter(self, occ, padding = 'SAME'):
        encoding = self.get_encoding(occ, padding = padding)

        with tf.variable_scope('Filter', reuse=tf.AUTO_REUSE):
            pred_filter = tf.contrib.layers.conv2d(encoding, num_outputs = 2, kernel_size = 3,
                    activation_fn = tf.nn.leaky_relu, padding = padding, scope='l1')

            filter_probs = tf.nn.softmax(logits = pred_filter)

        return pred_filter, filter_probs

    def make_filter(self, occ):
        self.pred_filter, self.filter_probs = self.get_filter(occ)

        is_background = tf.equal(self.filter, 0)
        is_foreground = tf.equal(self.filter, 1)
        invalid_filter = self.filter < 0
        valid_filter = tf.logical_not(invalid_filter)
        num_valid = tf.reduce_sum(tf.cast(valid_filter, tf.float32))

        # Make filter loss
        masked_filter = tf.where(invalid_filter, tf.zeros_like(self.filter), self.filter)

        filter_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_filter,
                                                                     logits=self.pred_filter)

        neg_loss = self.neg_weight * filter_loss
        pos_loss = self.pos_weight * filter_loss

        neg_filter_loss = tf.where(is_background, neg_loss, tf.zeros_like(neg_loss))
        pos_filter_loss = tf.where(is_foreground, pos_loss, tf.zeros_like(pos_loss))

        weighted_filter_loss = neg_filter_loss + pos_filter_loss
        total_filter_loss = tf.reduce_sum(weighted_filter_loss)
        self.normalized_filter_loss = total_filter_loss / num_valid

        # Filter accuracy
        ml_filter = tf.argmax(self.pred_filter, 3)
        correct_filter = tf.equal(self.filter, tf.cast(ml_filter, tf.int32))

        num_correct = tf.reduce_sum(tf.cast(correct_filter, tf.float32))

        self.filter_accuracy = num_correct / num_valid

        correct_background = tf.logical_and(tf.equal(tf.cast(ml_filter, tf.int32), 0), is_background)
        num_background = tf.reduce_sum(tf.cast(is_background, tf.float32))
        num_background_correct = tf.reduce_sum(tf.cast(correct_background, tf.float32))
        self.bg_accuracy = num_background_correct / num_background

        correct_foreground = tf.logical_and(tf.equal(tf.cast(ml_filter, tf.int32), 1), is_foreground)
        num_foreground = tf.reduce_sum(tf.cast(is_foreground, tf.float32))
        num_foreground_correct = tf.reduce_sum(tf.cast(correct_foreground, tf.float32))
        self.fg_accuracy = num_foreground_correct / num_foreground

    def make_metric_distance(self, patch1, patch2, match, err2, true_patch_filter):
        latent1 = self.get_encoding(patch1, padding = 'VALID')
        latent2 = self.get_encoding(patch2, padding = 'VALID')

        pred_patch_filter, pred_patch_filter_prob = self.get_filter(patch1, padding = 'VALID')

        assert latent1.shape[1] == 3
        assert latent1.shape[2] == 3

        assert latent2.shape[1] == 3
        assert latent2.shape[2] == 3

        assert pred_patch_filter.shape[1] == 1
        assert pred_patch_filter.shape[2] == 1

        assert pred_patch_filter_prob.shape[1] == 1
        assert pred_patch_filter_prob.shape[2] == 1

        # Take center encoding
        latent1 = latent1[:, 1, 1, :]
        latent2 = latent2[:, 1, 1, :]
        #latent1 = tf.squeeze(latent1, axis=[1, 2])
        #latent2 = tf.squeeze(latent2, axis=[1, 2])

        pred_patch_filter = tf.squeeze(pred_patch_filter, axis=[1, 2])
        pred_patch_filter_prob = tf.squeeze(pred_patch_filter_prob, axis=[1, 2])
        prob_bg = pred_patch_filter_prob[:, 0]
        prob_fg = pred_patch_filter_prob[:, 1]

        # Spatial distance between patch1 and path2 (in grid dimensions)
        #err = tf.sqrt(err2)
        #err_thresh = 1.0           # 0.3 meters

        # Soft weights
        #match_weight = tf.clip_by_value(err_thresh - err, clip_value_min=0, clip_value_max=err_thresh)/err_thresh
        #non_match_weight = tf.clip_by_value(err, clip_value_min=0, clip_value_max=err_thresh)/err_thresh

        metric_dist2 = tf.reduce_sum(tf.squared_difference(latent1, latent2), axis=1)
        metric_dist = tf.sqrt(metric_dist2)

        # Loss based on distance
        max_dist2 = 10
        non_match_loss = tf.clip_by_value(max_dist2 - metric_dist2, clip_value_min=0, clip_value_max=max_dist2)
        match_loss = metric_dist2

        # Select which loss according to match flag
        is_match = match > 0
        metric_dist_loss = tf.where(is_match, match_loss, non_match_loss, 'loss_switch')

        # Weight according to filter prob (if it should be background)
        is_background = true_patch_filter == 0
        is_foreground = true_patch_filter == 1
        #weighted_loss = tf.where(is_background, tf.multiply(prob_fg, match_loss), match_loss)
        weighted_loss = tf.where(is_background, 0.1*metric_dist_loss, metric_dist_loss)

        # Penalize incorrect filter
        filter_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_patch_filter,
                                                                     logits=pred_patch_filter)

        net_loss = filter_loss + weighted_loss

        net_loss = tf.reduce_mean(net_loss)

        return net_loss, metric_dist, prob_fg

    def eval_dist(self, patch1, patch2):
        fd = {self.patch1: patch1, self.patch2: patch2, self.keep_prob: 1.0}
        dist = self.dist.eval(session = self.sess, feed_dict = fd)

        return dist

    def eval_patch_prob_fg(self, patch):
        fd = {self.patch1: patch, self.keep_prob: 1.0}
        prob_fg = self.pred_patch_prob_fg.eval(session = self.sess, feed_dict = fd)

        return prob_fg

    def eval_filter_prob(self, occ):
        if len(occ.shape) == 3:
            occ = np.expand_dims(occ, 0)

        fd = {self.full_occ: occ, self.keep_prob: 1.0}

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
        self.writer = tf.summary.FileWriter('./%s/logs' % (self.exp_name), self.sess.graph)

        # Save checkpoints
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        fd_valid = { self.full_occ:        self.filter_validation_set.occ,
                     self.filter:          self.filter_validation_set.filter,

                     self.patch1:          self.flow_validation_set.occ1,
                     self.patch2:          self.flow_validation_set.occ2,
                     self.err2:            self.flow_validation_set.err2,
                     self.match:           self.flow_validation_set.match,
                     self.patch1_filter:   self.flow_validation_set.filter,

                     self.keep_prob:   1.0,
                   }

        iteration = start_iter + 1

        it_save = 1000
        it_plot = 1000
        it_summ = 1000
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
                self.make_plots(save='%s/res_%010d.png' % (self.exp_name, iteration / it_plot))
                #self.make_metric_plot(self.flow_validation_set, save='%s/metric_%010d.png' % (self.exp_name, iteration / it_plot))
                #self.make_filter_plot(self.filter_validation_set, save='%s/filter_%010d.png' % (self.exp_name, iteration / it_plot))
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
            filter_samples = self.filter_batches.get_next_batch()
            flow_samples = self.flow_batches.get_next_batch()
            toc = time.time()

            t_data += (toc - tic)

            tic = time.time()
            fd = { self.full_occ:        filter_samples.occ,
                   self.filter:          filter_samples.filter,

                   self.patch1:          flow_samples.occ1,
                   self.patch2:          flow_samples.occ2,
                   self.err2:            flow_samples.err2,
                   self.match:           flow_samples.match,
                   self.patch1_filter:   flow_samples.filter,

                   self.keep_prob:       self.default_keep_prob,
                 }
            self.filter_train_step.run(session = self.sess, feed_dict = fd)
            self.metric_dist_train_step.run(session = self.sess, feed_dict = fd)
            toc = time.time()

            t_train += (toc - tic)

            iteration += 1

    def make_plots(self, save=None, show=False):
        # Get Data

        # Flow
        distances = self.eval_dist(self.flow_validation_set.occ1, self.flow_validation_set.occ2)
        prob_fg = self.eval_patch_prob_fg(self.flow_validation_set.occ1)
        match = self.flow_validation_set.match
        patch_is_foreground = self.flow_validation_set.filter == 1

        # Filter
        probs = self.eval_filter_prob(self.filter_validation_set.occ)

        true_label = self.filter_validation_set.filter

        prob_background = probs[:, :, :, 0]
        prob_foreground = probs[:, :, :, 1]

        true_label = true_label.flatten()
        prob_background = prob_background.flatten()
        prob_foreground = prob_foreground.flatten()

        is_background = (true_label == 0)
        is_foreground = (true_label == 1)

        valid = np.logical_or(is_background, is_foreground)

        # Make plots
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(20, 20)

        plt.subplot(3, 2, 2)
        self.make_pr_curve(is_background[valid], prob_background[valid], 'Pos. Class = Background')
        self.make_pr_curve(is_foreground[valid], prob_foreground[valid], 'Pos. Class = Foreground')
        plt.grid()
        plt.title('Filter Performance')

        plt.subplot(3, 2, 1)
        self.make_match_pr(distances, self.flow_validation_set.match, prob_fg, patch_is_foreground)
        plt.title('Match Distance PR')

        for i, cumul in enumerate([False, True]):
            if i == 0:
                plot_type = 'Histogram'
            if i == 1:
                plot_type = 'CDF'

            plt.subplot(3, 2, 3 + 2*i)
            plt.title('Metric Distance %s' % plot_type)
            idx_match = match == 1
            idx_nonmatch = match == 0

            plt.hist(distances[idx_match], bins=100, range=(0, 10.0), cumulative=cumul,
                    normed=cumul, histtype='step', label='Matching (%d)' % np.sum(idx_match))
            plt.hist(distances[idx_nonmatch], bins=100, range=(0, 10.0), cumulative=cumul,
                    normed=cumul, histtype='step', label='Non-Matching (%d)' % np.sum(idx_nonmatch))
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('Distance')

            plt.subplot(3, 2, 4 + 2*i)
            plt.title('Metric Distance %s (Foreground only)' % plot_type)
            idx_match = (match == 1) & (patch_is_foreground)
            idx_nonmatch = (match == 0) & (patch_is_foreground)

            plt.hist(distances[idx_match], bins=100, range=(0, 10.0), cumulative=cumul,
                    normed=cumul, histtype='step', label='Matching (%d)' % np.sum(idx_match))
            plt.hist(distances[idx_nonmatch], bins=100, range=(0, 10.0), cumulative=cumul,
                    normed=cumul, histtype='step', label='Non-Matching (%d)' % np.sum(idx_nonmatch))
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('Distance')

        if save:
            plt.savefig(save)


    def make_metric_plot(self, dataset, save=None, show=False):
        distances = self.eval_dist(dataset.occ1, dataset.occ2)
        prob_fg = self.eval_patch_prob_fg(dataset.occ1)
        err = np.sqrt(dataset.err2)

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(10, 20)

        plt.subplot(3, 1, 1)
        self.make_match_pr(distances, dataset.match, prob_fg, dataset.filter==1)

        plt.subplot(3, 1, 2)
        plt.title('Metric Distance CDF')
        idx_match = dataset.match == 1
        idx_nonmatch = dataset.match == 0

        plt.hist(distances[idx_match], bins=100, range=(0, 10.0), cumulative=True,
                normed=1, histtype='step', label='Matching (%d)' % np.sum(idx_match))
        plt.hist(distances[idx_nonmatch], bins=100, range=(0, 10.0), cumulative=True,
                normed=1, histtype='step', label='Non-Matching (%d)' % np.sum(idx_nonmatch))
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('Distance')

        plt.subplot(3, 1, 3)
        plt.title('Metric Distance CDF (Foreground only)')
        idx_match = (dataset.match == 1) & (dataset.filter == 1)
        idx_nonmatch = (dataset.match == 0) & (dataset.filter == 1)

        plt.hist(distances[idx_match], bins=100, range=(0, 10.0), cumulative=True,
                normed=1, histtype='step', label='Matching (%d)' % np.sum(idx_match))
        plt.hist(distances[idx_nonmatch], bins=100, range=(0, 10.0), cumulative=True,
                normed=1, histtype='step', label='Non-Matching (%d)' % np.sum(idx_nonmatch))
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('Distance')

        if save:
            plt.savefig(save)

    def make_filter_plot(self, dataset, save=None, show=False):
        probs = self.eval_filter_prob(dataset.occ)

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
        fig = plt.gcf()
        fig.set_size_inches(10, 8)

        self.make_pr_curve(is_background[valid], prob_background[valid], 'Background')
        self.make_pr_curve(is_foreground[valid], prob_foreground[valid], 'Foreground')

        plt.grid()

        if save:
            plt.savefig(save)

    def make_match_pr(self, distances, match, prob_fg, is_foreground):
        plt.title('PR by Filter Prob')
        for cutoff in [0.2, 0.4, 0.6, 0.8]:
            idx = prob_fg > cutoff

            if np.sum(idx) == 0:
                continue

            dist_c = distances[idx]
            match_c = match[idx]
            n_samples = np.sum(idx)

            self.make_pr_curve(match_c, -dist_c, 'P_fg > %3.1f (%d samples)' % (cutoff, n_samples))

        n_samples = np.sum(is_foreground)
        self.make_pr_curve(match[is_foreground], -distances[is_foreground], 'All Foreground (%d samples)' % (n_samples))

        n_samples = len(match)
        self.make_pr_curve(match, -distances, 'All (%d samples)' % (n_samples))

        plt.grid()

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

    filter_dm = FilterDataManager()
    flow_dm = FlowDataManager()

    filter_dm.make_validation(50)
    flow_dm.make_validation(10000)

    ml = MetricLearning(filter_dm, flow_dm)

    if len(sys.argv) > 1:
        load_iter = int(sys.argv[1])
        print 'Loading from iteration %d' % (load_iter)

        ml.restore('model.ckpt-%d' % (load_iter))
        ml.train(start_iter = load_iter+1)
    else:
        ml.train()
