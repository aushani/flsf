import tensorflow as tf
from data import *
import time
import sample

plt.switch_backend('agg')

class Icra2017Learning:

    def __init__(self, data_manager):

        self.dm = data_manager

        self.width = self.dm.width
        self.length = self.dm.length
        self.height = self.dm.height
        self.dim_data = self.width * self.length * self.height

        self.n_classes = sample.n_classes

        self.latent_dim = 25

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

    def train(self):
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

        # Set up writer
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

        # Save checkpoints
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        valid_set = self.dm.validation_set

        fd_valid = {
                        self.fv : valid_set.feature_vector,
                        self.match : valid_set.match,
                   }

        iteration = 0

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
                save_path = saver.save(self.sess, "./model.ckpt", global_step = iteration)
                toc = time.time()

                #print '\tTook %5.3f sec to save checkpoint' % (toc - tic)

            if iteration % it_plot == 0:
                print 'Iteration %d' % (iteration)

                self.make_match_plot(valid_set, save='metric_%010d.png' % iteration)

                print '  Match accuracy = %5.3f %%' % (100.0 * self.accuracy.eval(session = self.sess, feed_dict = fd_valid))

                if iteration > 0:
                    print '  Loading data at %5.3f ms / iteration' % (t_data*1000.0/iteration)
                    print '  Training at %5.3f ms / iteration' % (t_train*1000.0/iteration)

                print ''

            tic = time.time()
            samples = self.dm.get_next_samples(1)
            toc = time.time()

            t_data += (toc - tic)

            tic = time.time()
            fd = {
                   self.fv: samples.feature_vector,
                   self.match: samples.match,
                 }
            self.train_step.run(session = self.sess, feed_dict = fd)
            toc = time.time()

            t_train += (toc - tic)

            iteration += 1

    def make_match_plot(self, dataset, save=None, show=False):
        scores = self.score.eval(session = self.sess, feed_dict = {self.fv: dataset.feature_vector})

        scores = np.squeeze(scores)

        plt.clf()

        self.make_pr_curve(dataset.match, scores, 'ICRA 2017')

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

    ml = Icra2017Learning(dm)
    ml.train()
