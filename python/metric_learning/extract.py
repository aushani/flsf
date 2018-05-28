from tensorflow.python.tools import inspect_checkpoint as chkp
import sys
import tensorflow as tf
import numpy as np

if len(sys.argv) < 2:
    print 'Please specify tensorflow model file'
    sys.exit(1)

if len(sys.argv) < 3:
    print 'Please specify output directory'
    sys.exit(1)

model_file = sys.argv[1]
out_dir = sys.argv[2]

tmp_str = model_file[::-1]
idx = tmp_str.find('/')

if idx == -1:
    model_dir = './'
else:
    model_dir = tmp_str[:idx:-1]

print model_file
print model_dir

chkp.print_tensors_in_checkpoint_file(model_file, tensor_name='', all_tensors=False, all_tensor_names=True)

sess = tf.Session()

saver = tf.train.import_meta_graph(model_file + '.meta')
saver.restore(sess, tf.train.latest_checkpoint(model_dir))

graph = tf.get_default_graph()

var_names = [
'Encoder/l1/biases',
'Encoder/l1/weights',
'Encoder/l2/biases',
'Encoder/l2/weights',
'Encoder/l3/biases',
'Encoder/l3/weights',
'Encoder/latent/biases',
'Encoder/latent/weights',
'Filter/l1/biases',
'Filter/l1/weights',
]

dim_file = open('%s/dim.dat' % out_dir, 'w')

for var_name in var_names:
    print var_name
    var = graph.get_tensor_by_name(var_name + ":0")
    print var.shape
    var_val = var.eval(session=sess)

    if 'weights' in var_name:
        var_val = np.moveaxis(var_val, [0, 1, 2, 3], [1, 2, 3, 0])
        print 'Reshape to ', var_val.shape

    var_val = var_val.flatten()
    print var_val.shape

    fn = out_dir + '/' + var_name.replace('/', '_') + '.dat'
    np.savetxt(fn, var_val, delimiter=' ')

    shape = var.shape
    shape_str = ''
    for s in shape:
        shape_str += '%d ' % (s)
    shape_str += '\n'

    dim_file.write(shape_str)

dim_file.close()
