*******************************************************
*                                                     *
*          Feature Learning for Scene Flow            *
*                                                     *
*******************************************************

-------------------------------
|           Setup             |
-------------------------------

0) This project depends on cuda and cudnn and assumes you have an NVIDIA GPU.

1) Run bootstrap.sh. This will install other dependencies and get some KITTI data.

2) Build the C++ code from flsf/cc using a command such as:

  $ bazel build //... -c opt

-------------------------------
|    Generate Training Data   |
-------------------------------

Note: You can generate your own training data and train the network from
scratch. For ease of use, we provide data for a pretrained network. If you wish
to use this pretrained network, you may skip ahead.

1) Run scripts/gen_training_data.sh. You should open the script and change
parameters as necessary, such as the path to which to store data.

-------------------------------
|    Training the Network     |
-------------------------------

Note: You can generate your own training data and train the network from
scratch. For ease of use, we provide data for a pretrained network. If you wish
to use this pretrained network, you may skip ahead.

1) Run python/feature_learning/feature_learning.py. You will have to pass in the
path to the training data from above. For example,

  $ python feature_learning.py /my/training/data/

2) The training program will periodically output intermediate results and also
tensorflow model files for the current state of the network.

-------------------------------
|   Extracting the network    |
-------------------------------

Note: You can generate your own training data and train the network from
scratch. For ease of use, we provide data for a pretrained network. If you wish
to use this pretrained network, you may skip ahead.

1) Run python/feature_learning/extract.py to extract the network from the
tensorflow model file at the desired iteration. For example,

  $ python extract.py path/to/model.ckpt-0000000 /flsf/data/network

-------------------------------
|        Visualizing          |
-------------------------------

Note: If you skipped training the network from scratch, you may start from here.

1) Run cc/bazel-bin/app/flow/flow. This will launch a viewer that will allow you
to visualize results. Some sample commands for the viewer:

  1-6 for visualizing different data
  Ctrl+Click to view the distance in feature space to neighboring locations
  C to clear the distances
  N to advance to the next scan

Note that the runtime here is increased due to the time it takes to copy
visualization data from the GPU.

-------------------------------
|        Evaluation           |
-------------------------------

1) Run scripts/gen_eval_data.sh. You should open the script and change
parameters as necessary, such as the path to which to store data.

2) Run python/eval/eval.py. This will process the generated evaluation files and
output error statistics and runtimes. For example,

  $ python eval.py /path/to/eval/files

-------------------------------
|     Acknowledgements        |
-------------------------------

Based off of https://github.com/aushani/tsf
