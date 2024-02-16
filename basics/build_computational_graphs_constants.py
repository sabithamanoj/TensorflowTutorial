#######################################################
#       Tensorflow tutorial
######################################################
# Import packages
import logging
import os
import tensorflow as tf


def main():

    # Log file
    # Check if log directory exist, if not create one
    dir_name = "log"
    isExist = os.path.exists(dir_name)
    if not isExist:
        # Create a log directory
        os.makedirs(dir_name)
        print('log directory is created!!')

    logging.basicConfig(filename='log/build_computational_graph_constants.log', level=logging.INFO, filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('Tensorflow version: {}'.format(tf.__version__))

    # Build a data flow graph
    const1 = tf.constant(5)
    const2 = tf.constant(1)
    c = const1 + const2
    logging.info('Addition operation')
    logging.info('Input1 : {}, Input2 : {}'.format(const1, const2))
    logging.info('result - (Input1 + Input2) : {}'.format(c))


if __name__ == '__main__':
    main()
