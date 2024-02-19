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
    else:
        print('log directory already exists!!')

    logging.basicConfig(filename='log/single_perceptron.log', level=logging.INFO, filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info('Tensorflow version: {}'.format(tf.__version__))

    # Disable eager execution
    #tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()

    w = tf.Variable([.2]) # Initial weight
    b = tf.Variable([-.2]) # Initial bias
    x = tf.compat.v1.placeholder(tf.float32) # Input
    logging.info('Single Perceptron')
    logging.info('weight : {}, bias : {}'.format(w, b))
    linear_model = w*x + b


    # Initialize variables
    init = tf.compat.v1.global_variables_initializer()
    ssn = tf.compat.v1.Session()
    ssn.run(init)
    ssn.run(w)
    ssn.run(b)
    logging.info('Predicted output is:')
    logging.info(ssn.run(linear_model, {x:[1, 2, 3, 4]}))



if __name__ == '__main__':
    main()
