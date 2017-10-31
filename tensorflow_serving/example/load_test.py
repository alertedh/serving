import tensorflow as tf

with tf.Session(graph=tf.Graph()) as sess:
    abc = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING], '/tmp/mnist_model/1')
    ergn = tf.get_collection("ASSETS")