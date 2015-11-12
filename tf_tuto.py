import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# placeholder for the inputs
x = tf.placeholder("float", [None, 784])

# not inputs, but model parameters
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# the probability distribution (a way to normalize evidence = Wx+b, where
# evidence is how much an input belongs to a class, as one-hot vector)
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Training -----------------

# here we need an input placeholder for "labels" from training data,
# aka the "true" distribution
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


print dir(x)
for i in range(1000):
    if i % 100 == 0:
        print "step", i
    # not feeding the whole training data set, but rather small batches instead
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy,
               feed_dict={x: mnist.test.images, y_: mnist.test.labels})