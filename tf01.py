import numpy as np

# def vectorize_sequences(sequences, dimension=10000):
# results = np.zeros((len(sequences), dimension))
# for i, sequence in enumerate(sequences):
# results[i, sequence] = 1.
# return results
# x_train = vectorize_sequences(train_data)
# x_test = vectorize_sequences(test_data)

# import tensorflow as tf
# from keras.datasets import imdb

# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(
#     path="imdb.npz",
#     num_words=None,
#     skip_top=0,
#     maxlen=None,
#     seed=113,
#     start_char=1,
#     oov_char=2,
#     index_from=3,
# )


# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# save np.load
# np_load_old = np.load

# modify the default parameters of np.load
# np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true


# restore np.load for future normal usage
# np.load = np_load_old

# from keras.datasets import reuters

# (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
#     num_words=10000
# )

# len(train_data)

# from keras import models
# from keras import layers

# model = models.Sequential()
# model.add(layers.Dense(32, activation="relu", input_shape=(784,)))
# model.add(layers.Dense(10, activation="softmax"))

# input_tensor = layers.Input(shape=(784,))
# x = layers.Dense(32, activation="relu")(input_tensor)
# output_tensor = layers.Dense(10, activation="softmax")(x)
# model = models.Model(inputs=input_tensor, outputs=output_tensor)
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss="mse", metrics=["accuracy"])
# model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)

# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

# tf.set_random_seed(1)  # set random seed

# # 导入数据
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# # hyperparameters
# lr = 0.001  # learning rate
# training_iters = 100000  # train step 上限
# batch_size = 128
# n_inputs = 28  # MNIST data input (img shape: 28*28)
# n_steps = 28  # time steps
# n_hidden_units = 128  # neurons in hidden layer
# n_classes = 10  # MNIST classes (0-9 digits)

# # x y placeholder
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# y = tf.placeholder(tf.float32, [None, n_classes])

# # 对 weights biases 初始值的定义
# weights = {
#     # shape (28, 128)
#     "in": tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
#     # shape (128, 10)
#     "out": tf.Variable(tf.random_normal([n_hidden_units, n_classes])),
# }
# biases = {
#     # shape (128, )
#     "in": tf.Variable(tf.constant(0.1, shape=[n_hidden_units])),
#     # shape (10, )
#     "out": tf.Variable(tf.constant(0.1, shape=[n_classes])),
# }


# def RNN(X, weights, biases):
#     # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
#     # X ==> (128 batches * 28 steps, 28 inputs)
#     X = tf.reshape(X, [-1, n_inputs])

#     # X_in = W*X + b
#     X_in = tf.matmul(X, weights["in"]) + biases["in"]
#     # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
#     X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

#     # 使用 basic LSTM Cell.
#     lstm_cell = tf.contrib.rnn.BasicLSTMCell(
#         n_hidden_units, forget_bias=1.0, state_is_tuple=True
#     )
#     init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state

#     outputs, final_state = tf.nn.dynamic_rnn(
#         lstm_cell, X_in, initial_state=init_state, time_major=False
#     )
