import numpy as np
import tensorflow as tf
import urllib.request

# 1.データセットの取得
## Web上でホスティングされている学習データ（CSV形式）の取得
IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

## Web上でホスティングされているテストデータ（CSV形式）の取得
IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

## データを取得してCSVファイルとして書き出す
with open(IRIS_TRAINING, 'wb') as f:
    f.write(urllib.request.urlopen(IRIS_TRAINING_URL).read())

with open(IRIS_TEST, 'wb') as f:
    f.write(urllib.request.urlopen(IRIS_TEST_URL).read())

## データを使いやすい形に整形
training_data = np.loadtxt('./iris_training.csv', delimiter=',', skiprows=1)
train_x = training_data[:, :-1]
train_y = training_data[:, -1]

test_data = np.loadtxt('./iris_test.csv', delimiter=',', skiprows=1)
test_x = test_data[:, :-1]
test_y = test_data[:, -1]

## データの中から、指定されたバッチサイズの分だけランダムに取得
def next_batch(data, label, batch_size):
    indices = np.random.randint(data.shape[0], size=batch_size)
    return data[indices], label[indices]

# 2.ニューラルネットワークの構築
## 入力層
x = tf.placeholder(tf.float32, [None, 4], name='input')

## 中間層1
hidden1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu, name='hidden1')
## 中間層2
hidden2 = tf.layers.dense(inputs=hidden1, units=20, activation=tf.nn.relu, name='hidden2')
## 中間層3
hidden3 = tf.layers.dense(inputs=hidden2, units=10, activation=tf.nn.relu)
## 出力
y = tf.layers.dense(inputs=hidden3, units=3, activation=tf.nn.softmax, name='output')

