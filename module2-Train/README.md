### Backpropagation & Gradient Descent: 
It refers to the process of after each batch of input the gradient of loss function (some difference of target and estimate) vs each weight is calculated and the weights are updated based on a learning rate. Steps are summarized here:


### feed forward linear perceptron:
 The perceptron is implemented from scratch and trained. The target is derived from `outputs = inputs * TRUE_W + TRUE_b + noise`. Noise and inputs are drawn from a random numpy array in the form of tensor:
```
inputs = tf.random.normal(shape=[1000], mean=0, stddev=1, dtype=tf.float32, seed=42)
```
The loss function is calculated as MSE tensor: `tf.reduce_mean(tf.square(target_y - predicted_y))`. The tensor data type is built on numpy and similar to that expands the dimensions when doing mathematical operations. The model class is defined as:
```
class Model(object):

  def __init__(self):
    self.W = tf.Variable(8.0)
    self.b = tf.Variable(40.0)

  # So, the __init__ method is used when the class is called to initialize the instance, 
  # while the __call__ method allows the class's instance to be called as a function, and returns a value
  def __call__(self, x):
    return self.W * x + self.b

model = Model()
```
To train the model we update the weight and bias with a learning rate based on the gradient. 
GradientTape(): Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched". Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. Tensors can be manually watched by invoking the watch method on this context manager.
```
 def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t: 
        # recording :
        current_loss = loss(outputs, model(inputs))

    #takes the gradient of current_loss vs W and b, dw is actually d(Loss)/dW
    dW, db = t.gradient(current_loss, [model.W, model.b])
    # for dW>0 lowering W with lower Loss (Learning_rate>0): W = W - learning_rate * dW
    # assign_sub(ref, value) >> ref = ref-value (assign subtraction)
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

    return current_loss
```
driver code for training:
```
model = Model()
epochs = range(10)
for epoch in epochs:
    train(model, inputs, outputs, learning_rate=0.1)
```
### Normalizing data:
Whenever all data is normalized to values within 0 and 1, that ensures that the update to all the weights are updated in equal proportions which can lead to quicker convergence on the optimal weight values. If your dataset's values range across multiple orders of magnitude (i.e.  101,  102,  103,  104 ), then gradient descent will update the weights in grossly uneven proportions.

### Batch Gradient Descent:
Strictly speaking, "Minibatch" Gradient Descent means that instead of passing all of our data through the network for a given epoch (Batch GD), we just pass a randomized portion of our data through the network for each epoch.
Stochastic Gradient Descent is when we make updates to our weights after forward propagating each individual training observation.
Batch GD refers to a random draw of a batch size from the training set and updating the weights and bias and continuing that until the entire training set is consumed and that would account as one epoch. 


### Batch Size:
Batches are the number of observations our model is shown to make predictions and update the weights. Batches are selected randomly during epoch. All observations are considered when passing through an epoch at some point. 
Smaller Batch = Slower Run Time (but maybe more accurate results)
Default Batch (32) = Balance between speed and accuracy
Large Batch = Very fast, but not nearly as accurate.

### mnist dataset:
Next we use tensorflow.keras.datasets.mnist is used to train a NN with 784 inputs representing a handwritten digit from 0 to 9. The training set is 60K and validation data is 10K. We try 32 hidden dense layer connected to the input tensor, another 32 hidden dense layer and 10 output dense layer.
```
    model = Sequential(
        [
        Dense(32, activation='relu', input_dim=784),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')       
        ]
    )
    model.compile(optimizer=SGD, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
After building the model by defining the layers, we compile the model by defining the loss function, optimizer and the metrics. Next we need to fit the model.
```
result = model.fit(X_train, y_train, epochs, batch_size, validation_data=(X_test, y_test))
```
With epoch=25, batch_size=32, Optimizer=SGD, we achieve an accuracy of 0.95. For this dataset batch_=4096 is as accurate as batch_size=8. For a given epoch, in general a smaller batch size tends to be more accurate as there are more back-propagation corrections and at the same time more time consuming. However a larger batch size tends to average out the noise in input samples and the validation curves tend to be smoother. In regards to learning rate, bigger numbers take large steps in large variance and could cause convergence issues. However, it could help with passing over local minimums. Small learning rates need a large number of epochs to train as the correction steps are small.

The second notebook, ann_train-422a.ipynb, uses the same topology but for quickdraw dataset. X shape is (100K, 784) and X_test is 20% of the data. With SGD optimizer, 0.01 learning rate and batch_size=512 we get an accuracy of 0.84.

### Libraries:
```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import *
from tensorflow.keras import initializers

from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import wget
```

