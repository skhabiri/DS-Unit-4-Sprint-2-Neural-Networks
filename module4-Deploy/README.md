# Regularization and deployment
Common ways of regularization in neural networks:

* **EarlyStopping:** This strategy will prevent your weights from being updated well past the point of their peak usefulness.
* Use EarlyStopping, **Weight Decay** and **Dropout**
* Use EarlyStopping, **Weight Constraint** and Dropout

Weight Decay and Weigh Constraint accomplish similar purposes .
### dataset: 
mnist digits with shape (60000, 28, 28). we use flatten layer to make the array one dimension.

### Flatten Layer:
Whenever you need to convert a multidimensional tensor into a single 1-D tensor, you can use Flatten layer. One use case is connecting the output of a convolutional+pooling layer to a dense layer. We insert the flatten before the dense layer. For instance if the output of the previous layer is of shape (15, 3, 3, 4), flatten unstacks all the tensor values into a 1-D tensor of shape (1533*4,) so that it can be used as input for a Dense layer.
```
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3)
model = tf.keras.Sequential([Flatten(input_shape=(28,28)), Dense(128), ReLU(negative_slope=.01), Dense(128), ReLU(negative_slope=.01), Dense(128), ReLU(negative_slope=.01), Dense(10, activation='softmax')])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=99, validation_data=(X_test,y_test), callbacks=[tensorboard_callback, stop])
```
### Ridge Regression L2 vs Lasso Regression L1:
* Ridge L2:
OLS provides what is called the Best Linear Unbiased Estimator. That means that if you take any other unbiased estimator, it is bound to have a higher variance then the OLS solution. when you estimate your prediction error, it is a combination of three things: 

E[(𝑦−𝑓̂ (𝑥))2]=Bias[𝑓̂ (𝑥))]2+Var[𝑓̂ (𝑥))]+𝜎2

The last part is the irreducible error, so we have no control over that. Using the OLS solution the bias term is zero. But it might be that the second term is large. It might be a good idea, (if we want good predictions), to add in some bias and hopefully reduce the variance.

In Ridge, as a loss function we have sum of the squared errors (Ordinary Least Square) plus lambda times some of squared of model paramters except the y intercept (bias). 
<center><img src="https://github.com/skhabiri/ML-ANN/raw/main/module4-Deploy/image/l2formula.png" width=200/><center>

Using *cross-validation* and *ridge* with a smaller number of training samples we can still determine the coefficients. The higher value of $\lambda$ lowers the slope (coefficients) which makes the fit to be more represented by y intercept. But it lowers the MSE for the unseen data as the model coefficients (slope) is less which means the model would not make a drastic change to fit the noisy data (overfitting). The MSE between the estimated model and training data represent the **bias**. The MSE between the estimated model and validation data represents the **variance**. Ridge regression adds to the model bias and in return lowers the variance.

### Lasso L1:
In Lasso loss is: <center><img src="https://github.com/skhabiri/ML-ANN/raw/main/module4-Deploy/image/l1formula.png" width=200/><center>
Similar to ridge regression different coefficients may reduce ununiformely. However unlike ridge where a coefficient might reduce to zero for 𝜆→∞, in Lasso a coefficient can reduce to **exactly** zero for a limited value of 𝜆. This is a useful property where our data has some irrelevant features that we want to eliminate them from the model.
```
from tensorflow.keras import regularizers
model = tf.keras.Sequential([Flatten(input_shape=(28,28)), Dense(512, kernel_regularizer=regularizers.l2(l2=0.01)), ReLU(negative_slope=.01), Dense(512, kernel_regularizer=regularizers.l2(l2=0.01)), ReLU(negative_slope=.01), Dense(512, kernel_regularizer=regularizers.l2(l2=0.01)), ReLU(negative_slope=.01), Dense(10, activation='softmax')])
```

### Weight constraint:
MaxNorm=m will, if the L2-Norm of your weights exceeds m, scale your whole weight matrix by a factor that reduces the norm to m. If you use a simple L2 regularization term you penalize high weights with your loss function. With this constraint, you regularize directly. This seems to work especially well in combination with a dropout layer.
```
from tensorflow.keras.constraints import MaxNorm
wc = MaxNorm(max_value=2)
model = tf.keras.Sequential([Flatten(input_shape=(28,28)), Dense(512, kernel_constraint=wc), ReLU(negative_slope=.01), Dense(512, kernel_constraint=wc), ReLU(negative_slope=.01), Dense(512, kernel_constraint=wc), ReLU(negative_slope=.01), Dense(10, activation='softmax')])
```
### Dropout:
Dropout deactivates a percentage of neurons of each layer randomly, each epoch the neurons will be randomly selected and deactivated, therefore the Forwardpropagation not Backward propagation will not use these neurons to train the model.If we are going to use dropout we should know two tricks:
Since dropout deactivates a percentage of neurons we could need to add more layers to avoid Underfitting.
We can use a bigger value for the learning rate hyperparameter.
```
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
wc = MaxNorm(max_value=2)
logdir = os.path.join("logs", "EarlyStopping+WeightConstraint+Dropout")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3)
model = tf.keras.Sequential([Flatten(input_shape=(28,28)), Dense(256, kernel_constraint=wc), ReLU(negative_slope=.01), Dropout(.2), Dense(256, kernel_constraint=wc), ReLU(negative_slope=.01), Dropout(.2), Dense(256, kernel_constraint=wc), ReLU(negative_slope=.01), Dropout(.2), Dense(10, activation='softmax')])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=99, validation_data=(X_test,y_test), callbacks=[tensorboard_callback, stop])
```
### Deployment:
**ModelCheckpoint callback:** is used in conjunction with training using model.fit() to save a model or weights (in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved.
A few options this callback provides include:
Whether to only keep the model that has achieved the "best performance" so far, or whether to save the model at the end of every epoch regardless of performance.
Definition of 'best'; which quantity to monitor and whether it should be maximized or minimized.
The frequency it should save at. Currently, the callback supports saving at the end of every epoch, or after a fixed number of training batches.
Whether only weights are saved, or the whole model is saved.
```
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="weights_best.h5", monitor='val_loss', verbose=1, save_best_only=False, save_freq='epoch', save_weights_only=True)
model = create_model()
model.fit(X_train, y_train, epochs=5, validation_data=(X_test,y_test), verbose=2, callbacks=[checkpoint_callback])
m = create_model()
m.load_weights('./weights_best.h5')  # Load instead of train
m.evaluate(X_test, y_test, verbose=1)
```
For saving the entire model:
```
model = create_model()
model.fit(X_train, y_train, epochs=5)
model.save('saved_model/my_model_dir') 
loaded_model = tf.keras.models.load_model('saved_model/my_model')
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
loaded_model.evaluate(X_test, y_test)
```
Floating point numbers are stored at a certain precision in number of bits - e.g. 16, 32, etc. Above that precision, numbers are rounded off. When you train, you want full precision! Training is sensitive... But at the very end, it may be okay to reduce the precision as another way to reduce model size. Let's say to go from e.g. 32 bit numbers to 16 or 8 bit numbers. Basically truncate/round numbers.
### why loss from .fit on the last epoch for train data is different from the loss from evaluate() method but those loss numbers are the same for eval data?
.fit(): During the train process of even the last epoch at the end of each of its batch, we update the weights. Even at the end of the last batch of the last epoch we update the weight one last time. That is for training data. The validation of the last epoch is fed in after the last epoch model is finalized.
.evaluate(): The model is already finalized and the validation data is fed to the same model as the one after the last batch of the last epoch. So validation results are the same. That wouldn't be the case for the training data as the model was evolving in the .fit() phase of the last epoch as the training data was fed to.
For training data in addition to back propagation update at the end of each batch, if we have random dropout, that would also come into effect during the fit(). However during evaluate() we do not have any random drop out for neurons.
### Libraries:
```
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import ReLU
import tensorflow as tf
import os
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
