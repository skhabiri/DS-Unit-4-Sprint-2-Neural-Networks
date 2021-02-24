## Hyperparameter Tuning:
Some of the important hyperparameters in neural networks to tune:
batch_size, training epochs, optimization algorithms, learning rate, momentum, activation functions, dropout regularization, number of neurons in the hidden layer, number of the layers.

### Batch Size:
Batch size determines how many observations the model is shown before it calculates loss/error and updates the model weights via gradient descent. You're showing the model enough observations that you have enough information to update the weights, but not such a large batch size that you don't get a lot of weight update iterations performed in a given epoch. Feed-forward Neural Networks aren't as sensitive to bach_size as other networks. Smaller batch sizes will also take longer to train. Keras defaults to batch size of 32. Increasing the minibatch size could lower the effective learning rate that provides stable convergence.

### Learning rate:
For a given number of epochs, a small learning rate may not reach the optimum point and under fit. A very large learning rate can cause divergence behavior. 

### Momentum:
Momentum is a property that decides the willingness of an optimizer to overshoot the minimum. Imagine a ball rolling down one side of a bowl and then up the opposite side a little bit before settling back to the bottom. The purpose of momentum is to try and escape local minima.

### Activation Function:
This is another hyperparameter to tune. Typically you'd want to use ReLU for hidden layers and either Sigmoid, or Softmax for output layers of binary and multi-class classification implementations respectively.

### Network Weight Initialization:
 Your model will get further with less epochs if you initialize it with weights that are well suited to the problem you're trying to solve. ```init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']```

### Dropout Regularization and the Weight Constraint:
The Dropout Regularization value is a percentage of neurons that you want to be randomly deactivated during training. The weight constraint is a second regularization parameter that works in tandem with dropout regularization. You should tune these two values at the same time. Using dropout on visible vs hidden layers might have a different effect. 

### Number of units (neurons) per layer and number of layers:
Typically depth (more layers) is more important than width (more nodes) for neural networks. The more nodes and layers the longer it will take to train a network, and higher the probability of overfitting. The larger your network gets the more you'll need dropout regularization or other regularization techniques to keep it in check.


## Search Strategies:
**GridSearch:** This has a specific downside in that if I specify 5 hyperparameters with 5 options each then I've just created 5^5 combinations of hyperparameters to check. If I use 5-fold Cross Validation on that then my model has to run 15,525 times. When using Grid Search don't use it to test combinations of different hyperparameters, only use it to test different specifications of a single hyperparameter. It's rare that combinations between different hyperparameters lead to big performance gains. Then retain the best result for that single parameter while you test another, until you tune all the parameters in that way.

**RandomSearch:** Grid Search treats every parameter as if it was equally important, but this just isn't the case. Random Search allows searching to be specified along the most important parameter and experiments less along the dimensions of less important hyperparameters. The downside of Random search is that it won't find the absolute best hyperparameters, but it is much less costly to perform than Grid Search.

**Bayesian Optimization:** Bayesian Optimization is a search strategy that tries to take into account the results of past searches in order to improve future ones. That is tuning our hyperparameter tuning. `keras-tuner` offers Bayesian methods implementation.


### Dataset: Load mnist digits with X_train.shape=(60000, 28, 28), X_test.shape=((10000, 28, 28). Normalizing the data makes training faster and reduces the chances that gradient descent gets stuck in a local minimum. Next we reshape the input to a 784 array.

## Hyperparameter techniques in Neural Networks:
### HP Tuning with GridSearchCV through Keras sklearn wrapper:
In order to utilize the GridSearchCV, we use sklearn wrapper for keras, KerasClassifier. GridSearchCV will handle the parameter grid and cross validation folding and the KerasClassifier will train the NN for each parameter set and run for the specified number of epochs. For each parameter set Pj and input fold of Xi, keras will train a model. The parameter set which yields the maximum average score over all the folds, Pjmax will be selected to train a keras model with the entire input data X.
```
model = KerasClassifier(modelbuild_fn)
param_grid = {'batch_size': [32,64,512], 'epochs': [20]}
grid = GridSearchCV(estimator=model, param_grid, cv=5, refit=True)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)

# best trained keras model
best_NN = grid.best_estimator_.model
```

### Hyperparameter Tuning with the HParams Dashboard in TensorBoard:
HParams works with TensorBoard, which provides an **Experiment Tracking Framework** to manage the tuning work including the parameter set data, date, and metric results.

**Define hyperparameters and score metrics:**
```
from tensorboard.plugins.hparams import api as hp
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16,32]))
METRIC_ACCURACY = 'hpaccuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
hp.hparams_config(hparams=[HP_NUM_UNITS,..], metrics=[hp.Metric(METRIC_ACCURACY,      display_name='HPaccuracy'),..]
```

**Adapt model function with HParams:**
```
def train_test_model(hparams):
model = tf.keras.Sequential(
[tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation='relu')])

model.compile(
optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'mse'])

model.fit(X_train, y_train, epochs, batch_size)
_, accuracy = model.evaluate(X_test, y_test)

return accuracy
```

HParam can work with tensor.summary file which is accessible by tensorboard. We use HParam to record parameter sets in the file. For each run, log an hparams summary with the hyperparameters and final accuracy.
```
def run(run_dir, hparams):
with tf.summary.create_file_writer(run_dir).as_default():
hp.hparams(hparams)  # record the parameter values used in this trial
       
 _, accuracy= train_test_model(hparams)
tf.summary.scalar(name=METRIC_ACCURACY, data=accuracy, step=1)
```
Now create a dictionary of `hparams` and run each parameter set.
```
session_num = 0
for num_units in HP_NUM_UNITS.domain.values:
hparams = {HP_NUM_UNITS: num_units,}

run_name = f'run-{session_num}'
run('logs/hparam_tuning/' + run_name, hparams)

session_num += 1
```
The dumped file can be opened by TensorBoard: `%tensorboard  --logdir "logs/hparam_tuning"`.

### Hyperparameter Tuning in keras-tuner library:
RandomSearch():
In RandomSearch() an instance of HyperParameters() class is passed to the hypermodel parameter as the argument. An instance of `HyperParameters` class contains information about both the search space and the current values of each hyperparameter. Hyperparameters can be defined inline with the model-building code that uses them.
```
import kerastuner as kt
import tensorflow as tf
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = tf.keras.Sequential()
    for i in range(hp.Int('layers', 3, 10)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int('units_' + str(i), 50, 100, step=10),
            activation=hp.Choice('act_' + str(i), ['relu', 'tanh'])))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

# hp = kt.HyperParameters()
# model = build_model(hp)
tuner = RandomSearch(
hypermodel=build_model, objective='val_accuracy', max_trials=5, executions_per_trial=3, directory='./keras-tuner-trial')

print(tuner.search_space_summary())

tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

tuner.results_summary()
```
max_trials: represents the number of hyperparameter combinations that will be tested by the tuner.
execution_per_trial: is the number of models that should be built and fit for each trial for robustness purposes. Those are different from epoch.
for best model:
```
best_model = tuner.get_best_models()[0]
# Evaluate the best model.
loss0, accuracy0 = best_model.evaluate(X_test, y_test)
tuner.get_best_hyperparameters(num_trials=1)[0].values
```
Hyperband(): Hyperband is an optimized version of random search which uses early-stopping to speed up the hyperparameter tuning process. The main idea is to fit a large number of models for a small number of epochs and to only continue training for the models achieving the highest accuracy on the validation set. The max_epochs variable is the max number of epochs that a model can be trained for. While Hyperbrand runs faster, RandomSearch tuner does a better job in finding the optimum hyper parameters.
```
tuner_hb = kt.Hyperband(build_model,
                     objective = 'val_accuracy', 
                     max_epochs = 5,
                     factor = 3,
                     directory = './kt-hyperband',
                     project_name = 'kt-HB')  
```

Libraries:
```
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner.tuners as kt
from kerastuner.tuners import RandomSearch, Hyperband
import os
import datetime
```
