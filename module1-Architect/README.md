### Architecture of a simple Perceptron: 
It consists of input tensor, weights and bias and output neuron. Neurons apply an activation function to the weighted sum of incoming tensors.

### Optimizing the loss function
gradient descent: 
First initialize the weights to some random number. For a given input sample of xi, the calculated output is y^. A loss function can conceptually be defined as ½*(y-y^)**2, Or for a batch of inputs is the summation of individual errors. By calculating the derivative of loss in respect to each weight we find the direction of a small change in the weight where it leads to reduction in loss amount. The derivative is a function of input sample and activation function.
Stochastic gradient descent: The quadratic nature of the loss function helps to have a convex curvature and avoid local minima. Additionally feeding different input samples moves the operating point to different regions and helps to avoid getting trapped in a local minimum.
Batch gradient descent: Taking a batch of input and calculating the combined loss and moving weights in a direction to lower the combined loss helps to regularize our training and avoid fitting to the noise in the input.

### Tensorflow data:
Rank: number of dimensions in a tensor
Shape: number of elements within each dimension
 rank-1 Tensor is a vector; it has magnitude and direction; ex. shape = (3,)
A rank-2 Tensor is a matrix; it is a table of numbers; ex. shape = (3, 6)

### Keras input/output, input_shape vs input_dim:
What flows between layers are tensors. Tensors can be seen as matrices, with shapes. In Keras, the input layer itself is not a layer, but a tensor. It's the starting tensor you send to the first hidden layer. This tensor must have the same shape as your training data. Example: if you have 30 images of 50x50 pixels in RGB (3 channels), the shape of your input data is (30,50,50,3). Since the keras model needs to work with any batch size, we define the input layer as input_shape = (50,50,3). If a model strictly requires to have the batch size, We use batch_input_shape=(30,50,50,3). Either way keras will have the batch dimension, when reporting the model summary. In the first case would be like (None,50,50,3). The first dimension is the batch size, it's None because it can vary depending on how many examples you give for training. 
The "units" of each layer will define the output shape (the shape of the tensor that is produced by the layer and that will be the input of the next layer). Each type of layer works in a particular way. Dense layers have output shape based on "units", convolutional layers have output shape based on "filters".
input_shape is a tuple. If your input shape has only one dimension, you don't need to give it as a tuple, you give input_dim as a scalar number. So, in your model, where your input layer has 3 elements, you can use any of these two:
```
input_shape=(3,) -- The comma is necessary when you have only one dimension
input_dim = 3
```
Each type of layer requires the input with a certain number of dimensions:
Dense layers require inputs as: `(batch_size, input_size)`
2D convolutional layers need inputs as:
if using channels_last: `(batch_size, imageside1, imageside2, channels)`
if using channels_first: `(batch_size, channels, imageside1, imageside2)`
1D convolutions and recurrent layers use: `(batch_size, sequence_length, features)`
ex/ A training set for mnist image has a shape of x_train.shape = (60000, 28, 28). 60K is the number of samples and 28,28 are x, y dimensions and the channel is missing. So we assume it's grey scale with channel=1. with images, we would often use Convolutional Neural Networks. In those models, we use Conv layers, which expect the input_shape as follows: (x_shape, y_shape, channels). By consequence, our value for input_shape will be (28, 28, 1).

### Define a NN Model:
Keras has two ways of doing it, Sequential models, or the functional API Model. With the Sequential model:
```
from keras.models import Sequential  
from keras.layers import *  

model = Sequential()    

#start from the first hidden layer, since the input is not actually a layer   
#but inform the shape of the input, with 3 elements.    
model.add(Dense(units=4,input_shape=(3,))) #hidden layer 1 with input

#further layers:    
model.add(Dense(units=4)) #hidden layer 2
model.add(Dense(units=1)) #output layer
```
With the functional API Model:
```
from keras.models import Model   
from keras.layers import * 

#Start defining the input tensor:
inpTensor = Input((3,))   

#create the layers and pass them the input tensor to get the output tensor:    
hidden1Out = Dense(units=4)(inpTensor)    
hidden2Out = Dense(units=4)(hidden1Out)    
finalOut = Dense(units=1)(hidden2Out)   

#define the model's start and end points    
model = Model(inpTensor,finalOut)
```
Shapes of the tensors:
Remember you ignore batch sizes when defining layers: `inpTensor: (None,3) hidden1Out: (None,4) hidden2Out: (None,4) finalOut: (None,1)`

## NN_Architect-421.ipynb:
### Load dataset:
dataset: `from tensorflow.keras.datasets import mnist` handwritten digits with 60k samples of 28x28 ((60000, 28, 28), numpy.ndarray, dtype('uint8')) and 10K of test. We normalize the data by dividing them to the maximum, 255. We also flatten the input data to 784 features, `X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))`, (60000, 784).
### Define the model:
Input layer is not really a layer but a tensor with no activation function, So we start the actual layer definition with the dense layer and since the input dimension is already 1 (input_shape=(784,) we can instead of input_shape pass the input_dim parameter with an integer value of 784 instead of tuple for input_shape. softmax is used for classification of multiclasses as the output values are probabilities that sums up to 1. (None, ...) in the output shape refers to batch size. It is not defined and hence could work with any batch size Default batch size during training phase is 32 samples at a time.
```
model.add(
        # Hidden Layer
        Dense(units=32, # our hidden layer has 32 neurons in it!
            activation="relu", 
            # the only outside parameter that needs to be defined
            input_dim=784,
            kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42),
            bias_initializer=initializers.Ones())
        )
model.add(
    # output layer, specify the number of labels to predict, here that's 10
    Dense(10, activation='softmax')
)
```
### Compile the model:
**epoch:** is one iteration over the entire input dataset, 60K. Batches are parts of epoch.
**Loss function:** Use categorical_crossentropy (cce) loss function when there are two or more label classes. We expect training labels to be provided in a one_hot representation. If you want to provide labels as integers, use sparse_categorical_crossentropy (scce) loss.
```
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

```
**Tensorboard:** provides the visualization and tooling needed for machine learning experimentation. It does that by tracking and visualizing metrics such as loss and accuracy, visualizing the model graph and so on. For two dense layer of (32, and 10 units) with 5 epochs we get a validation accuracy of 0.95.

## NN_Architect-421a.ipynb:
### Load dataset:
It’s quickdraw dataset with 100K drawing for ten classes. Since the target label is known, it’s considered a supervised learning. X.shape= (100000, 784), y.shape=(100000,)

### Build the model:
Here is a function to build a 3 layer NN.
```
def create_model(lr=.01):
  """
  Stochastic Gradient Descent SGD optimizer
  784+1 * 32+1 * 32+1 * 10
  """
  opt = SGD(learning_rate=lr)

  model = Sequential(
      [
      #  784 inputs + 1 bias connect to 32 1st layer Hiddent neurons
       Dense(32, activation='relu', input_dim=784),
      #  32 1st-H-Neurons + 1 bias connected to 32 2'nd layer H-Neurons
       Dense(32, activation='relu'),
      #  32 2nd-H-neurons connect to 10 Output neurons
       Dense(10, activation='softmax')       
      ]
    )
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

model = create_model()
model.summary()
```
And for fitting the model:
```
baseline = model.fit(X, y,
                       validation_split=0.2,
                      #Hyperparameters!
                       epochs=20,
                       batch_size=32,
                       )
```
We get a validation accuracy of 10%. Not that great. Using adam optimizer increases the accuracy to about 0.2.
