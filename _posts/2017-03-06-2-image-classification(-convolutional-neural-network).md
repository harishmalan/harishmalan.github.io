
# Image Classification
In this project, you'll classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  The dataset consists of airplanes, dogs, cats, and other objects. You'll preprocess the images, then train a convolutional neural network on all the samples. The images need to be normalized and the labels need to be one-hot encoded.  You'll get to apply what you learned and build a convolutional, max pooling, dropout, and fully connected layers.  At the end, you'll get to see your neural network's predictions on the sample images.
## Get the Data
Run the following cell to download the [CIFAR-10 dataset for python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)
```

    All files found!
    

## Explore the Data
The dataset is broken into batches to prevent your machine from running out of memory.  The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Understanding a dataset is part of making predictions on the data.  Play around with the code cell below by changing the `batch_id` and `sample_id`. The `batch_id` is the id for a batch (1-5). The `sample_id` is the id for a image and label pair in the batch.

Ask yourself "What are all possible labels?", "What is the range of values for the image data?", "Are the labels in order or random?".  Answers to questions like these will help you preprocess the data and end up with better predictions.


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper
import numpy as np

# Explore the dataset
batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
```

    
    Stats of batch 1:
    Samples: 10000
    Label Counts: {0: 1005, 1: 974, 2: 1032, 3: 1016, 4: 999, 5: 937, 6: 1030, 7: 1001, 8: 1025, 9: 981}
    First 20 Labels: [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7, 7, 2, 9, 9, 9, 3, 2, 6]
    
    Example of Image 5:
    Image - Min Value: 0 Max Value: 252
    Image - Shape: (32, 32, 3)
    Label - Label Id: 1 Name: automobile
    


![png](output_3_1.png)
	<img src="{{ site.img_path }}/output_3_1.png" width="10%">

## Implement Preprocess Functions
### Normalize
In the cell below, implement the `normalize` function to take in image data, `x`, and return it as a normalized Numpy array. The values should be in the range of 0 to 1, inclusive.  The return object should be the same shape as `x`.


```python
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    a = 0
    b = 1
    grayscale_min = np.min(x)
    grayscale_max = np.max(x)
    return a + ( ( (x - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)
```

    Tests Passed
    

### One-hot encode
Just like the previous code cell, you'll be implementing a function for preprocessing.  This time, you'll implement the `one_hot_encode` function. The input, `x`, are a list of labels.  Implement the function to return the list of labels as One-Hot encoded Numpy array.  The possible values for labels are 0 to 9. The one-hot encoding function should return the same encoding for each value between each call to `one_hot_encode`.  Make sure to save the map of encodings outside the function.

Hint: Don't reinvent the wheel.


```python
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    import numpy as np
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    encoder.fit(np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]))
    y = encoder.transform(x)
    y = y.astype(np.float32)
    return y


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)
```

    Tests Passed
    

### Randomize Data
As you saw from exploring the data above, the order of the samples are randomized.  It doesn't hurt to randomize it again, but you don't need to for this dataset.

## Preprocess all the data and save it
Running the code cell below will preprocess all the CIFAR-10 data and save it to file. The code below also uses 10% of the training data for validation.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
```

# Check Point
This is your first checkpoint.  If you ever decide to come back to this notebook or have to restart the notebook, you can start from here.  The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
```

## Build the network
For the neural network, you'll build each layer into a function.  Most of the code you've seen has been outside of functions. To test your code more thoroughly, we require that you put each layer in a function.  This allows us to give you better feedback and test for simple mistakes using our unittests before you submit your project.

>**Note:** If you're finding it hard to dedicate enough time for this course each week, we've provided a small shortcut to this part of the project. In the next couple of problems, you'll have the option to use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages to build each layer, except the layers you build in the "Convolutional and Max Pooling Layer" section.  TF Layers is similar to Keras's and TFLearn's abstraction to layers, so it's easy to pickup.

>However, if you would like to get the most out of this course, try to solve all the problems _without_ using anything from the TF Layers packages. You **can** still use classes from other packages that happen to have the same name as ones you find in TF Layers! For example, instead of using the TF Layers version of the `conv2d` class, [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d), you would want to use the TF Neural Network version of `conv2d`, [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d). 

Let's begin!

### Input
The neural network needs to read the image data, one-hot encoded labels, and dropout keep probability. Implement the following functions
* Implement `neural_net_image_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `image_shape` with batch size set to `None`.
 * Name the TensorFlow placeholder "x" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_label_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `n_classes` with batch size set to `None`.
 * Name the TensorFlow placeholder "y" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_keep_prob_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) for dropout keep probability.
 * Name the TensorFlow placeholder "keep_prob" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).

These names will be used at the end of the project to load your saved model.

Note: `None` for shapes in TensorFlow allow for a dynamic size.


```python
import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return  tf.placeholder(tf.float32, [None, image_shape[0],image_shape[1],image_shape[2]] , name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return  tf.placeholder(tf.float32, [None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, name='keep_prob')


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
```

    Image Input Tests Passed.
    Label Input Tests Passed.
    Keep Prob Tests Passed.
    

### Convolution and Max Pooling Layer
Convolution layers have a lot of success with images. For this code cell, you should implement the function `conv2d_maxpool` to apply convolution then max pooling:
* Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
* Apply a convolution to `x_tensor` using weight and `conv_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.
* Add bias
* Add a nonlinear activation to the convolution.
* Apply Max Pooling using `pool_ksize` and `pool_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.

**Note:** You **can't** use [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) for **this** layer, but you can still use TensorFlow's [Neural Network](https://www.tensorflow.org/api_docs/python/tf/nn) package. You may still use the shortcut option for all the **other** layers.


```python
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    x_depth = x_tensor.get_shape().as_list()[-1]
    #y_depth = conv_num_outputs
    weight= tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], x_depth, conv_num_outputs],stddev=0.1))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    convolution  = tf.nn.conv2d(x_tensor, weight, [1, conv_strides[0], conv_strides[1], 1], 'SAME') + bias
    #convolution  = tf.nn.bias_add(convolution, bias)
    convolution  = tf.nn.relu(convolution)
    convolution =  tf.nn.max_pool(convolution,[1, pool_ksize[0], pool_ksize[1], 1],[1, pool_strides[0], pool_strides[1], 1],'SAME')
    return convolution 


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)
```

    Tests Passed
    

### Flatten Layer
Implement the `flatten` function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor.  The output should be the shape (*Batch Size*, *Flattened Image Size*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    batch_size, *img_size = x_tensor.get_shape().as_list()
    img_size = img_size[0] * img_size[1] * img_size[2]
    return tf.reshape(x_tensor, [-1, img_size])


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_flatten(flatten)
```

    Tests Passed
    

### Fully-Connected Layer
Implement the `fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), bias))


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)
```

    Tests Passed
    

### Output Layer
Implement the `output` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.

**Note:** Activation, softmax, or cross entropy should **not** be applied to this.


```python
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul(x_tensor, weight), bias)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)
```

    Tests Passed
    

### Create Convolutional Model
Implement the function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits.  Use the layers you created above to create this model:

* Apply 1, 2, or 3 Convolution and Max Pool layers
* Apply a Flatten Layer
* Apply 1, 2, or 3 Fully Connected Layers
* Apply an Output Layer
* Return the output
* Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using `keep_prob`. 


```python
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    #layer = conv2d_maxpool(x, 16, (6,6),(1,1),(2,2),(2,2))
    #layer = conv2d_maxpool(layer, 32, (4,4), (1,1), (1,1), (1,1))
    layer = conv2d_maxpool(x, 64, (4,4), (1,1), (2,2), (2,2))
    tf.nn.dropout(layer, keep_prob=keep_prob)
  

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    layer = flatten(layer)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    layer = fully_conn(layer,500)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = fully_conn(layer,100)
    layer = tf.nn.dropout(layer, keep_prob)
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    
    
    # TODO: return output
    return output(layer,10)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)
```

    Neural Network Built!
    

## Train the Neural Network
### Single Optimization
Implement the function `train_neural_network` to do a single optimization.  The optimization should use `optimizer` to optimize in `session` with a `feed_dict` of the following:
* `x` for image input
* `y` for labels
* `keep_prob` for keep probability for dropout

This function will be called for each batch, so `tf.global_variables_initializer()` has already been called.

Note: Nothing needs to be returned. This function is only optimizing the neural network.


```python
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_train_nn(train_neural_network)
```

    Tests Passed
    

### Show Stats
Implement the function `print_stats` to print loss and validation accuracy.  Use the global variables `valid_features` and `valid_labels` to calculate validation accuracy.  Use a keep probability of `1.0` to calculate the loss and validation accuracy.


```python
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    cost = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    validation_accuracy = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
    print('Cost = {0} - Validation Accuracy = {1}'.format(cost, validation_accuracy))

```

### Hyperparameters
Tune the following parameters:
* Set `epochs` to the number of iterations until the network stops learning or start overfitting
* Set `batch_size` to the highest number that your machine has memory for.  Most people set them to common sizes of memory:
 * 64
 * 128
 * 256
 * ...
* Set `keep_probability` to the probability of keeping a node using dropout


```python
# TODO: Tune Parameters
epochs = 45
batch_size = 256
keep_probability = 0.5
```

### Train on a Single CIFAR-10 Batch
Instead of training the neural network on all the CIFAR-10 batches of data, let's use a single batch. This should save time while you iterate on the model to get a better accuracy.  Once the final validation accuracy is 50% or greater, run the model on all the data in the next section.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
```

    Checking the Training on a Single Batch...
    Epoch  1, CIFAR-10 Batch 1:  Cost = 2.1485657691955566 - Validation Accuracy = 0.24219997227191925
    Epoch  2, CIFAR-10 Batch 1:  Cost = 1.9556281566619873 - Validation Accuracy = 0.33879998326301575
    Epoch  3, CIFAR-10 Batch 1:  Cost = 1.7695608139038086 - Validation Accuracy = 0.40059995651245117
    Epoch  4, CIFAR-10 Batch 1:  Cost = 1.581284761428833 - Validation Accuracy = 0.40939998626708984
    Epoch  5, CIFAR-10 Batch 1:  Cost = 1.374506950378418 - Validation Accuracy = 0.43039995431900024
    Epoch  6, CIFAR-10 Batch 1:  Cost = 1.1556897163391113 - Validation Accuracy = 0.4673999547958374
    Epoch  7, CIFAR-10 Batch 1:  Cost = 1.0067230463027954 - Validation Accuracy = 0.4889999330043793
    Epoch  8, CIFAR-10 Batch 1:  Cost = 0.9074106216430664 - Validation Accuracy = 0.4957999289035797
    Epoch  9, CIFAR-10 Batch 1:  Cost = 0.7944024801254272 - Validation Accuracy = 0.5091999769210815
    Epoch 10, CIFAR-10 Batch 1:  Cost = 0.6911913156509399 - Validation Accuracy = 0.5257999300956726
    Epoch 11, CIFAR-10 Batch 1:  Cost = 0.6137291789054871 - Validation Accuracy = 0.5181999206542969
    Epoch 12, CIFAR-10 Batch 1:  Cost = 0.5314263105392456 - Validation Accuracy = 0.5339999198913574
    Epoch 13, CIFAR-10 Batch 1:  Cost = 0.4951411485671997 - Validation Accuracy = 0.5247999429702759
    Epoch 14, CIFAR-10 Batch 1:  Cost = 0.4075576364994049 - Validation Accuracy = 0.5203999280929565
    Epoch 15, CIFAR-10 Batch 1:  Cost = 0.3852843642234802 - Validation Accuracy = 0.5175999402999878
    Epoch 16, CIFAR-10 Batch 1:  Cost = 0.30147936940193176 - Validation Accuracy = 0.5331999063491821
    Epoch 17, CIFAR-10 Batch 1:  Cost = 0.2582905888557434 - Validation Accuracy = 0.5477999448776245
    Epoch 18, CIFAR-10 Batch 1:  Cost = 0.22398968040943146 - Validation Accuracy = 0.5389999151229858
    Epoch 19, CIFAR-10 Batch 1:  Cost = 0.1869264394044876 - Validation Accuracy = 0.5565999746322632
    Epoch 20, CIFAR-10 Batch 1:  Cost = 0.16749420762062073 - Validation Accuracy = 0.5511999130249023
    Epoch 21, CIFAR-10 Batch 1:  Cost = 0.14295132458209991 - Validation Accuracy = 0.5501999258995056
    Epoch 22, CIFAR-10 Batch 1:  Cost = 0.11564118415117264 - Validation Accuracy = 0.5601999759674072
    Epoch 23, CIFAR-10 Batch 1:  Cost = 0.11416701972484589 - Validation Accuracy = 0.5595999360084534
    Epoch 24, CIFAR-10 Batch 1:  Cost = 0.08596203476190567 - Validation Accuracy = 0.5601999163627625
    Epoch 25, CIFAR-10 Batch 1:  Cost = 0.06862218677997589 - Validation Accuracy = 0.5679999589920044
    Epoch 26, CIFAR-10 Batch 1:  Cost = 0.04974818229675293 - Validation Accuracy = 0.5745998620986938
    Epoch 27, CIFAR-10 Batch 1:  Cost = 0.04549533501267433 - Validation Accuracy = 0.5657999515533447
    Epoch 28, CIFAR-10 Batch 1:  Cost = 0.03608117252588272 - Validation Accuracy = 0.5675999522209167
    Epoch 29, CIFAR-10 Batch 1:  Cost = 0.028677256777882576 - Validation Accuracy = 0.5759999752044678
    Epoch 30, CIFAR-10 Batch 1:  Cost = 0.02756013721227646 - Validation Accuracy = 0.5699998736381531
    Epoch 31, CIFAR-10 Batch 1:  Cost = 0.021485690027475357 - Validation Accuracy = 0.5797999501228333
    Epoch 32, CIFAR-10 Batch 1:  Cost = 0.012499160133302212 - Validation Accuracy = 0.5799999237060547
    Epoch 33, CIFAR-10 Batch 1:  Cost = 0.01440395973622799 - Validation Accuracy = 0.5753999352455139
    Epoch 34, CIFAR-10 Batch 1:  Cost = 0.011615422554314137 - Validation Accuracy = 0.5727999210357666
    Epoch 35, CIFAR-10 Batch 1:  Cost = 0.013035455718636513 - Validation Accuracy = 0.5769999027252197
    Epoch 36, CIFAR-10 Batch 1:  Cost = 0.00868083443492651 - Validation Accuracy = 0.5669999718666077
    Epoch 37, CIFAR-10 Batch 1:  Cost = 0.009316467680037022 - Validation Accuracy = 0.5703998804092407
    Epoch 38, CIFAR-10 Batch 1:  Cost = 0.005669460631906986 - Validation Accuracy = 0.569399893283844
    Epoch 39, CIFAR-10 Batch 1:  Cost = 0.0032136263325810432 - Validation Accuracy = 0.5707999467849731
    Epoch 40, CIFAR-10 Batch 1:  Cost = 0.0031791268847882748 - Validation Accuracy = 0.576200008392334
    Epoch 41, CIFAR-10 Batch 1:  Cost = 0.0029399653431028128 - Validation Accuracy = 0.5735999345779419
    Epoch 42, CIFAR-10 Batch 1:  Cost = 0.003802852239459753 - Validation Accuracy = 0.5741999745368958
    Epoch 43, CIFAR-10 Batch 1:  Cost = 0.0027553949039429426 - Validation Accuracy = 0.5803999304771423
    Epoch 44, CIFAR-10 Batch 1:  Cost = 0.0015565428184345365 - Validation Accuracy = 0.5673999190330505
    Epoch 45, CIFAR-10 Batch 1:  Cost = 0.0017994245281443 - Validation Accuracy = 0.5591999292373657
    

### Fully Train the Model
Now that you got a good accuracy with a single CIFAR-10 batch, try it with all five batches.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
```

    Training...
    Epoch  1, CIFAR-10 Batch 1:  Cost = 2.1811623573303223 - Validation Accuracy = 0.18539997935295105
    Epoch  1, CIFAR-10 Batch 2:  Cost = 1.9584850072860718 - Validation Accuracy = 0.31520000100135803
    Epoch  1, CIFAR-10 Batch 3:  Cost = 1.6230461597442627 - Validation Accuracy = 0.3877999782562256
    Epoch  1, CIFAR-10 Batch 4:  Cost = 1.6687123775482178 - Validation Accuracy = 0.4227999746799469
    Epoch  1, CIFAR-10 Batch 5:  Cost = 1.5890612602233887 - Validation Accuracy = 0.4399999976158142
    Epoch  2, CIFAR-10 Batch 1:  Cost = 1.6975443363189697 - Validation Accuracy = 0.45799997448921204
    Epoch  2, CIFAR-10 Batch 2:  Cost = 1.4218182563781738 - Validation Accuracy = 0.46879997849464417
    Epoch  2, CIFAR-10 Batch 3:  Cost = 1.161777377128601 - Validation Accuracy = 0.4997999668121338
    Epoch  2, CIFAR-10 Batch 4:  Cost = 1.4113024473190308 - Validation Accuracy = 0.5035999417304993
    Epoch  2, CIFAR-10 Batch 5:  Cost = 1.3082618713378906 - Validation Accuracy = 0.5079999566078186
    Epoch  3, CIFAR-10 Batch 1:  Cost = 1.318356990814209 - Validation Accuracy = 0.5321999788284302
    Epoch  3, CIFAR-10 Batch 2:  Cost = 1.094077467918396 - Validation Accuracy = 0.5389999151229858
    Epoch  3, CIFAR-10 Batch 3:  Cost = 0.8922118544578552 - Validation Accuracy = 0.5281999111175537
    Epoch  3, CIFAR-10 Batch 4:  Cost = 1.1266894340515137 - Validation Accuracy = 0.5593999624252319
    Epoch  3, CIFAR-10 Batch 5:  Cost = 0.9697490930557251 - Validation Accuracy = 0.5627999305725098
    Epoch  4, CIFAR-10 Batch 1:  Cost = 1.0761635303497314 - Validation Accuracy = 0.556999921798706
    Epoch  4, CIFAR-10 Batch 2:  Cost = 0.8976505994796753 - Validation Accuracy = 0.569399893283844
    Epoch  4, CIFAR-10 Batch 3:  Cost = 0.7298280596733093 - Validation Accuracy = 0.5779999494552612
    Epoch  4, CIFAR-10 Batch 4:  Cost = 0.9780511856079102 - Validation Accuracy = 0.5749998688697815
    Epoch  4, CIFAR-10 Batch 5:  Cost = 0.867372989654541 - Validation Accuracy = 0.5877999067306519
    Epoch  5, CIFAR-10 Batch 1:  Cost = 0.8423333168029785 - Validation Accuracy = 0.577799916267395
    Epoch  5, CIFAR-10 Batch 2:  Cost = 0.725346028804779 - Validation Accuracy = 0.590999960899353
    Epoch  5, CIFAR-10 Batch 3:  Cost = 0.6361026763916016 - Validation Accuracy = 0.5845999121665955
    Epoch  5, CIFAR-10 Batch 4:  Cost = 0.7695732116699219 - Validation Accuracy = 0.5995999574661255
    Epoch  5, CIFAR-10 Batch 5:  Cost = 0.684511661529541 - Validation Accuracy = 0.6073999404907227
    Epoch  6, CIFAR-10 Batch 1:  Cost = 0.6892574429512024 - Validation Accuracy = 0.6009998917579651
    Epoch  6, CIFAR-10 Batch 2:  Cost = 0.6148948669433594 - Validation Accuracy = 0.6009999513626099
    Epoch  6, CIFAR-10 Batch 3:  Cost = 0.5389288663864136 - Validation Accuracy = 0.605199933052063
    Epoch  6, CIFAR-10 Batch 4:  Cost = 0.659376859664917 - Validation Accuracy = 0.6067999005317688
    Epoch  6, CIFAR-10 Batch 5:  Cost = 0.5738847255706787 - Validation Accuracy = 0.6195999383926392
    Epoch  7, CIFAR-10 Batch 1:  Cost = 0.5839495062828064 - Validation Accuracy = 0.6061999201774597
    Epoch  7, CIFAR-10 Batch 2:  Cost = 0.49656039476394653 - Validation Accuracy = 0.6191999912261963
    Epoch  7, CIFAR-10 Batch 3:  Cost = 0.40735548734664917 - Validation Accuracy = 0.6243999600410461
    Epoch  7, CIFAR-10 Batch 4:  Cost = 0.5513085722923279 - Validation Accuracy = 0.6209999322891235
    Epoch  7, CIFAR-10 Batch 5:  Cost = 0.4443703293800354 - Validation Accuracy = 0.6275998950004578
    Epoch  8, CIFAR-10 Batch 1:  Cost = 0.45770615339279175 - Validation Accuracy = 0.6215999126434326
    Epoch  8, CIFAR-10 Batch 2:  Cost = 0.42067572474479675 - Validation Accuracy = 0.6319998502731323
    Epoch  8, CIFAR-10 Batch 3:  Cost = 0.3599691092967987 - Validation Accuracy = 0.6373999118804932
    Epoch  8, CIFAR-10 Batch 4:  Cost = 0.3793061375617981 - Validation Accuracy = 0.6339998841285706
    Epoch  8, CIFAR-10 Batch 5:  Cost = 0.37761759757995605 - Validation Accuracy = 0.6333999037742615
    Epoch  9, CIFAR-10 Batch 1:  Cost = 0.40303468704223633 - Validation Accuracy = 0.6207998991012573
    Epoch  9, CIFAR-10 Batch 2:  Cost = 0.3016594350337982 - Validation Accuracy = 0.6339999437332153
    Epoch  9, CIFAR-10 Batch 3:  Cost = 0.30533456802368164 - Validation Accuracy = 0.6401998996734619
    Epoch  9, CIFAR-10 Batch 4:  Cost = 0.3292393684387207 - Validation Accuracy = 0.6441998481750488
    Epoch  9, CIFAR-10 Batch 5:  Cost = 0.2791125178337097 - Validation Accuracy = 0.6501998901367188
    Epoch 10, CIFAR-10 Batch 1:  Cost = 0.3611883819103241 - Validation Accuracy = 0.6269999146461487
    Epoch 10, CIFAR-10 Batch 2:  Cost = 0.265232115983963 - Validation Accuracy = 0.6399998664855957
    Epoch 10, CIFAR-10 Batch 3:  Cost = 0.21370574831962585 - Validation Accuracy = 0.6525999307632446
    Epoch 10, CIFAR-10 Batch 4:  Cost = 0.2971021831035614 - Validation Accuracy = 0.6455998420715332
    Epoch 10, CIFAR-10 Batch 5:  Cost = 0.21383322775363922 - Validation Accuracy = 0.6547998785972595
    Epoch 11, CIFAR-10 Batch 1:  Cost = 0.31287872791290283 - Validation Accuracy = 0.6515998840332031
    Epoch 11, CIFAR-10 Batch 2:  Cost = 0.20220965147018433 - Validation Accuracy = 0.6429998874664307
    Epoch 11, CIFAR-10 Batch 3:  Cost = 0.19101281464099884 - Validation Accuracy = 0.6561998724937439
    Epoch 11, CIFAR-10 Batch 4:  Cost = 0.23675405979156494 - Validation Accuracy = 0.6581998467445374
    Epoch 11, CIFAR-10 Batch 5:  Cost = 0.16420884430408478 - Validation Accuracy = 0.6493998765945435
    Epoch 12, CIFAR-10 Batch 1:  Cost = 0.26295560598373413 - Validation Accuracy = 0.6495999097824097
    Epoch 12, CIFAR-10 Batch 2:  Cost = 0.15415263175964355 - Validation Accuracy = 0.6477998495101929
    Epoch 12, CIFAR-10 Batch 3:  Cost = 0.15891015529632568 - Validation Accuracy = 0.6539998650550842
    Epoch 12, CIFAR-10 Batch 4:  Cost = 0.17362220585346222 - Validation Accuracy = 0.6577998399734497
    Epoch 12, CIFAR-10 Batch 5:  Cost = 0.13566216826438904 - Validation Accuracy = 0.6671998500823975
    Epoch 13, CIFAR-10 Batch 1:  Cost = 0.20044055581092834 - Validation Accuracy = 0.6439998745918274
    Epoch 13, CIFAR-10 Batch 2:  Cost = 0.15572568774223328 - Validation Accuracy = 0.6571998596191406
    Epoch 13, CIFAR-10 Batch 3:  Cost = 0.1294330358505249 - Validation Accuracy = 0.6505998969078064
    Epoch 13, CIFAR-10 Batch 4:  Cost = 0.17886698246002197 - Validation Accuracy = 0.6645998954772949
    Epoch 13, CIFAR-10 Batch 5:  Cost = 0.1121206283569336 - Validation Accuracy = 0.6533998847007751
    Epoch 14, CIFAR-10 Batch 1:  Cost = 0.19587206840515137 - Validation Accuracy = 0.6613999605178833
    Epoch 14, CIFAR-10 Batch 2:  Cost = 0.10635516047477722 - Validation Accuracy = 0.6607998609542847
    Epoch 14, CIFAR-10 Batch 3:  Cost = 0.11580963432788849 - Validation Accuracy = 0.6523998975753784
    Epoch 14, CIFAR-10 Batch 4:  Cost = 0.13344885408878326 - Validation Accuracy = 0.6653998494148254
    Epoch 14, CIFAR-10 Batch 5:  Cost = 0.0907227098941803 - Validation Accuracy = 0.6611999273300171
    Epoch 15, CIFAR-10 Batch 1:  Cost = 0.1783110350370407 - Validation Accuracy = 0.6469998359680176
    Epoch 15, CIFAR-10 Batch 2:  Cost = 0.0775488018989563 - Validation Accuracy = 0.6507999300956726
    Epoch 15, CIFAR-10 Batch 3:  Cost = 0.09549462795257568 - Validation Accuracy = 0.66159987449646
    Epoch 15, CIFAR-10 Batch 4:  Cost = 0.11135688424110413 - Validation Accuracy = 0.6613999009132385
    Epoch 15, CIFAR-10 Batch 5:  Cost = 0.06292182952165604 - Validation Accuracy = 0.6685999035835266
    Epoch 16, CIFAR-10 Batch 1:  Cost = 0.13217127323150635 - Validation Accuracy = 0.6593998670578003
    Epoch 16, CIFAR-10 Batch 2:  Cost = 0.05809121951460838 - Validation Accuracy = 0.6607998609542847
    Epoch 16, CIFAR-10 Batch 3:  Cost = 0.07866101711988449 - Validation Accuracy = 0.6589998602867126
    Epoch 16, CIFAR-10 Batch 4:  Cost = 0.11868923157453537 - Validation Accuracy = 0.6629998683929443
    Epoch 16, CIFAR-10 Batch 5:  Cost = 0.04895960912108421 - Validation Accuracy = 0.6693998575210571
    Epoch 17, CIFAR-10 Batch 1:  Cost = 0.09686502814292908 - Validation Accuracy = 0.6555998921394348
    Epoch 17, CIFAR-10 Batch 2:  Cost = 0.058843646198511124 - Validation Accuracy = 0.6545999050140381
    Epoch 17, CIFAR-10 Batch 3:  Cost = 0.06467927247285843 - Validation Accuracy = 0.6571999192237854
    Epoch 17, CIFAR-10 Batch 4:  Cost = 0.08178563416004181 - Validation Accuracy = 0.66159987449646
    Epoch 17, CIFAR-10 Batch 5:  Cost = 0.04623532295227051 - Validation Accuracy = 0.6543998718261719
    Epoch 18, CIFAR-10 Batch 1:  Cost = 0.08406947553157806 - Validation Accuracy = 0.6533998847007751
    Epoch 18, CIFAR-10 Batch 2:  Cost = 0.053819119930267334 - Validation Accuracy = 0.6571999192237854
    Epoch 18, CIFAR-10 Batch 3:  Cost = 0.051309000700712204 - Validation Accuracy = 0.6625999212265015
    Epoch 18, CIFAR-10 Batch 4:  Cost = 0.053479500114917755 - Validation Accuracy = 0.663399875164032
    Epoch 18, CIFAR-10 Batch 5:  Cost = 0.03291318938136101 - Validation Accuracy = 0.6577999591827393
    Epoch 19, CIFAR-10 Batch 1:  Cost = 0.0845826044678688 - Validation Accuracy = 0.6511999368667603
    Epoch 19, CIFAR-10 Batch 2:  Cost = 0.05210895091295242 - Validation Accuracy = 0.6583998799324036
    Epoch 19, CIFAR-10 Batch 3:  Cost = 0.056531500071287155 - Validation Accuracy = 0.6611998677253723
    Epoch 19, CIFAR-10 Batch 4:  Cost = 0.053767550736665726 - Validation Accuracy = 0.6573998928070068
    Epoch 19, CIFAR-10 Batch 5:  Cost = 0.02408386766910553 - Validation Accuracy = 0.6627999544143677
    Epoch 20, CIFAR-10 Batch 1:  Cost = 0.06639882177114487 - Validation Accuracy = 0.6613999009132385
    Epoch 20, CIFAR-10 Batch 2:  Cost = 0.04071289300918579 - Validation Accuracy = 0.6571999192237854
    Epoch 20, CIFAR-10 Batch 3:  Cost = 0.0342855229973793 - Validation Accuracy = 0.6569998860359192
    Epoch 20, CIFAR-10 Batch 4:  Cost = 0.03909813240170479 - Validation Accuracy = 0.6643998622894287
    Epoch 20, CIFAR-10 Batch 5:  Cost = 0.021189263090491295 - Validation Accuracy = 0.6655998826026917
    Epoch 21, CIFAR-10 Batch 1:  Cost = 0.05197606235742569 - Validation Accuracy = 0.6549999117851257
    Epoch 21, CIFAR-10 Batch 2:  Cost = 0.024691374972462654 - Validation Accuracy = 0.662199854850769
    Epoch 21, CIFAR-10 Batch 3:  Cost = 0.03122517466545105 - Validation Accuracy = 0.6657998561859131
    Epoch 21, CIFAR-10 Batch 4:  Cost = 0.046056777238845825 - Validation Accuracy = 0.6545999050140381
    Epoch 21, CIFAR-10 Batch 5:  Cost = 0.01546941976994276 - Validation Accuracy = 0.6601998805999756
    Epoch 22, CIFAR-10 Batch 1:  Cost = 0.05320039391517639 - Validation Accuracy = 0.6637999415397644
    Epoch 22, CIFAR-10 Batch 2:  Cost = 0.04059157520532608 - Validation Accuracy = 0.6663998961448669
    Epoch 22, CIFAR-10 Batch 3:  Cost = 0.024964045733213425 - Validation Accuracy = 0.6701998710632324
    Epoch 22, CIFAR-10 Batch 4:  Cost = 0.037080492824316025 - Validation Accuracy = 0.6703998446464539
    Epoch 22, CIFAR-10 Batch 5:  Cost = 0.009691141545772552 - Validation Accuracy = 0.6683998107910156
    Epoch 23, CIFAR-10 Batch 1:  Cost = 0.03670954704284668 - Validation Accuracy = 0.6435999274253845
    Epoch 23, CIFAR-10 Batch 2:  Cost = 0.028754815459251404 - Validation Accuracy = 0.66159987449646
    Epoch 23, CIFAR-10 Batch 3:  Cost = 0.012782066129148006 - Validation Accuracy = 0.6643998622894287
    Epoch 23, CIFAR-10 Batch 4:  Cost = 0.02388462983071804 - Validation Accuracy = 0.6691999435424805
    Epoch 23, CIFAR-10 Batch 5:  Cost = 0.011726578697562218 - Validation Accuracy = 0.6685998439788818
    Epoch 24, CIFAR-10 Batch 1:  Cost = 0.031538233160972595 - Validation Accuracy = 0.6615999341011047
    Epoch 24, CIFAR-10 Batch 2:  Cost = 0.022785622626543045 - Validation Accuracy = 0.6739998459815979
    Epoch 24, CIFAR-10 Batch 3:  Cost = 0.007632040418684483 - Validation Accuracy = 0.6649999022483826
    Epoch 24, CIFAR-10 Batch 4:  Cost = 0.018511448055505753 - Validation Accuracy = 0.6587998867034912
    Epoch 24, CIFAR-10 Batch 5:  Cost = 0.006494627799838781 - Validation Accuracy = 0.666999876499176
    Epoch 25, CIFAR-10 Batch 1:  Cost = 0.02493128925561905 - Validation Accuracy = 0.6553998589515686
    Epoch 25, CIFAR-10 Batch 2:  Cost = 0.021074315533041954 - Validation Accuracy = 0.6673998832702637
    Epoch 25, CIFAR-10 Batch 3:  Cost = 0.010587708093225956 - Validation Accuracy = 0.6685999035835266
    Epoch 25, CIFAR-10 Batch 4:  Cost = 0.030966220423579216 - Validation Accuracy = 0.6667999029159546
    Epoch 25, CIFAR-10 Batch 5:  Cost = 0.005528775043785572 - Validation Accuracy = 0.6675999164581299
    Epoch 26, CIFAR-10 Batch 1:  Cost = 0.0187509935349226 - Validation Accuracy = 0.6587998867034912
    Epoch 26, CIFAR-10 Batch 2:  Cost = 0.015595516189932823 - Validation Accuracy = 0.6659998297691345
    Epoch 26, CIFAR-10 Batch 3:  Cost = 0.009900491684675217 - Validation Accuracy = 0.6593999266624451
    Epoch 26, CIFAR-10 Batch 4:  Cost = 0.026268253102898598 - Validation Accuracy = 0.6587998867034912
    Epoch 26, CIFAR-10 Batch 5:  Cost = 0.004856211133301258 - Validation Accuracy = 0.6713998913764954
    Epoch 27, CIFAR-10 Batch 1:  Cost = 0.022516800090670586 - Validation Accuracy = 0.6463999152183533
    Epoch 27, CIFAR-10 Batch 2:  Cost = 0.009971871972084045 - Validation Accuracy = 0.6667998433113098
    Epoch 27, CIFAR-10 Batch 3:  Cost = 0.006218905095010996 - Validation Accuracy = 0.6619998812675476
    Epoch 27, CIFAR-10 Batch 4:  Cost = 0.01596318557858467 - Validation Accuracy = 0.653799831867218
    Epoch 27, CIFAR-10 Batch 5:  Cost = 0.004206402227282524 - Validation Accuracy = 0.6693998575210571
    Epoch 28, CIFAR-10 Batch 1:  Cost = 0.015472047962248325 - Validation Accuracy = 0.6593998670578003
    Epoch 28, CIFAR-10 Batch 2:  Cost = 0.007865852676331997 - Validation Accuracy = 0.6693998575210571
    Epoch 28, CIFAR-10 Batch 3:  Cost = 0.004838103428483009 - Validation Accuracy = 0.6751998662948608
    Epoch 28, CIFAR-10 Batch 4:  Cost = 0.011746803298592567 - Validation Accuracy = 0.6547998785972595
    Epoch 28, CIFAR-10 Batch 5:  Cost = 0.0029801935888826847 - Validation Accuracy = 0.6631999015808105
    Epoch 29, CIFAR-10 Batch 1:  Cost = 0.01960574835538864 - Validation Accuracy = 0.650999903678894
    Epoch 29, CIFAR-10 Batch 2:  Cost = 0.006634612567722797 - Validation Accuracy = 0.6661999225616455
    Epoch 29, CIFAR-10 Batch 3:  Cost = 0.003191433846950531 - Validation Accuracy = 0.6661998629570007
    Epoch 29, CIFAR-10 Batch 4:  Cost = 0.007825599052011967 - Validation Accuracy = 0.6663998365402222
    Epoch 29, CIFAR-10 Batch 5:  Cost = 0.0029130885377526283 - Validation Accuracy = 0.6685998439788818
    Epoch 30, CIFAR-10 Batch 1:  Cost = 0.012895254418253899 - Validation Accuracy = 0.6641998887062073
    Epoch 30, CIFAR-10 Batch 2:  Cost = 0.012302185408771038 - Validation Accuracy = 0.6745998859405518
    Epoch 30, CIFAR-10 Batch 3:  Cost = 0.0027916717808693647 - Validation Accuracy = 0.6673999428749084
    Epoch 30, CIFAR-10 Batch 4:  Cost = 0.0056531354784965515 - Validation Accuracy = 0.662199854850769
    Epoch 30, CIFAR-10 Batch 5:  Cost = 0.0032759862951934338 - Validation Accuracy = 0.6699998378753662
    Epoch 31, CIFAR-10 Batch 1:  Cost = 0.012614419683814049 - Validation Accuracy = 0.645599901676178
    Epoch 31, CIFAR-10 Batch 2:  Cost = 0.006445364095270634 - Validation Accuracy = 0.6723998785018921
    Epoch 31, CIFAR-10 Batch 3:  Cost = 0.0019524764502421021 - Validation Accuracy = 0.6679998636245728
    Epoch 31, CIFAR-10 Batch 4:  Cost = 0.004196139518171549 - Validation Accuracy = 0.6677998900413513
    Epoch 31, CIFAR-10 Batch 5:  Cost = 0.003365949960425496 - Validation Accuracy = 0.6663998961448669
    Epoch 32, CIFAR-10 Batch 1:  Cost = 0.012483161874115467 - Validation Accuracy = 0.6641998887062073
    Epoch 32, CIFAR-10 Batch 2:  Cost = 0.006317113526165485 - Validation Accuracy = 0.6707998514175415
    Epoch 32, CIFAR-10 Batch 3:  Cost = 0.0026280051097273827 - Validation Accuracy = 0.6701998710632324
    Epoch 32, CIFAR-10 Batch 4:  Cost = 0.004565236158668995 - Validation Accuracy = 0.6737999320030212
    Epoch 32, CIFAR-10 Batch 5:  Cost = 0.003478959435597062 - Validation Accuracy = 0.6637998819351196
    Epoch 33, CIFAR-10 Batch 1:  Cost = 0.010324607603251934 - Validation Accuracy = 0.6527999043464661
    Epoch 33, CIFAR-10 Batch 2:  Cost = 0.007717241533100605 - Validation Accuracy = 0.6637998819351196
    Epoch 33, CIFAR-10 Batch 3:  Cost = 0.0016325084725394845 - Validation Accuracy = 0.6701999306678772
    Epoch 33, CIFAR-10 Batch 4:  Cost = 0.0029003836680203676 - Validation Accuracy = 0.6667999029159546
    Epoch 33, CIFAR-10 Batch 5:  Cost = 0.0010103737004101276 - Validation Accuracy = 0.6701998710632324
    Epoch 34, CIFAR-10 Batch 1:  Cost = 0.0066850390285253525 - Validation Accuracy = 0.6591999530792236
    Epoch 34, CIFAR-10 Batch 2:  Cost = 0.002386814681813121 - Validation Accuracy = 0.665199875831604
    Epoch 34, CIFAR-10 Batch 3:  Cost = 0.0008805051911622286 - Validation Accuracy = 0.6637998819351196
    Epoch 34, CIFAR-10 Batch 4:  Cost = 0.0017101236153393984 - Validation Accuracy = 0.6617999076843262
    Epoch 34, CIFAR-10 Batch 5:  Cost = 0.0013109631836414337 - Validation Accuracy = 0.6699998378753662
    Epoch 35, CIFAR-10 Batch 1:  Cost = 0.005285251419991255 - Validation Accuracy = 0.6683999300003052
    Epoch 35, CIFAR-10 Batch 2:  Cost = 0.002105340361595154 - Validation Accuracy = 0.6655998229980469
    Epoch 35, CIFAR-10 Batch 3:  Cost = 0.0019856037106364965 - Validation Accuracy = 0.6569998860359192
    Epoch 35, CIFAR-10 Batch 4:  Cost = 0.003490191651508212 - Validation Accuracy = 0.6661999225616455
    Epoch 35, CIFAR-10 Batch 5:  Cost = 0.0016874147113412619 - Validation Accuracy = 0.6729997992515564
    Epoch 36, CIFAR-10 Batch 1:  Cost = 0.00283743254840374 - Validation Accuracy = 0.6593998670578003
    Epoch 36, CIFAR-10 Batch 2:  Cost = 0.005635113455355167 - Validation Accuracy = 0.6713998913764954
    Epoch 36, CIFAR-10 Batch 3:  Cost = 0.001015348476357758 - Validation Accuracy = 0.6727998852729797
    Epoch 36, CIFAR-10 Batch 4:  Cost = 0.0016757077537477016 - Validation Accuracy = 0.6711997985839844
    Epoch 36, CIFAR-10 Batch 5:  Cost = 0.002047642134130001 - Validation Accuracy = 0.671799898147583
    Epoch 37, CIFAR-10 Batch 1:  Cost = 0.0033659362234175205 - Validation Accuracy = 0.665199875831604
    Epoch 37, CIFAR-10 Batch 2:  Cost = 0.0029200247954577208 - Validation Accuracy = 0.6757998466491699
    Epoch 37, CIFAR-10 Batch 3:  Cost = 0.0014993862714618444 - Validation Accuracy = 0.6551998853683472
    Epoch 37, CIFAR-10 Batch 4:  Cost = 0.0016865063225850463 - Validation Accuracy = 0.663399875164032
    Epoch 37, CIFAR-10 Batch 5:  Cost = 0.001766604371368885 - Validation Accuracy = 0.6695998907089233
    Epoch 38, CIFAR-10 Batch 1:  Cost = 0.0038788248784840107 - Validation Accuracy = 0.6603999137878418
    Epoch 38, CIFAR-10 Batch 2:  Cost = 0.001171788084320724 - Validation Accuracy = 0.6639999151229858
    Epoch 38, CIFAR-10 Batch 3:  Cost = 0.0007930172141641378 - Validation Accuracy = 0.66239994764328
    Epoch 38, CIFAR-10 Batch 4:  Cost = 0.001449519651941955 - Validation Accuracy = 0.6689998507499695
    Epoch 38, CIFAR-10 Batch 5:  Cost = 0.001233890769071877 - Validation Accuracy = 0.6683999300003052
    Epoch 39, CIFAR-10 Batch 1:  Cost = 0.0019883711356669664 - Validation Accuracy = 0.6599999070167542
    Epoch 39, CIFAR-10 Batch 2:  Cost = 0.0041258446872234344 - Validation Accuracy = 0.6611998677253723
    Epoch 39, CIFAR-10 Batch 3:  Cost = 0.0016383989714086056 - Validation Accuracy = 0.6585999131202698
    Epoch 39, CIFAR-10 Batch 4:  Cost = 0.0031344895251095295 - Validation Accuracy = 0.6667999029159546
    Epoch 39, CIFAR-10 Batch 5:  Cost = 0.0019876989535987377 - Validation Accuracy = 0.6611998677253723
    Epoch 40, CIFAR-10 Batch 1:  Cost = 0.003138413652777672 - Validation Accuracy = 0.6597998142242432
    Epoch 40, CIFAR-10 Batch 2:  Cost = 0.0010219828691333532 - Validation Accuracy = 0.6663998365402222
    Epoch 40, CIFAR-10 Batch 3:  Cost = 0.000410854525398463 - Validation Accuracy = 0.6671998500823975
    Epoch 40, CIFAR-10 Batch 4:  Cost = 0.0014980913838371634 - Validation Accuracy = 0.6673998832702637
    Epoch 40, CIFAR-10 Batch 5:  Cost = 0.0012542542535811663 - Validation Accuracy = 0.6667998433113098
    Epoch 41, CIFAR-10 Batch 1:  Cost = 0.0019268719479441643 - Validation Accuracy = 0.66159987449646
    Epoch 41, CIFAR-10 Batch 2:  Cost = 0.0014940788969397545 - Validation Accuracy = 0.6657998561859131
    Epoch 41, CIFAR-10 Batch 3:  Cost = 0.0005783720989711583 - Validation Accuracy = 0.6639999151229858
    Epoch 41, CIFAR-10 Batch 4:  Cost = 0.0012891984079033136 - Validation Accuracy = 0.669999897480011
    Epoch 41, CIFAR-10 Batch 5:  Cost = 0.0010297567350789905 - Validation Accuracy = 0.6659998297691345
    Epoch 42, CIFAR-10 Batch 1:  Cost = 0.0019327896879985929 - Validation Accuracy = 0.6649998426437378
    Epoch 42, CIFAR-10 Batch 2:  Cost = 0.0007563873077742755 - Validation Accuracy = 0.6671999096870422
    Epoch 42, CIFAR-10 Batch 3:  Cost = 0.00033068028278648853 - Validation Accuracy = 0.6659998297691345
    Epoch 42, CIFAR-10 Batch 4:  Cost = 0.0025618895888328552 - Validation Accuracy = 0.671799898147583
    Epoch 42, CIFAR-10 Batch 5:  Cost = 0.0004886928363703191 - Validation Accuracy = 0.6725998520851135
    Epoch 43, CIFAR-10 Batch 1:  Cost = 0.0016638105735182762 - Validation Accuracy = 0.6663998961448669
    Epoch 43, CIFAR-10 Batch 2:  Cost = 0.0009806005982682109 - Validation Accuracy = 0.6629998683929443
    Epoch 43, CIFAR-10 Batch 3:  Cost = 0.0005594820831902325 - Validation Accuracy = 0.6723998785018921
    Epoch 43, CIFAR-10 Batch 4:  Cost = 0.0030585655476897955 - Validation Accuracy = 0.6731998324394226
    Epoch 43, CIFAR-10 Batch 5:  Cost = 0.00048071963828988373 - Validation Accuracy = 0.668799877166748
    Epoch 44, CIFAR-10 Batch 1:  Cost = 0.002417758572846651 - Validation Accuracy = 0.6613999009132385
    Epoch 44, CIFAR-10 Batch 2:  Cost = 0.0016584513941779733 - Validation Accuracy = 0.6743998527526855
    Epoch 44, CIFAR-10 Batch 3:  Cost = 0.0002183383039664477 - Validation Accuracy = 0.6649999022483826
    Epoch 44, CIFAR-10 Batch 4:  Cost = 0.0007314432878047228 - Validation Accuracy = 0.6675999164581299
    Epoch 44, CIFAR-10 Batch 5:  Cost = 0.0008765717502683401 - Validation Accuracy = 0.6631998419761658
    Epoch 45, CIFAR-10 Batch 1:  Cost = 0.0019422125769779086 - Validation Accuracy = 0.6635999083518982
    Epoch 45, CIFAR-10 Batch 2:  Cost = 0.00036410350003279746 - Validation Accuracy = 0.6619998216629028
    Epoch 45, CIFAR-10 Batch 3:  Cost = 0.0005971952923573554 - Validation Accuracy = 0.6711999177932739
    Epoch 45, CIFAR-10 Batch 4:  Cost = 0.0009821756975725293 - Validation Accuracy = 0.6671999096870422
    Epoch 45, CIFAR-10 Batch 5:  Cost = 0.0007103838725015521 - Validation Accuracy = 0.6647998690605164
    

# Checkpoint
The model has been saved to disk.
## Test Model
Test your model against the test dataset.  This will be your final accuracy. You should have an accuracy greater than 50%. If you don't, keep tweaking the model architecture and parameters.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
```

    Testing Accuracy: 0.66904296875
    
    



![png](output_36_1.png)
	<img src="{{ site.img_path }}/output_36_1.png" width="50%">


## Why 50-70% Accuracy?
You might be wondering why you can't get an accuracy any higher. First things first, 50% isn't bad for a simple CNN.  Pure guessing would get you 10% accuracy. However, you might notice people are getting scores [well above 70%](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).  That's because we haven't taught you all there is to know about neural networks. We still need to cover a few more techniques.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook.  Save the notebook file as "dlnd_image_classification.ipynb" and save it as a HTML file under "File" -> "Download as".  Include the "helper.py" and "problem_unittests.py" files in your submission.
