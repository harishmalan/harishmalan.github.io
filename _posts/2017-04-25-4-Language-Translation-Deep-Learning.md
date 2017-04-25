
# Language Translation
In this project, you’re going to take a peek into the realm of neural network machine translation.  You’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
## Get the Data
Since translating the whole language of English to French will take lots of time to train, we have provided you with a small portion of the English corpus.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)
```

## Explore the Data
Play around with view_sentence_range to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 227
    Number of sentences: 137861
    Average number of words in a sentence: 13.225277634719028
    
    English sentences 0 to 10:
    new jersey is sometimes quiet during autumn , and it is snowy in april .
    the united states is usually chilly during july , and it is usually freezing in november .
    california is usually quiet during march , and it is usually hot in june .
    the united states is sometimes mild during june , and it is cold in september .
    your least liked fruit is the grape , but my least liked is the apple .
    his favorite fruit is the orange , but my favorite is the grape .
    paris is relaxing during december , but it is usually chilly in july .
    new jersey is busy during spring , and it is never hot in march .
    our least liked fruit is the lemon , but my least liked is the grape .
    the united states is sometimes busy during january , and it is sometimes warm in november .
    
    French sentences 0 to 10:
    new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
    les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
    california est généralement calme en mars , et il est généralement chaud en juin .
    les états-unis est parfois légère en juin , et il fait froid en septembre .
    votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
    son fruit préféré est l'orange , mais mon préféré est le raisin .
    paris est relaxant en décembre , mais il est généralement froid en juillet .
    new jersey est occupé au printemps , et il est jamais chaude en mars .
    notre fruit est moins aimé le citron , mais mon moins aimé est le raisin .
    les états-unis est parfois occupé en janvier , et il est parfois chaud en novembre .


## Implement Preprocessing Function
### Text to Word Ids
As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, you'll turn `source_text` and `target_text` from words to ids.  However, you need to add the `<EOS>` word id at the end of each sentence from `target_text`.  This will help the neural network predict when the sentence should end.

You can get the `<EOS>` word id by doing:
```python
target_vocab_to_int['<EOS>']
```
You can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.


```python
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_line = [line for line in source_text.split('\n')]
    target_line = [line + ' <EOS>' for line in target_text.split('\n')]
    
    source_id_text = [[source_vocab_to_int[word] for word in line.split()] for line in source_line]
    target_id_text = [[target_vocab_to_int[word] for word in line.split()] for line in target_line]
    
    return source_id_text, target_id_text
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)
```

    Tests Passed


### Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) in [LooseVersion('1.0.0'), LooseVersion('1.0.1')], 'This project requires TensorFlow version 1.0  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.1
    Default GPU Device: /gpu:0


## Build the Neural Network
You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
- `model_inputs`
- `process_decoding_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### Input
Implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.

Return the placeholders in the following the tuple (Input, Targets, Learing Rate, Keep Probability)


```python
def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    # TODO: Implement Function
    input = tf.placeholder(tf.int32, (None, None), name='input')
    targets = tf.placeholder(tf.int32, (None, None), name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_probability = tf.placeholder(tf.float32, name='keep_prob')
    
    return input, targets, learning_rate, keep_probability

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed


### Process Decoding Input
Implement `process_decoding_input` using TensorFlow to remove the last word id from each batch in `target_data` and concat the GO ID to the begining of each batch.


```python
def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for dencoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    go = target_vocab_to_int['<GO>']
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    return tf.concat([tf.fill([batch_size, 1], go), ending], 1)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_decoding_input(process_decoding_input)
```

    Tests Passed


### Encoding
Implement `encoding_layer()` to create a Encoder RNN layer using [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn).


```python
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """
    # TODO: Implement Function
    LSTM = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    dropout = tf.contrib.rnn.DropoutWrapper(LSTM, keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([LSTM] * num_layers)
    _, RNN_state = tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32)
    
    return RNN_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)
```

    Tests Passed


### Decoding - Training
Create training logits using [`tf.contrib.seq2seq.simple_decoder_fn_train()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_train) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).  Apply the `output_fn` to the [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) outputs.


```python
def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """
    # TODO: Implement Function
    decoding_fn_training = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
    output_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell,
        decoding_fn_training,
        dec_embed_input,
        sequence_length,
        scope=decoding_scope
    )
    train_logits = output_fn(output_logits)
    return train_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)
```

    Tests Passed


### Decoding - Inference
Create inference logits using [`tf.contrib.seq2seq.simple_decoder_fn_inference()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder). 


```python
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: The maximum allowed time steps to decode
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    # TODO: Implement Function
    inference_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
        output_fn,
        encoder_state,
        dec_embeddings,
        start_of_sequence_id,
        end_of_sequence_id,
        maximum_length,
        vocab_size
    )
    
    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, decoder_fn=inference_decoder_fn, scope=decoding_scope)
    
    return inference_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)
```

    Tests Passed


### Build the Decoding Layer
Implement `decoding_layer()` to create a Decoder RNN layer.

- Create RNN cell for decoding using `rnn_size` and `num_layers`.
- Create the output fuction using [`lambda`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) to transform it's input, logits, to class logits.
- Use the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)` function to get the training logits.
- Use your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, decoding_scope, output_fn, keep_prob)` function to get the inference logits.

Note: You'll need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.


```python
def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    with tf.variable_scope('decoding') as decoding_scope:
        dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
        
    with tf.variable_scope('decoding') as decoding_scope:
        train_logits = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)

    with tf.variable_scope('decoding', reuse=True) as decoding_scope:
        infer_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], sequence_length - 1, vocab_size, decoding_scope, output_fn, keep_prob)
        
    
    return train_logits, infer_logits

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)
```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:

- Apply embedding to the input data for the encoder.
- Encode the input using your `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)`.
- Process target data using your `process_decoding_input(target_data, target_vocab_to_int, batch_size)` function.
- Apply embedding to the target data for the decoder.
- Decode the encoded input using your `decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)`.


```python
def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    enc_inputs = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
    enc_state = encoding_layer(enc_inputs, rnn_size, num_layers, keep_prob)
    
    dec_inputs = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.truncated_normal([target_vocab_size, dec_embedding_size], stddev=0.01))
    dec_embed_inputs = tf.nn.embedding_lookup(dec_embeddings, dec_inputs)
    
    train_logits, infer_logits = decoding_layer(
        dec_embed_inputs,
        dec_embeddings,
        enc_state,
        target_vocab_size,
        sequence_length,
        rnn_size,
        num_layers,
        target_vocab_to_int,
        keep_prob
    )
    
    
    return train_logits, infer_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)
```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `num_layers` to the number of layers.
- Set `encoding_embedding_size` to the size of the embedding for the encoder.
- Set `decoding_embedding_size` to the size of the embedding for the decoder.
- Set `learning_rate` to the learning rate.
- Set `keep_probability` to the Dropout keep probability


```python
# Number of Epochs
epochs = 9
# Batch Size
batch_size = 512
# RNN Size
rnn_size = 512
# Number of Layers
num_layers = 3
# Embedding Size
encoding_embedding_size = 10
decoding_embedding_size = 10
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.5
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_source_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_source_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)
    
    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
```

### Train
Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forms to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            start_time = time.time()
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})
            
            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
                
            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/269 - Train Accuracy:  0.276, Validation Accuracy:  0.341, Loss:  5.881
    Epoch   0 Batch    1/269 - Train Accuracy:  0.268, Validation Accuracy:  0.341, Loss:  5.866
    Epoch   0 Batch    2/269 - Train Accuracy:  0.297, Validation Accuracy:  0.340, Loss:  5.746
    Epoch   0 Batch    3/269 - Train Accuracy:  0.256, Validation Accuracy:  0.322, Loss:  4.794
    Epoch   0 Batch    4/269 - Train Accuracy:  0.269, Validation Accuracy:  0.342, Loss:  4.974
    Epoch   0 Batch    5/269 - Train Accuracy:  0.269, Validation Accuracy:  0.342, Loss:  4.083
    Epoch   0 Batch    6/269 - Train Accuracy:  0.308, Validation Accuracy:  0.341, Loss:  4.018
    Epoch   0 Batch    7/269 - Train Accuracy:  0.326, Validation Accuracy:  0.358, Loss:  3.907
    Epoch   0 Batch    8/269 - Train Accuracy:  0.291, Validation Accuracy:  0.354, Loss:  3.769
    Epoch   0 Batch    9/269 - Train Accuracy:  0.318, Validation Accuracy:  0.359, Loss:  3.540
    Epoch   0 Batch   10/269 - Train Accuracy:  0.288, Validation Accuracy:  0.359, Loss:  3.557
    Epoch   0 Batch   11/269 - Train Accuracy:  0.333, Validation Accuracy:  0.367, Loss:  3.397
    Epoch   0 Batch   12/269 - Train Accuracy:  0.307, Validation Accuracy:  0.368, Loss:  3.469
    Epoch   0 Batch   13/269 - Train Accuracy:  0.369, Validation Accuracy:  0.368, Loss:  3.146
    Epoch   0 Batch   14/269 - Train Accuracy:  0.336, Validation Accuracy:  0.372, Loss:  3.270
    Epoch   0 Batch   15/269 - Train Accuracy:  0.328, Validation Accuracy:  0.372, Loss:  3.238
    Epoch   0 Batch   16/269 - Train Accuracy:  0.356, Validation Accuracy:  0.385, Loss:  3.169
    Epoch   0 Batch   17/269 - Train Accuracy:  0.341, Validation Accuracy:  0.379, Loss:  3.148
    Epoch   0 Batch   18/269 - Train Accuracy:  0.324, Validation Accuracy:  0.389, Loss:  3.262
    Epoch   0 Batch   19/269 - Train Accuracy:  0.394, Validation Accuracy:  0.395, Loss:  2.962
    Epoch   0 Batch   20/269 - Train Accuracy:  0.330, Validation Accuracy:  0.391, Loss:  3.170
    Epoch   0 Batch   21/269 - Train Accuracy:  0.347, Validation Accuracy:  0.403, Loss:  3.174
    Epoch   0 Batch   22/269 - Train Accuracy:  0.381, Validation Accuracy:  0.404, Loss:  2.980
    Epoch   0 Batch   23/269 - Train Accuracy:  0.383, Validation Accuracy:  0.402, Loss:  2.940
    Epoch   0 Batch   24/269 - Train Accuracy:  0.297, Validation Accuracy:  0.365, Loss:  3.065
    Epoch   0 Batch   25/269 - Train Accuracy:  0.357, Validation Accuracy:  0.418, Loss:  3.515
    Epoch   0 Batch   26/269 - Train Accuracy:  0.418, Validation Accuracy:  0.420, Loss:  2.845
    Epoch   0 Batch   27/269 - Train Accuracy:  0.372, Validation Accuracy:  0.401, Loss:  2.902
    Epoch   0 Batch   28/269 - Train Accuracy:  0.331, Validation Accuracy:  0.402, Loss:  3.092
    Epoch   0 Batch   29/269 - Train Accuracy:  0.340, Validation Accuracy:  0.399, Loss:  3.008
    Epoch   0 Batch   30/269 - Train Accuracy:  0.368, Validation Accuracy:  0.401, Loss:  2.887
    Epoch   0 Batch   31/269 - Train Accuracy:  0.383, Validation Accuracy:  0.408, Loss:  2.867
    Epoch   0 Batch   32/269 - Train Accuracy:  0.373, Validation Accuracy:  0.410, Loss:  2.874
    Epoch   0 Batch   33/269 - Train Accuracy:  0.403, Validation Accuracy:  0.427, Loss:  2.775
    Epoch   0 Batch   34/269 - Train Accuracy:  0.400, Validation Accuracy:  0.427, Loss:  2.801
    Epoch   0 Batch   35/269 - Train Accuracy:  0.395, Validation Accuracy:  0.422, Loss:  2.766
    Epoch   0 Batch   36/269 - Train Accuracy:  0.397, Validation Accuracy:  0.425, Loss:  2.763
    Epoch   0 Batch   37/269 - Train Accuracy:  0.408, Validation Accuracy:  0.428, Loss:  2.738
    Epoch   0 Batch   38/269 - Train Accuracy:  0.406, Validation Accuracy:  0.434, Loss:  2.734
    Epoch   0 Batch   39/269 - Train Accuracy:  0.404, Validation Accuracy:  0.431, Loss:  2.691
    Epoch   0 Batch   40/269 - Train Accuracy:  0.389, Validation Accuracy:  0.444, Loss:  2.810
    Epoch   0 Batch   41/269 - Train Accuracy:  0.409, Validation Accuracy:  0.439, Loss:  2.672
    Epoch   0 Batch   42/269 - Train Accuracy:  0.428, Validation Accuracy:  0.429, Loss:  2.554
    Epoch   0 Batch   43/269 - Train Accuracy:  0.386, Validation Accuracy:  0.435, Loss:  2.741
    Epoch   0 Batch   44/269 - Train Accuracy:  0.425, Validation Accuracy:  0.446, Loss:  2.611
    Epoch   0 Batch   45/269 - Train Accuracy:  0.387, Validation Accuracy:  0.442, Loss:  2.722
    Epoch   0 Batch   46/269 - Train Accuracy:  0.394, Validation Accuracy:  0.451, Loss:  2.732
    Epoch   0 Batch   47/269 - Train Accuracy:  0.453, Validation Accuracy:  0.450, Loss:  2.433
    Epoch   0 Batch   48/269 - Train Accuracy:  0.426, Validation Accuracy:  0.448, Loss:  2.510
    Epoch   0 Batch   49/269 - Train Accuracy:  0.401, Validation Accuracy:  0.453, Loss:  2.640
    Epoch   0 Batch   50/269 - Train Accuracy:  0.408, Validation Accuracy:  0.457, Loss:  2.631
    Epoch   0 Batch   51/269 - Train Accuracy:  0.419, Validation Accuracy:  0.452, Loss:  2.515
    Epoch   0 Batch   52/269 - Train Accuracy:  0.432, Validation Accuracy:  0.460, Loss:  2.476
    Epoch   0 Batch   53/269 - Train Accuracy:  0.394, Validation Accuracy:  0.447, Loss:  2.581
    Epoch   0 Batch   54/269 - Train Accuracy:  0.406, Validation Accuracy:  0.454, Loss:  2.611
    Epoch   0 Batch   55/269 - Train Accuracy:  0.408, Validation Accuracy:  0.436, Loss:  2.518
    Epoch   0 Batch   56/269 - Train Accuracy:  0.438, Validation Accuracy:  0.461, Loss:  2.474
    Epoch   0 Batch   57/269 - Train Accuracy:  0.436, Validation Accuracy:  0.458, Loss:  2.397
    Epoch   0 Batch   58/269 - Train Accuracy:  0.431, Validation Accuracy:  0.457, Loss:  2.439
    Epoch   0 Batch   59/269 - Train Accuracy:  0.434, Validation Accuracy:  0.461, Loss:  2.368
    Epoch   0 Batch   60/269 - Train Accuracy:  0.457, Validation Accuracy:  0.469, Loss:  2.292
    Epoch   0 Batch   61/269 - Train Accuracy:  0.465, Validation Accuracy:  0.464, Loss:  2.281
    Epoch   0 Batch   62/269 - Train Accuracy:  0.459, Validation Accuracy:  0.464, Loss:  2.252
    Epoch   0 Batch   63/269 - Train Accuracy:  0.436, Validation Accuracy:  0.463, Loss:  2.314
    Epoch   0 Batch   64/269 - Train Accuracy:  0.435, Validation Accuracy:  0.470, Loss:  2.338
    Epoch   0 Batch   65/269 - Train Accuracy:  0.436, Validation Accuracy:  0.463, Loss:  2.285
    Epoch   0 Batch   66/269 - Train Accuracy:  0.458, Validation Accuracy:  0.472, Loss:  2.234
    Epoch   0 Batch   67/269 - Train Accuracy:  0.439, Validation Accuracy:  0.473, Loss:  2.318
    Epoch   0 Batch   68/269 - Train Accuracy:  0.437, Validation Accuracy:  0.472, Loss:  2.282
    Epoch   0 Batch   69/269 - Train Accuracy:  0.419, Validation Accuracy:  0.477, Loss:  2.396
    Epoch   0 Batch   70/269 - Train Accuracy:  0.456, Validation Accuracy:  0.473, Loss:  2.226
    Epoch   0 Batch   71/269 - Train Accuracy:  0.420, Validation Accuracy:  0.471, Loss:  2.329
    Epoch   0 Batch   72/269 - Train Accuracy:  0.464, Validation Accuracy:  0.473, Loss:  2.153
    Epoch   0 Batch   73/269 - Train Accuracy:  0.447, Validation Accuracy:  0.471, Loss:  2.230
    Epoch   0 Batch   74/269 - Train Accuracy:  0.405, Validation Accuracy:  0.455, Loss:  2.315
    Epoch   0 Batch   75/269 - Train Accuracy:  0.442, Validation Accuracy:  0.468, Loss:  2.231
    Epoch   0 Batch   76/269 - Train Accuracy:  0.430, Validation Accuracy:  0.474, Loss:  2.245
    Epoch   0 Batch   77/269 - Train Accuracy:  0.462, Validation Accuracy:  0.479, Loss:  2.182
    Epoch   0 Batch   78/269 - Train Accuracy:  0.439, Validation Accuracy:  0.471, Loss:  2.191
    Epoch   0 Batch   79/269 - Train Accuracy:  0.428, Validation Accuracy:  0.460, Loss:  2.172
    Epoch   0 Batch   80/269 - Train Accuracy:  0.447, Validation Accuracy:  0.464, Loss:  2.138
    Epoch   0 Batch   81/269 - Train Accuracy:  0.422, Validation Accuracy:  0.449, Loss:  2.303
    Epoch   0 Batch   82/269 - Train Accuracy:  0.467, Validation Accuracy:  0.483, Loss:  2.190
    Epoch   0 Batch   83/269 - Train Accuracy:  0.464, Validation Accuracy:  0.481, Loss:  2.089
    Epoch   0 Batch   84/269 - Train Accuracy:  0.416, Validation Accuracy:  0.443, Loss:  2.171
    Epoch   0 Batch   85/269 - Train Accuracy:  0.444, Validation Accuracy:  0.473, Loss:  2.261
    Epoch   0 Batch   86/269 - Train Accuracy:  0.423, Validation Accuracy:  0.457, Loss:  2.124
    Epoch   0 Batch   87/269 - Train Accuracy:  0.397, Validation Accuracy:  0.472, Loss:  2.300
    Epoch   0 Batch   88/269 - Train Accuracy:  0.453, Validation Accuracy:  0.472, Loss:  2.120
    Epoch   0 Batch   89/269 - Train Accuracy:  0.468, Validation Accuracy:  0.484, Loss:  2.087
    Epoch   0 Batch   90/269 - Train Accuracy:  0.411, Validation Accuracy:  0.476, Loss:  2.259
    Epoch   0 Batch   91/269 - Train Accuracy:  0.438, Validation Accuracy:  0.472, Loss:  2.073
    Epoch   0 Batch   92/269 - Train Accuracy:  0.452, Validation Accuracy:  0.483, Loss:  2.089
    Epoch   0 Batch   93/269 - Train Accuracy:  0.479, Validation Accuracy:  0.489, Loss:  2.009
    Epoch   0 Batch   94/269 - Train Accuracy:  0.437, Validation Accuracy:  0.470, Loss:  2.058
    Epoch   0 Batch   95/269 - Train Accuracy:  0.466, Validation Accuracy:  0.485, Loss:  2.058
    Epoch   0 Batch   96/269 - Train Accuracy:  0.461, Validation Accuracy:  0.487, Loss:  2.038
    Epoch   0 Batch   97/269 - Train Accuracy:  0.447, Validation Accuracy:  0.474, Loss:  2.039
    Epoch   0 Batch   98/269 - Train Accuracy:  0.480, Validation Accuracy:  0.489, Loss:  2.020
    Epoch   0 Batch   99/269 - Train Accuracy:  0.431, Validation Accuracy:  0.490, Loss:  2.135
    Epoch   0 Batch  100/269 - Train Accuracy:  0.466, Validation Accuracy:  0.475, Loss:  1.958
    Epoch   0 Batch  101/269 - Train Accuracy:  0.441, Validation Accuracy:  0.493, Loss:  2.127
    Epoch   0 Batch  102/269 - Train Accuracy:  0.461, Validation Accuracy:  0.485, Loss:  2.006
    Epoch   0 Batch  103/269 - Train Accuracy:  0.448, Validation Accuracy:  0.476, Loss:  1.984
    Epoch   0 Batch  104/269 - Train Accuracy:  0.465, Validation Accuracy:  0.495, Loss:  2.012
    Epoch   0 Batch  105/269 - Train Accuracy:  0.445, Validation Accuracy:  0.478, Loss:  2.039
    Epoch   0 Batch  106/269 - Train Accuracy:  0.445, Validation Accuracy:  0.484, Loss:  2.004
    Epoch   0 Batch  107/269 - Train Accuracy:  0.427, Validation Accuracy:  0.490, Loss:  2.070
    Epoch   0 Batch  108/269 - Train Accuracy:  0.451, Validation Accuracy:  0.484, Loss:  1.985
    Epoch   0 Batch  109/269 - Train Accuracy:  0.463, Validation Accuracy:  0.500, Loss:  1.992
    Epoch   0 Batch  110/269 - Train Accuracy:  0.467, Validation Accuracy:  0.493, Loss:  1.949
    Epoch   0 Batch  111/269 - Train Accuracy:  0.422, Validation Accuracy:  0.486, Loss:  2.084
    Epoch   0 Batch  112/269 - Train Accuracy:  0.471, Validation Accuracy:  0.498, Loss:  1.946
    Epoch   0 Batch  113/269 - Train Accuracy:  0.493, Validation Accuracy:  0.496, Loss:  1.845
    Epoch   0 Batch  114/269 - Train Accuracy:  0.454, Validation Accuracy:  0.488, Loss:  1.906
    Epoch   0 Batch  115/269 - Train Accuracy:  0.441, Validation Accuracy:  0.489, Loss:  1.988
    Epoch   0 Batch  116/269 - Train Accuracy:  0.472, Validation Accuracy:  0.490, Loss:  1.963
    Epoch   0 Batch  117/269 - Train Accuracy:  0.472, Validation Accuracy:  0.504, Loss:  1.921
    Epoch   0 Batch  118/269 - Train Accuracy:  0.499, Validation Accuracy:  0.507, Loss:  1.830
    Epoch   0 Batch  119/269 - Train Accuracy:  0.428, Validation Accuracy:  0.475, Loss:  2.015
    Epoch   0 Batch  120/269 - Train Accuracy:  0.450, Validation Accuracy:  0.493, Loss:  2.068
    Epoch   0 Batch  121/269 - Train Accuracy:  0.473, Validation Accuracy:  0.507, Loss:  1.952
    Epoch   0 Batch  122/269 - Train Accuracy:  0.461, Validation Accuracy:  0.485, Loss:  1.854
    Epoch   0 Batch  123/269 - Train Accuracy:  0.435, Validation Accuracy:  0.490, Loss:  2.040
    Epoch   0 Batch  124/269 - Train Accuracy:  0.452, Validation Accuracy:  0.481, Loss:  1.999
    Epoch   0 Batch  125/269 - Train Accuracy:  0.456, Validation Accuracy:  0.488, Loss:  1.873
    Epoch   0 Batch  126/269 - Train Accuracy:  0.483, Validation Accuracy:  0.501, Loss:  1.859
    Epoch   0 Batch  127/269 - Train Accuracy:  0.425, Validation Accuracy:  0.478, Loss:  2.039
    Epoch   0 Batch  128/269 - Train Accuracy:  0.464, Validation Accuracy:  0.479, Loss:  1.868
    Epoch   0 Batch  129/269 - Train Accuracy:  0.462, Validation Accuracy:  0.492, Loss:  1.886
    Epoch   0 Batch  130/269 - Train Accuracy:  0.438, Validation Accuracy:  0.501, Loss:  2.033
    Epoch   0 Batch  131/269 - Train Accuracy:  0.463, Validation Accuracy:  0.505, Loss:  1.931
    Epoch   0 Batch  132/269 - Train Accuracy:  0.478, Validation Accuracy:  0.511, Loss:  1.854
    Epoch   0 Batch  133/269 - Train Accuracy:  0.475, Validation Accuracy:  0.506, Loss:  1.827
    Epoch   0 Batch  134/269 - Train Accuracy:  0.443, Validation Accuracy:  0.497, Loss:  1.892
    Epoch   0 Batch  135/269 - Train Accuracy:  0.442, Validation Accuracy:  0.498, Loss:  1.958
    Epoch   0 Batch  136/269 - Train Accuracy:  0.453, Validation Accuracy:  0.510, Loss:  1.914
    Epoch   0 Batch  137/269 - Train Accuracy:  0.440, Validation Accuracy:  0.487, Loss:  1.909
    Epoch   0 Batch  138/269 - Train Accuracy:  0.452, Validation Accuracy:  0.493, Loss:  1.863
    Epoch   0 Batch  139/269 - Train Accuracy:  0.474, Validation Accuracy:  0.489, Loss:  1.800
    Epoch   0 Batch  140/269 - Train Accuracy:  0.466, Validation Accuracy:  0.486, Loss:  1.799
    Epoch   0 Batch  141/269 - Train Accuracy:  0.472, Validation Accuracy:  0.503, Loss:  1.832
    Epoch   0 Batch  142/269 - Train Accuracy:  0.477, Validation Accuracy:  0.497, Loss:  1.797
    Epoch   0 Batch  143/269 - Train Accuracy:  0.454, Validation Accuracy:  0.489, Loss:  1.782
    Epoch   0 Batch  144/269 - Train Accuracy:  0.467, Validation Accuracy:  0.485, Loss:  1.777
    Epoch   0 Batch  145/269 - Train Accuracy:  0.461, Validation Accuracy:  0.492, Loss:  1.785
    Epoch   0 Batch  146/269 - Train Accuracy:  0.478, Validation Accuracy:  0.505, Loss:  1.749
    Epoch   0 Batch  147/269 - Train Accuracy:  0.506, Validation Accuracy:  0.505, Loss:  1.696
    Epoch   0 Batch  148/269 - Train Accuracy:  0.466, Validation Accuracy:  0.503, Loss:  1.799
    Epoch   0 Batch  149/269 - Train Accuracy:  0.476, Validation Accuracy:  0.498, Loss:  1.750
    Epoch   0 Batch  150/269 - Train Accuracy:  0.474, Validation Accuracy:  0.495, Loss:  1.767
    Epoch   0 Batch  151/269 - Train Accuracy:  0.510, Validation Accuracy:  0.498, Loss:  1.647
    Epoch   0 Batch  152/269 - Train Accuracy:  0.462, Validation Accuracy:  0.498, Loss:  1.743
    Epoch   0 Batch  153/269 - Train Accuracy:  0.473, Validation Accuracy:  0.493, Loss:  1.757
    Epoch   0 Batch  154/269 - Train Accuracy:  0.423, Validation Accuracy:  0.491, Loss:  1.838
    Epoch   0 Batch  155/269 - Train Accuracy:  0.497, Validation Accuracy:  0.493, Loss:  1.627
    Epoch   0 Batch  156/269 - Train Accuracy:  0.476, Validation Accuracy:  0.509, Loss:  1.784
    Epoch   0 Batch  157/269 - Train Accuracy:  0.469, Validation Accuracy:  0.497, Loss:  1.770
    Epoch   0 Batch  158/269 - Train Accuracy:  0.466, Validation Accuracy:  0.495, Loss:  1.686
    Epoch   0 Batch  159/269 - Train Accuracy:  0.473, Validation Accuracy:  0.494, Loss:  1.726
    Epoch   0 Batch  160/269 - Train Accuracy:  0.467, Validation Accuracy:  0.496, Loss:  1.775
    Epoch   0 Batch  161/269 - Train Accuracy:  0.469, Validation Accuracy:  0.503, Loss:  1.734
    Epoch   0 Batch  162/269 - Train Accuracy:  0.469, Validation Accuracy:  0.494, Loss:  1.683
    Epoch   0 Batch  163/269 - Train Accuracy:  0.480, Validation Accuracy:  0.503, Loss:  1.726
    Epoch   0 Batch  164/269 - Train Accuracy:  0.485, Validation Accuracy:  0.507, Loss:  1.721
    Epoch   0 Batch  165/269 - Train Accuracy:  0.462, Validation Accuracy:  0.505, Loss:  1.729
    Epoch   0 Batch  166/269 - Train Accuracy:  0.504, Validation Accuracy:  0.502, Loss:  1.584
    Epoch   0 Batch  167/269 - Train Accuracy:  0.490, Validation Accuracy:  0.505, Loss:  1.688
    Epoch   0 Batch  168/269 - Train Accuracy:  0.474, Validation Accuracy:  0.495, Loss:  1.667
    Epoch   0 Batch  169/269 - Train Accuracy:  0.471, Validation Accuracy:  0.502, Loss:  1.658
    Epoch   0 Batch  170/269 - Train Accuracy:  0.487, Validation Accuracy:  0.512, Loss:  1.655
    Epoch   0 Batch  171/269 - Train Accuracy:  0.475, Validation Accuracy:  0.510, Loss:  1.679
    Epoch   0 Batch  172/269 - Train Accuracy:  0.477, Validation Accuracy:  0.505, Loss:  1.679
    Epoch   0 Batch  173/269 - Train Accuracy:  0.486, Validation Accuracy:  0.510, Loss:  1.681
    Epoch   0 Batch  174/269 - Train Accuracy:  0.485, Validation Accuracy:  0.512, Loss:  1.624
    Epoch   0 Batch  175/269 - Train Accuracy:  0.466, Validation Accuracy:  0.493, Loss:  1.665
    Epoch   0 Batch  176/269 - Train Accuracy:  0.443, Validation Accuracy:  0.493, Loss:  1.741
    Epoch   0 Batch  177/269 - Train Accuracy:  0.498, Validation Accuracy:  0.507, Loss:  1.617
    Epoch   0 Batch  178/269 - Train Accuracy:  0.437, Validation Accuracy:  0.488, Loss:  1.694
    Epoch   0 Batch  179/269 - Train Accuracy:  0.492, Validation Accuracy:  0.508, Loss:  1.653
    Epoch   0 Batch  180/269 - Train Accuracy:  0.481, Validation Accuracy:  0.503, Loss:  1.655
    Epoch   0 Batch  181/269 - Train Accuracy:  0.477, Validation Accuracy:  0.503, Loss:  1.568
    Epoch   0 Batch  182/269 - Train Accuracy:  0.471, Validation Accuracy:  0.492, Loss:  1.635
    Epoch   0 Batch  183/269 - Train Accuracy:  0.553, Validation Accuracy:  0.509, Loss:  1.394
    Epoch   0 Batch  184/269 - Train Accuracy:  0.439, Validation Accuracy:  0.483, Loss:  1.642
    Epoch   0 Batch  185/269 - Train Accuracy:  0.499, Validation Accuracy:  0.511, Loss:  1.658
    Epoch   0 Batch  186/269 - Train Accuracy:  0.458, Validation Accuracy:  0.510, Loss:  1.667
    Epoch   0 Batch  187/269 - Train Accuracy:  0.479, Validation Accuracy:  0.502, Loss:  1.530
    Epoch   0 Batch  188/269 - Train Accuracy:  0.493, Validation Accuracy:  0.513, Loss:  1.565
    Epoch   0 Batch  189/269 - Train Accuracy:  0.488, Validation Accuracy:  0.512, Loss:  1.525
    Epoch   0 Batch  190/269 - Train Accuracy:  0.467, Validation Accuracy:  0.498, Loss:  1.531
    Epoch   0 Batch  191/269 - Train Accuracy:  0.476, Validation Accuracy:  0.506, Loss:  1.535
    Epoch   0 Batch  192/269 - Train Accuracy:  0.476, Validation Accuracy:  0.509, Loss:  1.536
    Epoch   0 Batch  193/269 - Train Accuracy:  0.481, Validation Accuracy:  0.504, Loss:  1.519
    Epoch   0 Batch  194/269 - Train Accuracy:  0.499, Validation Accuracy:  0.511, Loss:  1.538
    Epoch   0 Batch  195/269 - Train Accuracy:  0.459, Validation Accuracy:  0.490, Loss:  1.536
    Epoch   0 Batch  196/269 - Train Accuracy:  0.480, Validation Accuracy:  0.506, Loss:  1.504
    Epoch   0 Batch  197/269 - Train Accuracy:  0.456, Validation Accuracy:  0.508, Loss:  1.536
    Epoch   0 Batch  198/269 - Train Accuracy:  0.474, Validation Accuracy:  0.518, Loss:  1.607
    Epoch   0 Batch  199/269 - Train Accuracy:  0.482, Validation Accuracy:  0.517, Loss:  1.511
    Epoch   0 Batch  200/269 - Train Accuracy:  0.474, Validation Accuracy:  0.513, Loss:  1.539
    Epoch   0 Batch  201/269 - Train Accuracy:  0.490, Validation Accuracy:  0.510, Loss:  1.523
    Epoch   0 Batch  202/269 - Train Accuracy:  0.496, Validation Accuracy:  0.514, Loss:  1.473
    Epoch   0 Batch  203/269 - Train Accuracy:  0.476, Validation Accuracy:  0.518, Loss:  1.530
    Epoch   0 Batch  204/269 - Train Accuracy:  0.462, Validation Accuracy:  0.511, Loss:  1.534
    Epoch   0 Batch  205/269 - Train Accuracy:  0.473, Validation Accuracy:  0.511, Loss:  1.430
    Epoch   0 Batch  206/269 - Train Accuracy:  0.446, Validation Accuracy:  0.509, Loss:  1.533
    Epoch   0 Batch  207/269 - Train Accuracy:  0.486, Validation Accuracy:  0.494, Loss:  1.408
    Epoch   0 Batch  208/269 - Train Accuracy:  0.445, Validation Accuracy:  0.495, Loss:  1.561
    Epoch   0 Batch  209/269 - Train Accuracy:  0.474, Validation Accuracy:  0.516, Loss:  1.454
    Epoch   0 Batch  210/269 - Train Accuracy:  0.474, Validation Accuracy:  0.492, Loss:  1.438
    Epoch   0 Batch  211/269 - Train Accuracy:  0.480, Validation Accuracy:  0.494, Loss:  1.417
    Epoch   0 Batch  212/269 - Train Accuracy:  0.495, Validation Accuracy:  0.509, Loss:  1.378
    Epoch   0 Batch  213/269 - Train Accuracy:  0.482, Validation Accuracy:  0.499, Loss:  1.412
    Epoch   0 Batch  214/269 - Train Accuracy:  0.492, Validation Accuracy:  0.511, Loss:  1.400
    Epoch   0 Batch  215/269 - Train Accuracy:  0.516, Validation Accuracy:  0.517, Loss:  1.306
    Epoch   0 Batch  216/269 - Train Accuracy:  0.449, Validation Accuracy:  0.504, Loss:  1.499
    Epoch   0 Batch  217/269 - Train Accuracy:  0.456, Validation Accuracy:  0.511, Loss:  1.473
    Epoch   0 Batch  218/269 - Train Accuracy:  0.463, Validation Accuracy:  0.510, Loss:  1.427
    Epoch   0 Batch  219/269 - Train Accuracy:  0.458, Validation Accuracy:  0.494, Loss:  1.433
    Epoch   0 Batch  220/269 - Train Accuracy:  0.498, Validation Accuracy:  0.499, Loss:  1.327
    Epoch   0 Batch  221/269 - Train Accuracy:  0.506, Validation Accuracy:  0.515, Loss:  1.356
    Epoch   0 Batch  222/269 - Train Accuracy:  0.488, Validation Accuracy:  0.486, Loss:  1.339
    Epoch   0 Batch  223/269 - Train Accuracy:  0.487, Validation Accuracy:  0.508, Loss:  1.365
    Epoch   0 Batch  224/269 - Train Accuracy:  0.486, Validation Accuracy:  0.509, Loss:  1.382
    Epoch   0 Batch  225/269 - Train Accuracy:  0.464, Validation Accuracy:  0.510, Loss:  1.388
    Epoch   0 Batch  226/269 - Train Accuracy:  0.461, Validation Accuracy:  0.503, Loss:  1.380
    Epoch   0 Batch  227/269 - Train Accuracy:  0.558, Validation Accuracy:  0.509, Loss:  1.201
    Epoch   0 Batch  228/269 - Train Accuracy:  0.480, Validation Accuracy:  0.512, Loss:  1.357
    Epoch   0 Batch  229/269 - Train Accuracy:  0.479, Validation Accuracy:  0.501, Loss:  1.362
    Epoch   0 Batch  230/269 - Train Accuracy:  0.487, Validation Accuracy:  0.511, Loss:  1.365
    Epoch   0 Batch  231/269 - Train Accuracy:  0.458, Validation Accuracy:  0.495, Loss:  1.403
    Epoch   0 Batch  232/269 - Train Accuracy:  0.448, Validation Accuracy:  0.507, Loss:  1.389
    Epoch   0 Batch  233/269 - Train Accuracy:  0.474, Validation Accuracy:  0.504, Loss:  1.340
    Epoch   0 Batch  234/269 - Train Accuracy:  0.480, Validation Accuracy:  0.500, Loss:  1.313
    Epoch   0 Batch  235/269 - Train Accuracy:  0.496, Validation Accuracy:  0.511, Loss:  1.306
    Epoch   0 Batch  236/269 - Train Accuracy:  0.477, Validation Accuracy:  0.494, Loss:  1.323
    Epoch   0 Batch  237/269 - Train Accuracy:  0.493, Validation Accuracy:  0.512, Loss:  1.336
    Epoch   0 Batch  238/269 - Train Accuracy:  0.489, Validation Accuracy:  0.513, Loss:  1.311
    Epoch   0 Batch  239/269 - Train Accuracy:  0.497, Validation Accuracy:  0.508, Loss:  1.283
    Epoch   0 Batch  240/269 - Train Accuracy:  0.522, Validation Accuracy:  0.512, Loss:  1.229
    Epoch   0 Batch  241/269 - Train Accuracy:  0.498, Validation Accuracy:  0.513, Loss:  1.299
    Epoch   0 Batch  242/269 - Train Accuracy:  0.480, Validation Accuracy:  0.508, Loss:  1.265
    Epoch   0 Batch  243/269 - Train Accuracy:  0.504, Validation Accuracy:  0.511, Loss:  1.276
    Epoch   0 Batch  244/269 - Train Accuracy:  0.495, Validation Accuracy:  0.508, Loss:  1.325
    Epoch   0 Batch  245/269 - Train Accuracy:  0.475, Validation Accuracy:  0.517, Loss:  1.361
    Epoch   0 Batch  246/269 - Train Accuracy:  0.484, Validation Accuracy:  0.516, Loss:  1.276
    Epoch   0 Batch  247/269 - Train Accuracy:  0.485, Validation Accuracy:  0.515, Loss:  1.306
    Epoch   0 Batch  248/269 - Train Accuracy:  0.476, Validation Accuracy:  0.512, Loss:  1.255
    Epoch   0 Batch  249/269 - Train Accuracy:  0.517, Validation Accuracy:  0.514, Loss:  1.218
    Epoch   0 Batch  250/269 - Train Accuracy:  0.476, Validation Accuracy:  0.514, Loss:  1.278
    Epoch   0 Batch  251/269 - Train Accuracy:  0.500, Validation Accuracy:  0.520, Loss:  1.244
    Epoch   0 Batch  252/269 - Train Accuracy:  0.478, Validation Accuracy:  0.522, Loss:  1.253
    Epoch   0 Batch  253/269 - Train Accuracy:  0.491, Validation Accuracy:  0.520, Loss:  1.249
    Epoch   0 Batch  254/269 - Train Accuracy:  0.495, Validation Accuracy:  0.519, Loss:  1.234
    Epoch   0 Batch  255/269 - Train Accuracy:  0.526, Validation Accuracy:  0.518, Loss:  1.189
    Epoch   0 Batch  256/269 - Train Accuracy:  0.489, Validation Accuracy:  0.518, Loss:  1.256
    Epoch   0 Batch  257/269 - Train Accuracy:  0.476, Validation Accuracy:  0.503, Loss:  1.304
    Epoch   0 Batch  258/269 - Train Accuracy:  0.489, Validation Accuracy:  0.517, Loss:  1.263
    Epoch   0 Batch  259/269 - Train Accuracy:  0.505, Validation Accuracy:  0.518, Loss:  1.234
    Epoch   0 Batch  260/269 - Train Accuracy:  0.480, Validation Accuracy:  0.516, Loss:  1.278
    Epoch   0 Batch  261/269 - Train Accuracy:  0.477, Validation Accuracy:  0.531, Loss:  1.262
    Epoch   0 Batch  262/269 - Train Accuracy:  0.509, Validation Accuracy:  0.532, Loss:  1.221
    Epoch   0 Batch  263/269 - Train Accuracy:  0.492, Validation Accuracy:  0.521, Loss:  1.252
    Epoch   0 Batch  264/269 - Train Accuracy:  0.484, Validation Accuracy:  0.518, Loss:  1.301
    Epoch   0 Batch  265/269 - Train Accuracy:  0.457, Validation Accuracy:  0.509, Loss:  1.277
    Epoch   0 Batch  266/269 - Train Accuracy:  0.513, Validation Accuracy:  0.519, Loss:  1.231
    Epoch   0 Batch  267/269 - Train Accuracy:  0.497, Validation Accuracy:  0.522, Loss:  1.248
    Epoch   1 Batch    0/269 - Train Accuracy:  0.478, Validation Accuracy:  0.514, Loss:  1.268
    Epoch   1 Batch    1/269 - Train Accuracy:  0.473, Validation Accuracy:  0.510, Loss:  1.221
    Epoch   1 Batch    2/269 - Train Accuracy:  0.450, Validation Accuracy:  0.499, Loss:  1.278
    Epoch   1 Batch    3/269 - Train Accuracy:  0.466, Validation Accuracy:  0.518, Loss:  1.326
    Epoch   1 Batch    4/269 - Train Accuracy:  0.481, Validation Accuracy:  0.510, Loss:  1.299
    Epoch   1 Batch    5/269 - Train Accuracy:  0.455, Validation Accuracy:  0.517, Loss:  1.361
    Epoch   1 Batch    6/269 - Train Accuracy:  0.491, Validation Accuracy:  0.506, Loss:  1.244
    Epoch   1 Batch    7/269 - Train Accuracy:  0.497, Validation Accuracy:  0.510, Loss:  1.219
    Epoch   1 Batch    8/269 - Train Accuracy:  0.449, Validation Accuracy:  0.504, Loss:  1.285
    Epoch   1 Batch    9/269 - Train Accuracy:  0.477, Validation Accuracy:  0.516, Loss:  1.308
    Epoch   1 Batch   10/269 - Train Accuracy:  0.493, Validation Accuracy:  0.527, Loss:  1.337
    Epoch   1 Batch   11/269 - Train Accuracy:  0.493, Validation Accuracy:  0.522, Loss:  1.214
    Epoch   1 Batch   12/269 - Train Accuracy:  0.471, Validation Accuracy:  0.516, Loss:  1.256
    Epoch   1 Batch   13/269 - Train Accuracy:  0.538, Validation Accuracy:  0.519, Loss:  1.126
    Epoch   1 Batch   14/269 - Train Accuracy:  0.473, Validation Accuracy:  0.506, Loss:  1.181
    Epoch   1 Batch   15/269 - Train Accuracy:  0.471, Validation Accuracy:  0.507, Loss:  1.173
    Epoch   1 Batch   16/269 - Train Accuracy:  0.505, Validation Accuracy:  0.510, Loss:  1.200
    Epoch   1 Batch   17/269 - Train Accuracy:  0.492, Validation Accuracy:  0.516, Loss:  1.137
    Epoch   1 Batch   18/269 - Train Accuracy:  0.487, Validation Accuracy:  0.518, Loss:  1.207
    Epoch   1 Batch   19/269 - Train Accuracy:  0.522, Validation Accuracy:  0.516, Loss:  1.116
    Epoch   1 Batch   20/269 - Train Accuracy:  0.479, Validation Accuracy:  0.515, Loss:  1.204
    Epoch   1 Batch   21/269 - Train Accuracy:  0.467, Validation Accuracy:  0.521, Loss:  1.229
    Epoch   1 Batch   22/269 - Train Accuracy:  0.500, Validation Accuracy:  0.516, Loss:  1.142
    Epoch   1 Batch   23/269 - Train Accuracy:  0.492, Validation Accuracy:  0.506, Loss:  1.159
    Epoch   1 Batch   24/269 - Train Accuracy:  0.496, Validation Accuracy:  0.529, Loss:  1.187
    Epoch   1 Batch   25/269 - Train Accuracy:  0.480, Validation Accuracy:  0.526, Loss:  1.198
    Epoch   1 Batch   26/269 - Train Accuracy:  0.515, Validation Accuracy:  0.519, Loss:  1.081
    Epoch   1 Batch   27/269 - Train Accuracy:  0.496, Validation Accuracy:  0.519, Loss:  1.145
    Epoch   1 Batch   28/269 - Train Accuracy:  0.459, Validation Accuracy:  0.512, Loss:  1.205
    Epoch   1 Batch   29/269 - Train Accuracy:  0.462, Validation Accuracy:  0.507, Loss:  1.118
    Epoch   1 Batch   30/269 - Train Accuracy:  0.502, Validation Accuracy:  0.502, Loss:  1.100
    Epoch   1 Batch   31/269 - Train Accuracy:  0.508, Validation Accuracy:  0.512, Loss:  1.105
    Epoch   1 Batch   32/269 - Train Accuracy:  0.501, Validation Accuracy:  0.513, Loss:  1.108
    Epoch   1 Batch   33/269 - Train Accuracy:  0.486, Validation Accuracy:  0.510, Loss:  1.106
    Epoch   1 Batch   34/269 - Train Accuracy:  0.477, Validation Accuracy:  0.487, Loss:  1.199
    Epoch   1 Batch   35/269 - Train Accuracy:  0.503, Validation Accuracy:  0.512, Loss:  1.221
    Epoch   1 Batch   36/269 - Train Accuracy:  0.513, Validation Accuracy:  0.526, Loss:  1.150
    Epoch   1 Batch   37/269 - Train Accuracy:  0.513, Validation Accuracy:  0.524, Loss:  1.111
    Epoch   1 Batch   38/269 - Train Accuracy:  0.491, Validation Accuracy:  0.526, Loss:  1.139
    Epoch   1 Batch   39/269 - Train Accuracy:  0.498, Validation Accuracy:  0.517, Loss:  1.118
    Epoch   1 Batch   40/269 - Train Accuracy:  0.471, Validation Accuracy:  0.515, Loss:  1.168
    Epoch   1 Batch   41/269 - Train Accuracy:  0.494, Validation Accuracy:  0.513, Loss:  1.130
    Epoch   1 Batch   42/269 - Train Accuracy:  0.515, Validation Accuracy:  0.517, Loss:  1.055
    Epoch   1 Batch   43/269 - Train Accuracy:  0.480, Validation Accuracy:  0.518, Loss:  1.152
    Epoch   1 Batch   44/269 - Train Accuracy:  0.503, Validation Accuracy:  0.514, Loss:  1.112
    Epoch   1 Batch   45/269 - Train Accuracy:  0.462, Validation Accuracy:  0.514, Loss:  1.154
    Epoch   1 Batch   46/269 - Train Accuracy:  0.470, Validation Accuracy:  0.516, Loss:  1.119
    Epoch   1 Batch   47/269 - Train Accuracy:  0.525, Validation Accuracy:  0.518, Loss:  1.015
    Epoch   1 Batch   48/269 - Train Accuracy:  0.498, Validation Accuracy:  0.514, Loss:  1.046
    Epoch   1 Batch   49/269 - Train Accuracy:  0.478, Validation Accuracy:  0.518, Loss:  1.108
    Epoch   1 Batch   50/269 - Train Accuracy:  0.486, Validation Accuracy:  0.529, Loss:  1.130
    Epoch   1 Batch   51/269 - Train Accuracy:  0.493, Validation Accuracy:  0.526, Loss:  1.101
    Epoch   1 Batch   52/269 - Train Accuracy:  0.507, Validation Accuracy:  0.527, Loss:  1.023
    Epoch   1 Batch   53/269 - Train Accuracy:  0.469, Validation Accuracy:  0.521, Loss:  1.125
    Epoch   1 Batch   54/269 - Train Accuracy:  0.483, Validation Accuracy:  0.520, Loss:  1.106
    Epoch   1 Batch   55/269 - Train Accuracy:  0.497, Validation Accuracy:  0.520, Loss:  1.063
    Epoch   1 Batch   56/269 - Train Accuracy:  0.512, Validation Accuracy:  0.516, Loss:  1.059
    Epoch   1 Batch   57/269 - Train Accuracy:  0.499, Validation Accuracy:  0.510, Loss:  1.071
    Epoch   1 Batch   58/269 - Train Accuracy:  0.512, Validation Accuracy:  0.518, Loss:  1.048
    Epoch   1 Batch   59/269 - Train Accuracy:  0.503, Validation Accuracy:  0.514, Loss:  1.024
    Epoch   1 Batch   60/269 - Train Accuracy:  0.501, Validation Accuracy:  0.508, Loss:  0.993
    Epoch   1 Batch   61/269 - Train Accuracy:  0.516, Validation Accuracy:  0.512, Loss:  0.979
    Epoch   1 Batch   62/269 - Train Accuracy:  0.515, Validation Accuracy:  0.522, Loss:  1.011
    Epoch   1 Batch   63/269 - Train Accuracy:  0.485, Validation Accuracy:  0.528, Loss:  1.069
    Epoch   1 Batch   64/269 - Train Accuracy:  0.505, Validation Accuracy:  0.534, Loss:  1.022
    Epoch   1 Batch   65/269 - Train Accuracy:  0.509, Validation Accuracy:  0.526, Loss:  1.029
    Epoch   1 Batch   66/269 - Train Accuracy:  0.500, Validation Accuracy:  0.517, Loss:  0.982
    Epoch   1 Batch   67/269 - Train Accuracy:  0.501, Validation Accuracy:  0.517, Loss:  1.047
    Epoch   1 Batch   68/269 - Train Accuracy:  0.489, Validation Accuracy:  0.518, Loss:  1.038
    Epoch   1 Batch   69/269 - Train Accuracy:  0.473, Validation Accuracy:  0.519, Loss:  1.136
    Epoch   1 Batch   70/269 - Train Accuracy:  0.517, Validation Accuracy:  0.527, Loss:  1.020
    Epoch   1 Batch   71/269 - Train Accuracy:  0.486, Validation Accuracy:  0.527, Loss:  1.035
    Epoch   1 Batch   72/269 - Train Accuracy:  0.531, Validation Accuracy:  0.532, Loss:  0.985
    Epoch   1 Batch   73/269 - Train Accuracy:  0.503, Validation Accuracy:  0.528, Loss:  1.053
    Epoch   1 Batch   74/269 - Train Accuracy:  0.483, Validation Accuracy:  0.529, Loss:  1.020
    Epoch   1 Batch   75/269 - Train Accuracy:  0.497, Validation Accuracy:  0.529, Loss:  1.008
    Epoch   1 Batch   76/269 - Train Accuracy:  0.493, Validation Accuracy:  0.535, Loss:  1.014
    Epoch   1 Batch   77/269 - Train Accuracy:  0.520, Validation Accuracy:  0.524, Loss:  1.003
    Epoch   1 Batch   78/269 - Train Accuracy:  0.505, Validation Accuracy:  0.513, Loss:  0.989
    Epoch   1 Batch   79/269 - Train Accuracy:  0.502, Validation Accuracy:  0.521, Loss:  0.986
    Epoch   1 Batch   80/269 - Train Accuracy:  0.502, Validation Accuracy:  0.524, Loss:  0.977
    Epoch   1 Batch   81/269 - Train Accuracy:  0.501, Validation Accuracy:  0.512, Loss:  1.012
    Epoch   1 Batch   82/269 - Train Accuracy:  0.511, Validation Accuracy:  0.523, Loss:  0.977
    Epoch   1 Batch   83/269 - Train Accuracy:  0.524, Validation Accuracy:  0.529, Loss:  1.002
    Epoch   1 Batch   84/269 - Train Accuracy:  0.524, Validation Accuracy:  0.532, Loss:  0.970
    Epoch   1 Batch   85/269 - Train Accuracy:  0.517, Validation Accuracy:  0.530, Loss:  0.973
    Epoch   1 Batch   86/269 - Train Accuracy:  0.500, Validation Accuracy:  0.524, Loss:  0.985
    Epoch   1 Batch   87/269 - Train Accuracy:  0.469, Validation Accuracy:  0.524, Loss:  1.038
    Epoch   1 Batch   88/269 - Train Accuracy:  0.526, Validation Accuracy:  0.539, Loss:  0.964
    Epoch   1 Batch   89/269 - Train Accuracy:  0.529, Validation Accuracy:  0.535, Loss:  0.970
    Epoch   1 Batch   90/269 - Train Accuracy:  0.489, Validation Accuracy:  0.533, Loss:  1.056
    Epoch   1 Batch   91/269 - Train Accuracy:  0.527, Validation Accuracy:  0.545, Loss:  0.943
    Epoch   1 Batch   92/269 - Train Accuracy:  0.525, Validation Accuracy:  0.545, Loss:  0.981
    Epoch   1 Batch   93/269 - Train Accuracy:  0.551, Validation Accuracy:  0.540, Loss:  0.922
    Epoch   1 Batch   94/269 - Train Accuracy:  0.551, Validation Accuracy:  0.553, Loss:  0.996
    Epoch   1 Batch   95/269 - Train Accuracy:  0.514, Validation Accuracy:  0.535, Loss:  0.968
    Epoch   1 Batch   96/269 - Train Accuracy:  0.512, Validation Accuracy:  0.517, Loss:  0.969
    Epoch   1 Batch   97/269 - Train Accuracy:  0.488, Validation Accuracy:  0.524, Loss:  0.971
    Epoch   1 Batch   98/269 - Train Accuracy:  0.515, Validation Accuracy:  0.517, Loss:  0.973
    Epoch   1 Batch   99/269 - Train Accuracy:  0.495, Validation Accuracy:  0.524, Loss:  1.020
    Epoch   1 Batch  100/269 - Train Accuracy:  0.509, Validation Accuracy:  0.518, Loss:  0.938
    Epoch   1 Batch  101/269 - Train Accuracy:  0.479, Validation Accuracy:  0.520, Loss:  1.031
    Epoch   1 Batch  102/269 - Train Accuracy:  0.515, Validation Accuracy:  0.526, Loss:  0.961
    Epoch   1 Batch  103/269 - Train Accuracy:  0.512, Validation Accuracy:  0.519, Loss:  0.952
    Epoch   1 Batch  104/269 - Train Accuracy:  0.497, Validation Accuracy:  0.519, Loss:  0.946
    Epoch   1 Batch  105/269 - Train Accuracy:  0.523, Validation Accuracy:  0.536, Loss:  0.979
    Epoch   1 Batch  106/269 - Train Accuracy:  0.538, Validation Accuracy:  0.552, Loss:  0.949
    Epoch   1 Batch  107/269 - Train Accuracy:  0.500, Validation Accuracy:  0.552, Loss:  0.985
    Epoch   1 Batch  108/269 - Train Accuracy:  0.533, Validation Accuracy:  0.547, Loss:  0.911
    Epoch   1 Batch  109/269 - Train Accuracy:  0.517, Validation Accuracy:  0.552, Loss:  0.969
    Epoch   1 Batch  110/269 - Train Accuracy:  0.530, Validation Accuracy:  0.552, Loss:  0.926
    Epoch   1 Batch  111/269 - Train Accuracy:  0.510, Validation Accuracy:  0.544, Loss:  1.002
    Epoch   1 Batch  112/269 - Train Accuracy:  0.544, Validation Accuracy:  0.544, Loss:  0.922
    Epoch   1 Batch  113/269 - Train Accuracy:  0.549, Validation Accuracy:  0.540, Loss:  0.890
    Epoch   1 Batch  114/269 - Train Accuracy:  0.532, Validation Accuracy:  0.540, Loss:  0.927
    Epoch   1 Batch  115/269 - Train Accuracy:  0.517, Validation Accuracy:  0.536, Loss:  0.955
    Epoch   1 Batch  116/269 - Train Accuracy:  0.535, Validation Accuracy:  0.546, Loss:  0.942
    Epoch   1 Batch  117/269 - Train Accuracy:  0.516, Validation Accuracy:  0.536, Loss:  0.926
    Epoch   1 Batch  118/269 - Train Accuracy:  0.537, Validation Accuracy:  0.542, Loss:  0.897
    Epoch   1 Batch  119/269 - Train Accuracy:  0.530, Validation Accuracy:  0.550, Loss:  0.978
    Epoch   1 Batch  120/269 - Train Accuracy:  0.534, Validation Accuracy:  0.550, Loss:  0.953
    Epoch   1 Batch  121/269 - Train Accuracy:  0.540, Validation Accuracy:  0.550, Loss:  0.916
    Epoch   1 Batch  122/269 - Train Accuracy:  0.543, Validation Accuracy:  0.547, Loss:  0.920
    Epoch   1 Batch  123/269 - Train Accuracy:  0.508, Validation Accuracy:  0.543, Loss:  0.965
    Epoch   1 Batch  124/269 - Train Accuracy:  0.514, Validation Accuracy:  0.535, Loss:  0.885
    Epoch   1 Batch  125/269 - Train Accuracy:  0.530, Validation Accuracy:  0.547, Loss:  0.897
    Epoch   1 Batch  126/269 - Train Accuracy:  0.543, Validation Accuracy:  0.559, Loss:  0.911
    Epoch   1 Batch  127/269 - Train Accuracy:  0.534, Validation Accuracy:  0.557, Loss:  0.947
    Epoch   1 Batch  128/269 - Train Accuracy:  0.556, Validation Accuracy:  0.551, Loss:  0.909
    Epoch   1 Batch  129/269 - Train Accuracy:  0.544, Validation Accuracy:  0.541, Loss:  0.917
    Epoch   1 Batch  130/269 - Train Accuracy:  0.519, Validation Accuracy:  0.538, Loss:  0.944
    Epoch   1 Batch  131/269 - Train Accuracy:  0.543, Validation Accuracy:  0.552, Loss:  0.932
    Epoch   1 Batch  132/269 - Train Accuracy:  0.540, Validation Accuracy:  0.560, Loss:  0.907
    Epoch   1 Batch  133/269 - Train Accuracy:  0.531, Validation Accuracy:  0.558, Loss:  0.873
    Epoch   1 Batch  134/269 - Train Accuracy:  0.533, Validation Accuracy:  0.569, Loss:  0.934
    Epoch   1 Batch  135/269 - Train Accuracy:  0.500, Validation Accuracy:  0.537, Loss:  0.958
    Epoch   1 Batch  136/269 - Train Accuracy:  0.497, Validation Accuracy:  0.529, Loss:  0.955
    Epoch   1 Batch  137/269 - Train Accuracy:  0.518, Validation Accuracy:  0.536, Loss:  0.947
    Epoch   1 Batch  138/269 - Train Accuracy:  0.510, Validation Accuracy:  0.534, Loss:  0.925
    Epoch   1 Batch  139/269 - Train Accuracy:  0.557, Validation Accuracy:  0.541, Loss:  0.880
    Epoch   1 Batch  140/269 - Train Accuracy:  0.559, Validation Accuracy:  0.558, Loss:  0.913
    Epoch   1 Batch  141/269 - Train Accuracy:  0.542, Validation Accuracy:  0.561, Loss:  0.906
    Epoch   1 Batch  142/269 - Train Accuracy:  0.557, Validation Accuracy:  0.558, Loss:  0.866
    Epoch   1 Batch  143/269 - Train Accuracy:  0.553, Validation Accuracy:  0.553, Loss:  0.877
    Epoch   1 Batch  144/269 - Train Accuracy:  0.560, Validation Accuracy:  0.561, Loss:  0.845
    Epoch   1 Batch  145/269 - Train Accuracy:  0.539, Validation Accuracy:  0.554, Loss:  0.874
    Epoch   1 Batch  146/269 - Train Accuracy:  0.559, Validation Accuracy:  0.565, Loss:  0.868
    Epoch   1 Batch  147/269 - Train Accuracy:  0.569, Validation Accuracy:  0.569, Loss:  0.846
    Epoch   1 Batch  148/269 - Train Accuracy:  0.543, Validation Accuracy:  0.541, Loss:  0.899
    Epoch   1 Batch  149/269 - Train Accuracy:  0.550, Validation Accuracy:  0.531, Loss:  0.887
    Epoch   1 Batch  150/269 - Train Accuracy:  0.557, Validation Accuracy:  0.547, Loss:  0.898
    Epoch   1 Batch  151/269 - Train Accuracy:  0.587, Validation Accuracy:  0.562, Loss:  0.835
    Epoch   1 Batch  152/269 - Train Accuracy:  0.553, Validation Accuracy:  0.562, Loss:  0.874
    Epoch   1 Batch  153/269 - Train Accuracy:  0.555, Validation Accuracy:  0.560, Loss:  0.851
    Epoch   1 Batch  154/269 - Train Accuracy:  0.539, Validation Accuracy:  0.556, Loss:  0.874
    Epoch   1 Batch  155/269 - Train Accuracy:  0.584, Validation Accuracy:  0.552, Loss:  0.832
    Epoch   1 Batch  156/269 - Train Accuracy:  0.539, Validation Accuracy:  0.558, Loss:  0.917
    Epoch   1 Batch  157/269 - Train Accuracy:  0.560, Validation Accuracy:  0.558, Loss:  0.868
    Epoch   1 Batch  158/269 - Train Accuracy:  0.562, Validation Accuracy:  0.562, Loss:  0.858
    Epoch   1 Batch  159/269 - Train Accuracy:  0.562, Validation Accuracy:  0.562, Loss:  0.850
    Epoch   1 Batch  160/269 - Train Accuracy:  0.568, Validation Accuracy:  0.571, Loss:  0.853
    Epoch   1 Batch  161/269 - Train Accuracy:  0.557, Validation Accuracy:  0.569, Loss:  0.852
    Epoch   1 Batch  162/269 - Train Accuracy:  0.540, Validation Accuracy:  0.555, Loss:  0.848
    Epoch   1 Batch  163/269 - Train Accuracy:  0.558, Validation Accuracy:  0.548, Loss:  0.848
    Epoch   1 Batch  164/269 - Train Accuracy:  0.553, Validation Accuracy:  0.549, Loss:  0.836
    Epoch   1 Batch  165/269 - Train Accuracy:  0.518, Validation Accuracy:  0.562, Loss:  0.868
    Epoch   1 Batch  166/269 - Train Accuracy:  0.585, Validation Accuracy:  0.568, Loss:  0.807
    Epoch   1 Batch  167/269 - Train Accuracy:  0.555, Validation Accuracy:  0.563, Loss:  0.847
    Epoch   1 Batch  168/269 - Train Accuracy:  0.553, Validation Accuracy:  0.566, Loss:  0.866
    Epoch   1 Batch  169/269 - Train Accuracy:  0.552, Validation Accuracy:  0.574, Loss:  0.842
    Epoch   1 Batch  170/269 - Train Accuracy:  0.564, Validation Accuracy:  0.567, Loss:  0.823
    Epoch   1 Batch  171/269 - Train Accuracy:  0.554, Validation Accuracy:  0.578, Loss:  0.873
    Epoch   1 Batch  172/269 - Train Accuracy:  0.558, Validation Accuracy:  0.577, Loss:  0.853
    Epoch   1 Batch  173/269 - Train Accuracy:  0.549, Validation Accuracy:  0.562, Loss:  0.830
    Epoch   1 Batch  174/269 - Train Accuracy:  0.555, Validation Accuracy:  0.557, Loss:  0.835
    Epoch   1 Batch  175/269 - Train Accuracy:  0.552, Validation Accuracy:  0.554, Loss:  0.846
    Epoch   1 Batch  176/269 - Train Accuracy:  0.545, Validation Accuracy:  0.561, Loss:  0.889
    Epoch   1 Batch  177/269 - Train Accuracy:  0.559, Validation Accuracy:  0.562, Loss:  0.802
    Epoch   1 Batch  178/269 - Train Accuracy:  0.530, Validation Accuracy:  0.561, Loss:  0.861
    Epoch   1 Batch  179/269 - Train Accuracy:  0.578, Validation Accuracy:  0.568, Loss:  0.837
    Epoch   1 Batch  180/269 - Train Accuracy:  0.560, Validation Accuracy:  0.569, Loss:  0.813
    Epoch   1 Batch  181/269 - Train Accuracy:  0.544, Validation Accuracy:  0.560, Loss:  0.839
    Epoch   1 Batch  182/269 - Train Accuracy:  0.570, Validation Accuracy:  0.560, Loss:  0.832
    Epoch   1 Batch  183/269 - Train Accuracy:  0.612, Validation Accuracy:  0.564, Loss:  0.711
    Epoch   1 Batch  184/269 - Train Accuracy:  0.538, Validation Accuracy:  0.566, Loss:  0.866
    Epoch   1 Batch  185/269 - Train Accuracy:  0.563, Validation Accuracy:  0.570, Loss:  0.833
    Epoch   1 Batch  186/269 - Train Accuracy:  0.531, Validation Accuracy:  0.560, Loss:  0.849
    Epoch   1 Batch  187/269 - Train Accuracy:  0.567, Validation Accuracy:  0.551, Loss:  0.817
    Epoch   1 Batch  188/269 - Train Accuracy:  0.571, Validation Accuracy:  0.574, Loss:  0.809
    Epoch   1 Batch  189/269 - Train Accuracy:  0.568, Validation Accuracy:  0.571, Loss:  0.809
    Epoch   1 Batch  190/269 - Train Accuracy:  0.549, Validation Accuracy:  0.559, Loss:  0.802
    Epoch   1 Batch  191/269 - Train Accuracy:  0.573, Validation Accuracy:  0.563, Loss:  0.812
    Epoch   1 Batch  192/269 - Train Accuracy:  0.567, Validation Accuracy:  0.567, Loss:  0.826
    Epoch   1 Batch  193/269 - Train Accuracy:  0.564, Validation Accuracy:  0.566, Loss:  0.809
    Epoch   1 Batch  194/269 - Train Accuracy:  0.557, Validation Accuracy:  0.552, Loss:  0.824
    Epoch   1 Batch  195/269 - Train Accuracy:  0.551, Validation Accuracy:  0.561, Loss:  0.831
    Epoch   1 Batch  196/269 - Train Accuracy:  0.542, Validation Accuracy:  0.558, Loss:  0.810
    Epoch   1 Batch  197/269 - Train Accuracy:  0.532, Validation Accuracy:  0.560, Loss:  0.853
    Epoch   1 Batch  198/269 - Train Accuracy:  0.532, Validation Accuracy:  0.548, Loss:  0.856
    Epoch   1 Batch  199/269 - Train Accuracy:  0.559, Validation Accuracy:  0.575, Loss:  0.836
    Epoch   1 Batch  200/269 - Train Accuracy:  0.548, Validation Accuracy:  0.566, Loss:  0.844
    Epoch   1 Batch  201/269 - Train Accuracy:  0.560, Validation Accuracy:  0.567, Loss:  0.799
    Epoch   1 Batch  202/269 - Train Accuracy:  0.555, Validation Accuracy:  0.549, Loss:  0.800
    Epoch   1 Batch  203/269 - Train Accuracy:  0.536, Validation Accuracy:  0.555, Loss:  0.869
    Epoch   1 Batch  204/269 - Train Accuracy:  0.527, Validation Accuracy:  0.561, Loss:  0.830
    Epoch   1 Batch  205/269 - Train Accuracy:  0.552, Validation Accuracy:  0.560, Loss:  0.792
    Epoch   1 Batch  206/269 - Train Accuracy:  0.544, Validation Accuracy:  0.567, Loss:  0.851
    Epoch   1 Batch  207/269 - Train Accuracy:  0.575, Validation Accuracy:  0.556, Loss:  0.786
    Epoch   1 Batch  208/269 - Train Accuracy:  0.528, Validation Accuracy:  0.563, Loss:  0.857
    Epoch   1 Batch  209/269 - Train Accuracy:  0.561, Validation Accuracy:  0.570, Loss:  0.799
    Epoch   1 Batch  210/269 - Train Accuracy:  0.569, Validation Accuracy:  0.565, Loss:  0.791
    Epoch   1 Batch  211/269 - Train Accuracy:  0.558, Validation Accuracy:  0.567, Loss:  0.813
    Epoch   1 Batch  212/269 - Train Accuracy:  0.560, Validation Accuracy:  0.563, Loss:  0.799
    Epoch   1 Batch  213/269 - Train Accuracy:  0.561, Validation Accuracy:  0.561, Loss:  0.786
    Epoch   1 Batch  214/269 - Train Accuracy:  0.570, Validation Accuracy:  0.573, Loss:  0.795
    Epoch   1 Batch  215/269 - Train Accuracy:  0.587, Validation Accuracy:  0.572, Loss:  0.762
    Epoch   1 Batch  216/269 - Train Accuracy:  0.538, Validation Accuracy:  0.581, Loss:  0.849
    Epoch   1 Batch  217/269 - Train Accuracy:  0.539, Validation Accuracy:  0.569, Loss:  0.817
    Epoch   1 Batch  218/269 - Train Accuracy:  0.549, Validation Accuracy:  0.566, Loss:  0.823
    Epoch   1 Batch  219/269 - Train Accuracy:  0.550, Validation Accuracy:  0.570, Loss:  0.833
    Epoch   1 Batch  220/269 - Train Accuracy:  0.570, Validation Accuracy:  0.571, Loss:  0.743
    Epoch   1 Batch  221/269 - Train Accuracy:  0.575, Validation Accuracy:  0.565, Loss:  0.787
    Epoch   1 Batch  222/269 - Train Accuracy:  0.569, Validation Accuracy:  0.566, Loss:  0.768
    Epoch   1 Batch  223/269 - Train Accuracy:  0.568, Validation Accuracy:  0.567, Loss:  0.771
    Epoch   1 Batch  224/269 - Train Accuracy:  0.572, Validation Accuracy:  0.569, Loss:  0.816
    Epoch   1 Batch  225/269 - Train Accuracy:  0.551, Validation Accuracy:  0.575, Loss:  0.802
    Epoch   1 Batch  226/269 - Train Accuracy:  0.558, Validation Accuracy:  0.572, Loss:  0.776
    Epoch   1 Batch  227/269 - Train Accuracy:  0.623, Validation Accuracy:  0.573, Loss:  0.685
    Epoch   1 Batch  228/269 - Train Accuracy:  0.573, Validation Accuracy:  0.577, Loss:  0.769
    Epoch   1 Batch  229/269 - Train Accuracy:  0.563, Validation Accuracy:  0.580, Loss:  0.776
    Epoch   1 Batch  230/269 - Train Accuracy:  0.566, Validation Accuracy:  0.576, Loss:  0.774
    Epoch   1 Batch  231/269 - Train Accuracy:  0.542, Validation Accuracy:  0.562, Loss:  0.830
    Epoch   1 Batch  232/269 - Train Accuracy:  0.544, Validation Accuracy:  0.569, Loss:  0.804
    Epoch   1 Batch  233/269 - Train Accuracy:  0.577, Validation Accuracy:  0.575, Loss:  0.773
    Epoch   1 Batch  234/269 - Train Accuracy:  0.573, Validation Accuracy:  0.571, Loss:  0.767
    Epoch   1 Batch  235/269 - Train Accuracy:  0.584, Validation Accuracy:  0.575, Loss:  0.766
    Epoch   1 Batch  236/269 - Train Accuracy:  0.566, Validation Accuracy:  0.571, Loss:  0.762
    Epoch   1 Batch  237/269 - Train Accuracy:  0.572, Validation Accuracy:  0.580, Loss:  0.783
    Epoch   1 Batch  238/269 - Train Accuracy:  0.594, Validation Accuracy:  0.577, Loss:  0.758
    Epoch   1 Batch  239/269 - Train Accuracy:  0.581, Validation Accuracy:  0.575, Loss:  0.764
    Epoch   1 Batch  240/269 - Train Accuracy:  0.608, Validation Accuracy:  0.578, Loss:  0.697
    Epoch   1 Batch  241/269 - Train Accuracy:  0.577, Validation Accuracy:  0.584, Loss:  0.786
    Epoch   1 Batch  242/269 - Train Accuracy:  0.550, Validation Accuracy:  0.570, Loss:  0.762
    Epoch   1 Batch  243/269 - Train Accuracy:  0.593, Validation Accuracy:  0.572, Loss:  0.746
    Epoch   1 Batch  244/269 - Train Accuracy:  0.575, Validation Accuracy:  0.579, Loss:  0.759
    Epoch   1 Batch  245/269 - Train Accuracy:  0.559, Validation Accuracy:  0.581, Loss:  0.802
    Epoch   1 Batch  246/269 - Train Accuracy:  0.562, Validation Accuracy:  0.578, Loss:  0.771
    Epoch   1 Batch  247/269 - Train Accuracy:  0.564, Validation Accuracy:  0.582, Loss:  0.782
    Epoch   1 Batch  248/269 - Train Accuracy:  0.568, Validation Accuracy:  0.578, Loss:  0.749
    Epoch   1 Batch  249/269 - Train Accuracy:  0.596, Validation Accuracy:  0.580, Loss:  0.717
    Epoch   1 Batch  250/269 - Train Accuracy:  0.563, Validation Accuracy:  0.588, Loss:  0.771
    Epoch   1 Batch  251/269 - Train Accuracy:  0.591, Validation Accuracy:  0.582, Loss:  0.744
    Epoch   1 Batch  252/269 - Train Accuracy:  0.579, Validation Accuracy:  0.586, Loss:  0.754
    Epoch   1 Batch  253/269 - Train Accuracy:  0.573, Validation Accuracy:  0.579, Loss:  0.757
    Epoch   1 Batch  254/269 - Train Accuracy:  0.572, Validation Accuracy:  0.574, Loss:  0.749
    Epoch   1 Batch  255/269 - Train Accuracy:  0.599, Validation Accuracy:  0.572, Loss:  0.721
    Epoch   1 Batch  256/269 - Train Accuracy:  0.559, Validation Accuracy:  0.575, Loss:  0.760
    Epoch   1 Batch  257/269 - Train Accuracy:  0.546, Validation Accuracy:  0.574, Loss:  0.763
    Epoch   1 Batch  258/269 - Train Accuracy:  0.569, Validation Accuracy:  0.588, Loss:  0.761
    Epoch   1 Batch  259/269 - Train Accuracy:  0.595, Validation Accuracy:  0.586, Loss:  0.747
    Epoch   1 Batch  260/269 - Train Accuracy:  0.562, Validation Accuracy:  0.578, Loss:  0.788
    Epoch   1 Batch  261/269 - Train Accuracy:  0.551, Validation Accuracy:  0.581, Loss:  0.787
    Epoch   1 Batch  262/269 - Train Accuracy:  0.581, Validation Accuracy:  0.583, Loss:  0.743
    Epoch   1 Batch  263/269 - Train Accuracy:  0.575, Validation Accuracy:  0.583, Loss:  0.762
    Epoch   1 Batch  264/269 - Train Accuracy:  0.541, Validation Accuracy:  0.577, Loss:  0.805
    Epoch   1 Batch  265/269 - Train Accuracy:  0.555, Validation Accuracy:  0.584, Loss:  0.780
    Epoch   1 Batch  266/269 - Train Accuracy:  0.588, Validation Accuracy:  0.585, Loss:  0.730
    Epoch   1 Batch  267/269 - Train Accuracy:  0.580, Validation Accuracy:  0.578, Loss:  0.753
    Epoch   2 Batch    0/269 - Train Accuracy:  0.550, Validation Accuracy:  0.585, Loss:  0.792
    Epoch   2 Batch    1/269 - Train Accuracy:  0.562, Validation Accuracy:  0.576, Loss:  0.755
    Epoch   2 Batch    2/269 - Train Accuracy:  0.559, Validation Accuracy:  0.583, Loss:  0.754
    Epoch   2 Batch    3/269 - Train Accuracy:  0.562, Validation Accuracy:  0.588, Loss:  0.755
    Epoch   2 Batch    4/269 - Train Accuracy:  0.559, Validation Accuracy:  0.587, Loss:  0.778
    Epoch   2 Batch    5/269 - Train Accuracy:  0.559, Validation Accuracy:  0.584, Loss:  0.758
    Epoch   2 Batch    6/269 - Train Accuracy:  0.570, Validation Accuracy:  0.583, Loss:  0.715
    Epoch   2 Batch    7/269 - Train Accuracy:  0.572, Validation Accuracy:  0.580, Loss:  0.736
    Epoch   2 Batch    8/269 - Train Accuracy:  0.566, Validation Accuracy:  0.583, Loss:  0.785
    Epoch   2 Batch    9/269 - Train Accuracy:  0.568, Validation Accuracy:  0.588, Loss:  0.751
    Epoch   2 Batch   10/269 - Train Accuracy:  0.561, Validation Accuracy:  0.590, Loss:  0.752
    Epoch   2 Batch   11/269 - Train Accuracy:  0.580, Validation Accuracy:  0.591, Loss:  0.750
    Epoch   2 Batch   12/269 - Train Accuracy:  0.563, Validation Accuracy:  0.591, Loss:  0.770
    Epoch   2 Batch   13/269 - Train Accuracy:  0.601, Validation Accuracy:  0.590, Loss:  0.689
    Epoch   2 Batch   14/269 - Train Accuracy:  0.582, Validation Accuracy:  0.591, Loss:  0.729
    Epoch   2 Batch   15/269 - Train Accuracy:  0.577, Validation Accuracy:  0.592, Loss:  0.715
    Epoch   2 Batch   16/269 - Train Accuracy:  0.606, Validation Accuracy:  0.590, Loss:  0.717
    Epoch   2 Batch   17/269 - Train Accuracy:  0.589, Validation Accuracy:  0.588, Loss:  0.709
    Epoch   2 Batch   18/269 - Train Accuracy:  0.576, Validation Accuracy:  0.590, Loss:  0.740
    Epoch   2 Batch   19/269 - Train Accuracy:  0.629, Validation Accuracy:  0.596, Loss:  0.670
    Epoch   2 Batch   20/269 - Train Accuracy:  0.581, Validation Accuracy:  0.595, Loss:  0.753
    Epoch   2 Batch   21/269 - Train Accuracy:  0.574, Validation Accuracy:  0.596, Loss:  0.777
    Epoch   2 Batch   22/269 - Train Accuracy:  0.592, Validation Accuracy:  0.594, Loss:  0.701
    Epoch   2 Batch   23/269 - Train Accuracy:  0.598, Validation Accuracy:  0.595, Loss:  0.731
    Epoch   2 Batch   24/269 - Train Accuracy:  0.572, Validation Accuracy:  0.593, Loss:  0.747
    Epoch   2 Batch   25/269 - Train Accuracy:  0.549, Validation Accuracy:  0.592, Loss:  0.769
    Epoch   2 Batch   26/269 - Train Accuracy:  0.608, Validation Accuracy:  0.589, Loss:  0.676
    Epoch   2 Batch   27/269 - Train Accuracy:  0.569, Validation Accuracy:  0.587, Loss:  0.704
    Epoch   2 Batch   28/269 - Train Accuracy:  0.544, Validation Accuracy:  0.589, Loss:  0.769
    Epoch   2 Batch   29/269 - Train Accuracy:  0.571, Validation Accuracy:  0.586, Loss:  0.733
    Epoch   2 Batch   30/269 - Train Accuracy:  0.590, Validation Accuracy:  0.594, Loss:  0.708
    Epoch   2 Batch   31/269 - Train Accuracy:  0.600, Validation Accuracy:  0.593, Loss:  0.695
    Epoch   2 Batch   32/269 - Train Accuracy:  0.578, Validation Accuracy:  0.589, Loss:  0.706
    Epoch   2 Batch   33/269 - Train Accuracy:  0.608, Validation Accuracy:  0.592, Loss:  0.695
    Epoch   2 Batch   34/269 - Train Accuracy:  0.592, Validation Accuracy:  0.594, Loss:  0.717
    Epoch   2 Batch   35/269 - Train Accuracy:  0.586, Validation Accuracy:  0.589, Loss:  0.736
    Epoch   2 Batch   36/269 - Train Accuracy:  0.593, Validation Accuracy:  0.589, Loss:  0.706
    Epoch   2 Batch   37/269 - Train Accuracy:  0.596, Validation Accuracy:  0.593, Loss:  0.697
    Epoch   2 Batch   38/269 - Train Accuracy:  0.594, Validation Accuracy:  0.593, Loss:  0.705
    Epoch   2 Batch   39/269 - Train Accuracy:  0.591, Validation Accuracy:  0.587, Loss:  0.702
    Epoch   2 Batch   40/269 - Train Accuracy:  0.571, Validation Accuracy:  0.590, Loss:  0.739
    Epoch   2 Batch   41/269 - Train Accuracy:  0.575, Validation Accuracy:  0.586, Loss:  0.719
    Epoch   2 Batch   42/269 - Train Accuracy:  0.609, Validation Accuracy:  0.586, Loss:  0.663
    Epoch   2 Batch   43/269 - Train Accuracy:  0.574, Validation Accuracy:  0.593, Loss:  0.734
    Epoch   2 Batch   44/269 - Train Accuracy:  0.590, Validation Accuracy:  0.595, Loss:  0.709
    Epoch   2 Batch   45/269 - Train Accuracy:  0.563, Validation Accuracy:  0.588, Loss:  0.745
    Epoch   2 Batch   46/269 - Train Accuracy:  0.582, Validation Accuracy:  0.587, Loss:  0.731
    Epoch   2 Batch   47/269 - Train Accuracy:  0.610, Validation Accuracy:  0.587, Loss:  0.661
    Epoch   2 Batch   48/269 - Train Accuracy:  0.581, Validation Accuracy:  0.589, Loss:  0.701
    Epoch   2 Batch   49/269 - Train Accuracy:  0.575, Validation Accuracy:  0.585, Loss:  0.723
    Epoch   2 Batch   50/269 - Train Accuracy:  0.547, Validation Accuracy:  0.589, Loss:  0.743
    Epoch   2 Batch   51/269 - Train Accuracy:  0.574, Validation Accuracy:  0.592, Loss:  0.701
    Epoch   2 Batch   52/269 - Train Accuracy:  0.591, Validation Accuracy:  0.590, Loss:  0.668
    Epoch   2 Batch   53/269 - Train Accuracy:  0.574, Validation Accuracy:  0.591, Loss:  0.740
    Epoch   2 Batch   54/269 - Train Accuracy:  0.595, Validation Accuracy:  0.594, Loss:  0.718
    Epoch   2 Batch   55/269 - Train Accuracy:  0.592, Validation Accuracy:  0.591, Loss:  0.702
    Epoch   2 Batch   56/269 - Train Accuracy:  0.595, Validation Accuracy:  0.593, Loss:  0.704
    Epoch   2 Batch   57/269 - Train Accuracy:  0.591, Validation Accuracy:  0.591, Loss:  0.714
    Epoch   2 Batch   58/269 - Train Accuracy:  0.595, Validation Accuracy:  0.598, Loss:  0.689
    Epoch   2 Batch   59/269 - Train Accuracy:  0.594, Validation Accuracy:  0.594, Loss:  0.677
    Epoch   2 Batch   60/269 - Train Accuracy:  0.604, Validation Accuracy:  0.598, Loss:  0.668
    Epoch   2 Batch   61/269 - Train Accuracy:  0.616, Validation Accuracy:  0.598, Loss:  0.647
    Epoch   2 Batch   62/269 - Train Accuracy:  0.604, Validation Accuracy:  0.586, Loss:  0.681
    Epoch   2 Batch   63/269 - Train Accuracy:  0.557, Validation Accuracy:  0.589, Loss:  0.719
    Epoch   2 Batch   64/269 - Train Accuracy:  0.596, Validation Accuracy:  0.597, Loss:  0.677
    Epoch   2 Batch   65/269 - Train Accuracy:  0.596, Validation Accuracy:  0.592, Loss:  0.685
    Epoch   2 Batch   66/269 - Train Accuracy:  0.594, Validation Accuracy:  0.590, Loss:  0.659
    Epoch   2 Batch   67/269 - Train Accuracy:  0.586, Validation Accuracy:  0.591, Loss:  0.705
    Epoch   2 Batch   68/269 - Train Accuracy:  0.576, Validation Accuracy:  0.593, Loss:  0.697
    Epoch   2 Batch   69/269 - Train Accuracy:  0.565, Validation Accuracy:  0.596, Loss:  0.773
    Epoch   2 Batch   70/269 - Train Accuracy:  0.595, Validation Accuracy:  0.592, Loss:  0.698
    Epoch   2 Batch   71/269 - Train Accuracy:  0.577, Validation Accuracy:  0.598, Loss:  0.712
    Epoch   2 Batch   72/269 - Train Accuracy:  0.599, Validation Accuracy:  0.601, Loss:  0.678
    Epoch   2 Batch   73/269 - Train Accuracy:  0.599, Validation Accuracy:  0.602, Loss:  0.706
    Epoch   2 Batch   74/269 - Train Accuracy:  0.587, Validation Accuracy:  0.601, Loss:  0.698
    Epoch   2 Batch   75/269 - Train Accuracy:  0.583, Validation Accuracy:  0.594, Loss:  0.692
    Epoch   2 Batch   76/269 - Train Accuracy:  0.580, Validation Accuracy:  0.594, Loss:  0.700
    Epoch   2 Batch   77/269 - Train Accuracy:  0.618, Validation Accuracy:  0.591, Loss:  0.677
    Epoch   2 Batch   78/269 - Train Accuracy:  0.605, Validation Accuracy:  0.595, Loss:  0.675
    Epoch   2 Batch   79/269 - Train Accuracy:  0.597, Validation Accuracy:  0.594, Loss:  0.672
    Epoch   2 Batch   80/269 - Train Accuracy:  0.597, Validation Accuracy:  0.599, Loss:  0.684
    Epoch   2 Batch   81/269 - Train Accuracy:  0.605, Validation Accuracy:  0.608, Loss:  0.689
    Epoch   2 Batch   82/269 - Train Accuracy:  0.608, Validation Accuracy:  0.607, Loss:  0.655
    Epoch   2 Batch   83/269 - Train Accuracy:  0.599, Validation Accuracy:  0.607, Loss:  0.689
    Epoch   2 Batch   84/269 - Train Accuracy:  0.608, Validation Accuracy:  0.599, Loss:  0.660
    Epoch   2 Batch   85/269 - Train Accuracy:  0.598, Validation Accuracy:  0.603, Loss:  0.678
    Epoch   2 Batch   86/269 - Train Accuracy:  0.575, Validation Accuracy:  0.598, Loss:  0.669
    Epoch   2 Batch   87/269 - Train Accuracy:  0.576, Validation Accuracy:  0.599, Loss:  0.711
    Epoch   2 Batch   88/269 - Train Accuracy:  0.595, Validation Accuracy:  0.599, Loss:  0.668
    Epoch   2 Batch   89/269 - Train Accuracy:  0.613, Validation Accuracy:  0.595, Loss:  0.677
    Epoch   2 Batch   90/269 - Train Accuracy:  0.566, Validation Accuracy:  0.596, Loss:  0.716
    Epoch   2 Batch   91/269 - Train Accuracy:  0.601, Validation Accuracy:  0.594, Loss:  0.659
    Epoch   2 Batch   92/269 - Train Accuracy:  0.580, Validation Accuracy:  0.600, Loss:  0.656
    Epoch   2 Batch   93/269 - Train Accuracy:  0.612, Validation Accuracy:  0.597, Loss:  0.641
    Epoch   2 Batch   94/269 - Train Accuracy:  0.590, Validation Accuracy:  0.602, Loss:  0.682
    Epoch   2 Batch   95/269 - Train Accuracy:  0.586, Validation Accuracy:  0.600, Loss:  0.683
    Epoch   2 Batch   96/269 - Train Accuracy:  0.607, Validation Accuracy:  0.602, Loss:  0.662
    Epoch   2 Batch   97/269 - Train Accuracy:  0.589, Validation Accuracy:  0.604, Loss:  0.655
    Epoch   2 Batch   98/269 - Train Accuracy:  0.608, Validation Accuracy:  0.604, Loss:  0.675
    Epoch   2 Batch   99/269 - Train Accuracy:  0.598, Validation Accuracy:  0.597, Loss:  0.694
    Epoch   2 Batch  100/269 - Train Accuracy:  0.605, Validation Accuracy:  0.599, Loss:  0.673
    Epoch   2 Batch  101/269 - Train Accuracy:  0.565, Validation Accuracy:  0.596, Loss:  0.710
    Epoch   2 Batch  102/269 - Train Accuracy:  0.590, Validation Accuracy:  0.596, Loss:  0.656
    Epoch   2 Batch  103/269 - Train Accuracy:  0.598, Validation Accuracy:  0.598, Loss:  0.660
    Epoch   2 Batch  104/269 - Train Accuracy:  0.589, Validation Accuracy:  0.607, Loss:  0.664
    Epoch   2 Batch  105/269 - Train Accuracy:  0.599, Validation Accuracy:  0.608, Loss:  0.676
    Epoch   2 Batch  106/269 - Train Accuracy:  0.604, Validation Accuracy:  0.594, Loss:  0.655
    Epoch   2 Batch  107/269 - Train Accuracy:  0.563, Validation Accuracy:  0.604, Loss:  0.705
    Epoch   2 Batch  108/269 - Train Accuracy:  0.601, Validation Accuracy:  0.602, Loss:  0.679
    Epoch   2 Batch  109/269 - Train Accuracy:  0.575, Validation Accuracy:  0.582, Loss:  0.660
    Epoch   2 Batch  110/269 - Train Accuracy:  0.589, Validation Accuracy:  0.596, Loss:  0.675
    Epoch   2 Batch  111/269 - Train Accuracy:  0.559, Validation Accuracy:  0.591, Loss:  0.697
    Epoch   2 Batch  112/269 - Train Accuracy:  0.609, Validation Accuracy:  0.596, Loss:  0.671
    Epoch   2 Batch  113/269 - Train Accuracy:  0.597, Validation Accuracy:  0.597, Loss:  0.631
    Epoch   2 Batch  114/269 - Train Accuracy:  0.602, Validation Accuracy:  0.601, Loss:  0.668
    Epoch   2 Batch  115/269 - Train Accuracy:  0.582, Validation Accuracy:  0.603, Loss:  0.687
    Epoch   2 Batch  116/269 - Train Accuracy:  0.595, Validation Accuracy:  0.599, Loss:  0.674
    Epoch   2 Batch  117/269 - Train Accuracy:  0.591, Validation Accuracy:  0.608, Loss:  0.650
    Epoch   2 Batch  118/269 - Train Accuracy:  0.607, Validation Accuracy:  0.608, Loss:  0.642
    Epoch   2 Batch  119/269 - Train Accuracy:  0.592, Validation Accuracy:  0.603, Loss:  0.686
    Epoch   2 Batch  120/269 - Train Accuracy:  0.581, Validation Accuracy:  0.602, Loss:  0.677
    Epoch   2 Batch  121/269 - Train Accuracy:  0.606, Validation Accuracy:  0.614, Loss:  0.657
    Epoch   2 Batch  122/269 - Train Accuracy:  0.614, Validation Accuracy:  0.615, Loss:  0.642
    Epoch   2 Batch  123/269 - Train Accuracy:  0.576, Validation Accuracy:  0.599, Loss:  0.675
    Epoch   2 Batch  124/269 - Train Accuracy:  0.591, Validation Accuracy:  0.592, Loss:  0.632
    Epoch   2 Batch  125/269 - Train Accuracy:  0.590, Validation Accuracy:  0.601, Loss:  0.647
    Epoch   2 Batch  126/269 - Train Accuracy:  0.596, Validation Accuracy:  0.593, Loss:  0.647
    Epoch   2 Batch  127/269 - Train Accuracy:  0.576, Validation Accuracy:  0.601, Loss:  0.668
    Epoch   2 Batch  128/269 - Train Accuracy:  0.610, Validation Accuracy:  0.599, Loss:  0.646
    Epoch   2 Batch  129/269 - Train Accuracy:  0.610, Validation Accuracy:  0.604, Loss:  0.646
    Epoch   2 Batch  130/269 - Train Accuracy:  0.594, Validation Accuracy:  0.601, Loss:  0.669
    Epoch   2 Batch  131/269 - Train Accuracy:  0.599, Validation Accuracy:  0.607, Loss:  0.664
    Epoch   2 Batch  132/269 - Train Accuracy:  0.605, Validation Accuracy:  0.607, Loss:  0.641
    Epoch   2 Batch  133/269 - Train Accuracy:  0.610, Validation Accuracy:  0.606, Loss:  0.621
    Epoch   2 Batch  134/269 - Train Accuracy:  0.568, Validation Accuracy:  0.605, Loss:  0.667
    Epoch   2 Batch  135/269 - Train Accuracy:  0.589, Validation Accuracy:  0.614, Loss:  0.671
    Epoch   2 Batch  136/269 - Train Accuracy:  0.584, Validation Accuracy:  0.617, Loss:  0.674
    Epoch   2 Batch  137/269 - Train Accuracy:  0.608, Validation Accuracy:  0.605, Loss:  0.671
    Epoch   2 Batch  138/269 - Train Accuracy:  0.597, Validation Accuracy:  0.608, Loss:  0.643
    Epoch   2 Batch  139/269 - Train Accuracy:  0.620, Validation Accuracy:  0.604, Loss:  0.616
    Epoch   2 Batch  140/269 - Train Accuracy:  0.615, Validation Accuracy:  0.606, Loss:  0.651
    Epoch   2 Batch  141/269 - Train Accuracy:  0.603, Validation Accuracy:  0.606, Loss:  0.656
    Epoch   2 Batch  142/269 - Train Accuracy:  0.616, Validation Accuracy:  0.607, Loss:  0.612
    Epoch   2 Batch  143/269 - Train Accuracy:  0.604, Validation Accuracy:  0.611, Loss:  0.628
    Epoch   2 Batch  144/269 - Train Accuracy:  0.620, Validation Accuracy:  0.617, Loss:  0.605
    Epoch   2 Batch  145/269 - Train Accuracy:  0.612, Validation Accuracy:  0.619, Loss:  0.622
    Epoch   2 Batch  146/269 - Train Accuracy:  0.618, Validation Accuracy:  0.615, Loss:  0.618
    Epoch   2 Batch  147/269 - Train Accuracy:  0.631, Validation Accuracy:  0.612, Loss:  0.604
    Epoch   2 Batch  148/269 - Train Accuracy:  0.606, Validation Accuracy:  0.608, Loss:  0.627
    Epoch   2 Batch  149/269 - Train Accuracy:  0.608, Validation Accuracy:  0.609, Loss:  0.634
    Epoch   2 Batch  150/269 - Train Accuracy:  0.605, Validation Accuracy:  0.607, Loss:  0.629
    Epoch   2 Batch  151/269 - Train Accuracy:  0.637, Validation Accuracy:  0.614, Loss:  0.607
    Epoch   2 Batch  152/269 - Train Accuracy:  0.615, Validation Accuracy:  0.617, Loss:  0.635
    Epoch   2 Batch  153/269 - Train Accuracy:  0.618, Validation Accuracy:  0.617, Loss:  0.606
    Epoch   2 Batch  154/269 - Train Accuracy:  0.592, Validation Accuracy:  0.614, Loss:  0.623
    Epoch   2 Batch  155/269 - Train Accuracy:  0.648, Validation Accuracy:  0.618, Loss:  0.593
    Epoch   2 Batch  156/269 - Train Accuracy:  0.605, Validation Accuracy:  0.613, Loss:  0.650
    Epoch   2 Batch  157/269 - Train Accuracy:  0.623, Validation Accuracy:  0.618, Loss:  0.616
    Epoch   2 Batch  158/269 - Train Accuracy:  0.618, Validation Accuracy:  0.618, Loss:  0.619
    Epoch   2 Batch  159/269 - Train Accuracy:  0.614, Validation Accuracy:  0.614, Loss:  0.616
    Epoch   2 Batch  160/269 - Train Accuracy:  0.622, Validation Accuracy:  0.609, Loss:  0.626
    Epoch   2 Batch  161/269 - Train Accuracy:  0.600, Validation Accuracy:  0.607, Loss:  0.615
    Epoch   2 Batch  162/269 - Train Accuracy:  0.607, Validation Accuracy:  0.606, Loss:  0.620
    Epoch   2 Batch  163/269 - Train Accuracy:  0.616, Validation Accuracy:  0.604, Loss:  0.623
    Epoch   2 Batch  164/269 - Train Accuracy:  0.619, Validation Accuracy:  0.611, Loss:  0.598
    Epoch   2 Batch  165/269 - Train Accuracy:  0.583, Validation Accuracy:  0.604, Loss:  0.625
    Epoch   2 Batch  166/269 - Train Accuracy:  0.643, Validation Accuracy:  0.615, Loss:  0.587
    Epoch   2 Batch  167/269 - Train Accuracy:  0.618, Validation Accuracy:  0.612, Loss:  0.606
    Epoch   2 Batch  168/269 - Train Accuracy:  0.600, Validation Accuracy:  0.615, Loss:  0.621
    Epoch   2 Batch  169/269 - Train Accuracy:  0.610, Validation Accuracy:  0.618, Loss:  0.622
    Epoch   2 Batch  170/269 - Train Accuracy:  0.605, Validation Accuracy:  0.612, Loss:  0.599
    Epoch   2 Batch  171/269 - Train Accuracy:  0.601, Validation Accuracy:  0.608, Loss:  0.627
    Epoch   2 Batch  172/269 - Train Accuracy:  0.601, Validation Accuracy:  0.612, Loss:  0.614
    Epoch   2 Batch  173/269 - Train Accuracy:  0.611, Validation Accuracy:  0.612, Loss:  0.596
    Epoch   2 Batch  174/269 - Train Accuracy:  0.617, Validation Accuracy:  0.615, Loss:  0.604
    Epoch   2 Batch  175/269 - Train Accuracy:  0.622, Validation Accuracy:  0.623, Loss:  0.619
    Epoch   2 Batch  176/269 - Train Accuracy:  0.607, Validation Accuracy:  0.621, Loss:  0.642
    Epoch   2 Batch  177/269 - Train Accuracy:  0.619, Validation Accuracy:  0.622, Loss:  0.577
    Epoch   2 Batch  178/269 - Train Accuracy:  0.607, Validation Accuracy:  0.622, Loss:  0.621
    Epoch   2 Batch  179/269 - Train Accuracy:  0.625, Validation Accuracy:  0.623, Loss:  0.596
    Epoch   2 Batch  180/269 - Train Accuracy:  0.609, Validation Accuracy:  0.618, Loss:  0.590
    Epoch   2 Batch  181/269 - Train Accuracy:  0.603, Validation Accuracy:  0.618, Loss:  0.597
    Epoch   2 Batch  182/269 - Train Accuracy:  0.632, Validation Accuracy:  0.616, Loss:  0.595
    Epoch   2 Batch  183/269 - Train Accuracy:  0.670, Validation Accuracy:  0.621, Loss:  0.519
    Epoch   2 Batch  184/269 - Train Accuracy:  0.590, Validation Accuracy:  0.618, Loss:  0.615
    Epoch   2 Batch  185/269 - Train Accuracy:  0.629, Validation Accuracy:  0.619, Loss:  0.585
    Epoch   2 Batch  186/269 - Train Accuracy:  0.592, Validation Accuracy:  0.623, Loss:  0.615
    Epoch   2 Batch  187/269 - Train Accuracy:  0.619, Validation Accuracy:  0.623, Loss:  0.594
    Epoch   2 Batch  188/269 - Train Accuracy:  0.624, Validation Accuracy:  0.614, Loss:  0.588
    Epoch   2 Batch  189/269 - Train Accuracy:  0.626, Validation Accuracy:  0.617, Loss:  0.583
    Epoch   2 Batch  190/269 - Train Accuracy:  0.622, Validation Accuracy:  0.623, Loss:  0.576
    Epoch   2 Batch  191/269 - Train Accuracy:  0.634, Validation Accuracy:  0.628, Loss:  0.583
    Epoch   2 Batch  192/269 - Train Accuracy:  0.619, Validation Accuracy:  0.630, Loss:  0.595
    Epoch   2 Batch  193/269 - Train Accuracy:  0.617, Validation Accuracy:  0.626, Loss:  0.577
    Epoch   2 Batch  194/269 - Train Accuracy:  0.637, Validation Accuracy:  0.621, Loss:  0.598
    Epoch   2 Batch  195/269 - Train Accuracy:  0.622, Validation Accuracy:  0.622, Loss:  0.584
    Epoch   2 Batch  196/269 - Train Accuracy:  0.602, Validation Accuracy:  0.624, Loss:  0.577
    Epoch   2 Batch  197/269 - Train Accuracy:  0.596, Validation Accuracy:  0.621, Loss:  0.618
    Epoch   2 Batch  198/269 - Train Accuracy:  0.587, Validation Accuracy:  0.614, Loss:  0.622
    Epoch   2 Batch  199/269 - Train Accuracy:  0.607, Validation Accuracy:  0.615, Loss:  0.602
    Epoch   2 Batch  200/269 - Train Accuracy:  0.621, Validation Accuracy:  0.620, Loss:  0.612
    Epoch   2 Batch  201/269 - Train Accuracy:  0.626, Validation Accuracy:  0.625, Loss:  0.575
    Epoch   2 Batch  202/269 - Train Accuracy:  0.627, Validation Accuracy:  0.626, Loss:  0.585
    Epoch   2 Batch  203/269 - Train Accuracy:  0.599, Validation Accuracy:  0.620, Loss:  0.625
    Epoch   2 Batch  204/269 - Train Accuracy:  0.591, Validation Accuracy:  0.621, Loss:  0.614
    Epoch   2 Batch  205/269 - Train Accuracy:  0.621, Validation Accuracy:  0.624, Loss:  0.576
    Epoch   2 Batch  206/269 - Train Accuracy:  0.609, Validation Accuracy:  0.623, Loss:  0.600
    Epoch   2 Batch  207/269 - Train Accuracy:  0.646, Validation Accuracy:  0.627, Loss:  0.565
    Epoch   2 Batch  208/269 - Train Accuracy:  0.604, Validation Accuracy:  0.629, Loss:  0.608
    Epoch   2 Batch  209/269 - Train Accuracy:  0.627, Validation Accuracy:  0.623, Loss:  0.581
    Epoch   2 Batch  210/269 - Train Accuracy:  0.634, Validation Accuracy:  0.622, Loss:  0.560
    Epoch   2 Batch  211/269 - Train Accuracy:  0.626, Validation Accuracy:  0.626, Loss:  0.581
    Epoch   2 Batch  212/269 - Train Accuracy:  0.646, Validation Accuracy:  0.632, Loss:  0.574
    Epoch   2 Batch  213/269 - Train Accuracy:  0.623, Validation Accuracy:  0.621, Loss:  0.572
    Epoch   2 Batch  214/269 - Train Accuracy:  0.628, Validation Accuracy:  0.630, Loss:  0.581
    Epoch   2 Batch  215/269 - Train Accuracy:  0.649, Validation Accuracy:  0.625, Loss:  0.554
    Epoch   2 Batch  216/269 - Train Accuracy:  0.591, Validation Accuracy:  0.618, Loss:  0.611
    Epoch   2 Batch  217/269 - Train Accuracy:  0.588, Validation Accuracy:  0.629, Loss:  0.598
    Epoch   2 Batch  218/269 - Train Accuracy:  0.626, Validation Accuracy:  0.626, Loss:  0.600
    Epoch   2 Batch  219/269 - Train Accuracy:  0.615, Validation Accuracy:  0.627, Loss:  0.605
    Epoch   2 Batch  220/269 - Train Accuracy:  0.634, Validation Accuracy:  0.629, Loss:  0.536
    Epoch   2 Batch  221/269 - Train Accuracy:  0.646, Validation Accuracy:  0.623, Loss:  0.569
    Epoch   2 Batch  222/269 - Train Accuracy:  0.620, Validation Accuracy:  0.621, Loss:  0.567
    Epoch   2 Batch  223/269 - Train Accuracy:  0.625, Validation Accuracy:  0.629, Loss:  0.559
    Epoch   2 Batch  224/269 - Train Accuracy:  0.621, Validation Accuracy:  0.632, Loss:  0.597
    Epoch   2 Batch  225/269 - Train Accuracy:  0.607, Validation Accuracy:  0.639, Loss:  0.583
    Epoch   2 Batch  226/269 - Train Accuracy:  0.612, Validation Accuracy:  0.633, Loss:  0.569
    Epoch   2 Batch  227/269 - Train Accuracy:  0.675, Validation Accuracy:  0.624, Loss:  0.512
    Epoch   2 Batch  228/269 - Train Accuracy:  0.632, Validation Accuracy:  0.628, Loss:  0.561
    Epoch   2 Batch  229/269 - Train Accuracy:  0.614, Validation Accuracy:  0.636, Loss:  0.567
    Epoch   2 Batch  230/269 - Train Accuracy:  0.630, Validation Accuracy:  0.632, Loss:  0.568
    Epoch   2 Batch  231/269 - Train Accuracy:  0.589, Validation Accuracy:  0.626, Loss:  0.613
    Epoch   2 Batch  232/269 - Train Accuracy:  0.601, Validation Accuracy:  0.622, Loss:  0.619
    Epoch   2 Batch  233/269 - Train Accuracy:  0.639, Validation Accuracy:  0.624, Loss:  0.579
    Epoch   2 Batch  234/269 - Train Accuracy:  0.628, Validation Accuracy:  0.625, Loss:  0.566
    Epoch   2 Batch  235/269 - Train Accuracy:  0.641, Validation Accuracy:  0.629, Loss:  0.549
    Epoch   2 Batch  236/269 - Train Accuracy:  0.618, Validation Accuracy:  0.626, Loss:  0.563
    Epoch   2 Batch  237/269 - Train Accuracy:  0.598, Validation Accuracy:  0.624, Loss:  0.574
    Epoch   2 Batch  238/269 - Train Accuracy:  0.639, Validation Accuracy:  0.638, Loss:  0.555
    Epoch   2 Batch  239/269 - Train Accuracy:  0.644, Validation Accuracy:  0.637, Loss:  0.554
    Epoch   2 Batch  240/269 - Train Accuracy:  0.650, Validation Accuracy:  0.631, Loss:  0.507
    Epoch   2 Batch  241/269 - Train Accuracy:  0.621, Validation Accuracy:  0.629, Loss:  0.577
    Epoch   2 Batch  242/269 - Train Accuracy:  0.612, Validation Accuracy:  0.634, Loss:  0.559
    Epoch   2 Batch  243/269 - Train Accuracy:  0.653, Validation Accuracy:  0.622, Loss:  0.557
    Epoch   2 Batch  244/269 - Train Accuracy:  0.612, Validation Accuracy:  0.632, Loss:  0.562
    Epoch   2 Batch  245/269 - Train Accuracy:  0.616, Validation Accuracy:  0.628, Loss:  0.589
    Epoch   2 Batch  246/269 - Train Accuracy:  0.624, Validation Accuracy:  0.625, Loss:  0.566
    Epoch   2 Batch  247/269 - Train Accuracy:  0.605, Validation Accuracy:  0.620, Loss:  0.576
    Epoch   2 Batch  248/269 - Train Accuracy:  0.620, Validation Accuracy:  0.630, Loss:  0.554
    Epoch   2 Batch  249/269 - Train Accuracy:  0.646, Validation Accuracy:  0.627, Loss:  0.520
    Epoch   2 Batch  250/269 - Train Accuracy:  0.609, Validation Accuracy:  0.627, Loss:  0.559
    Epoch   2 Batch  251/269 - Train Accuracy:  0.654, Validation Accuracy:  0.628, Loss:  0.545
    Epoch   2 Batch  252/269 - Train Accuracy:  0.630, Validation Accuracy:  0.629, Loss:  0.548
    Epoch   2 Batch  253/269 - Train Accuracy:  0.632, Validation Accuracy:  0.631, Loss:  0.566
    Epoch   2 Batch  254/269 - Train Accuracy:  0.623, Validation Accuracy:  0.631, Loss:  0.554
    Epoch   2 Batch  255/269 - Train Accuracy:  0.648, Validation Accuracy:  0.632, Loss:  0.528
    Epoch   2 Batch  256/269 - Train Accuracy:  0.626, Validation Accuracy:  0.638, Loss:  0.556
    Epoch   2 Batch  257/269 - Train Accuracy:  0.594, Validation Accuracy:  0.618, Loss:  0.565
    Epoch   2 Batch  258/269 - Train Accuracy:  0.617, Validation Accuracy:  0.626, Loss:  0.557
    Epoch   2 Batch  259/269 - Train Accuracy:  0.646, Validation Accuracy:  0.635, Loss:  0.552
    Epoch   2 Batch  260/269 - Train Accuracy:  0.621, Validation Accuracy:  0.636, Loss:  0.573
    Epoch   2 Batch  261/269 - Train Accuracy:  0.608, Validation Accuracy:  0.637, Loss:  0.570
    Epoch   2 Batch  262/269 - Train Accuracy:  0.638, Validation Accuracy:  0.636, Loss:  0.544
    Epoch   2 Batch  263/269 - Train Accuracy:  0.631, Validation Accuracy:  0.638, Loss:  0.557
    Epoch   2 Batch  264/269 - Train Accuracy:  0.619, Validation Accuracy:  0.637, Loss:  0.582
    Epoch   2 Batch  265/269 - Train Accuracy:  0.625, Validation Accuracy:  0.639, Loss:  0.568
    Epoch   2 Batch  266/269 - Train Accuracy:  0.632, Validation Accuracy:  0.641, Loss:  0.542
    Epoch   2 Batch  267/269 - Train Accuracy:  0.642, Validation Accuracy:  0.642, Loss:  0.550
    Epoch   3 Batch    0/269 - Train Accuracy:  0.606, Validation Accuracy:  0.639, Loss:  0.567
    Epoch   3 Batch    1/269 - Train Accuracy:  0.614, Validation Accuracy:  0.638, Loss:  0.556
    Epoch   3 Batch    2/269 - Train Accuracy:  0.630, Validation Accuracy:  0.645, Loss:  0.549
    Epoch   3 Batch    3/269 - Train Accuracy:  0.622, Validation Accuracy:  0.635, Loss:  0.554
    Epoch   3 Batch    4/269 - Train Accuracy:  0.610, Validation Accuracy:  0.635, Loss:  0.568
    Epoch   3 Batch    5/269 - Train Accuracy:  0.618, Validation Accuracy:  0.638, Loss:  0.554
    Epoch   3 Batch    6/269 - Train Accuracy:  0.630, Validation Accuracy:  0.636, Loss:  0.512
    Epoch   3 Batch    7/269 - Train Accuracy:  0.635, Validation Accuracy:  0.637, Loss:  0.534
    Epoch   3 Batch    8/269 - Train Accuracy:  0.625, Validation Accuracy:  0.638, Loss:  0.561
    Epoch   3 Batch    9/269 - Train Accuracy:  0.620, Validation Accuracy:  0.642, Loss:  0.558
    Epoch   3 Batch   10/269 - Train Accuracy:  0.626, Validation Accuracy:  0.650, Loss:  0.554
    Epoch   3 Batch   11/269 - Train Accuracy:  0.629, Validation Accuracy:  0.641, Loss:  0.549
    Epoch   3 Batch   12/269 - Train Accuracy:  0.624, Validation Accuracy:  0.652, Loss:  0.564
    Epoch   3 Batch   13/269 - Train Accuracy:  0.658, Validation Accuracy:  0.651, Loss:  0.503
    Epoch   3 Batch   14/269 - Train Accuracy:  0.626, Validation Accuracy:  0.653, Loss:  0.538
    Epoch   3 Batch   15/269 - Train Accuracy:  0.629, Validation Accuracy:  0.651, Loss:  0.520
    Epoch   3 Batch   16/269 - Train Accuracy:  0.651, Validation Accuracy:  0.652, Loss:  0.530
    Epoch   3 Batch   17/269 - Train Accuracy:  0.631, Validation Accuracy:  0.639, Loss:  0.522
    Epoch   3 Batch   18/269 - Train Accuracy:  0.635, Validation Accuracy:  0.647, Loss:  0.549
    Epoch   3 Batch   19/269 - Train Accuracy:  0.678, Validation Accuracy:  0.648, Loss:  0.489
    Epoch   3 Batch   20/269 - Train Accuracy:  0.603, Validation Accuracy:  0.631, Loss:  0.562
    Epoch   3 Batch   21/269 - Train Accuracy:  0.631, Validation Accuracy:  0.647, Loss:  0.591
    Epoch   3 Batch   22/269 - Train Accuracy:  0.636, Validation Accuracy:  0.632, Loss:  0.523
    Epoch   3 Batch   23/269 - Train Accuracy:  0.598, Validation Accuracy:  0.600, Loss:  0.560
    Epoch   3 Batch   24/269 - Train Accuracy:  0.568, Validation Accuracy:  0.602, Loss:  0.685
    Epoch   3 Batch   25/269 - Train Accuracy:  0.556, Validation Accuracy:  0.595, Loss:  0.861
    Epoch   3 Batch   26/269 - Train Accuracy:  0.577, Validation Accuracy:  0.557, Loss:  0.627
    Epoch   3 Batch   27/269 - Train Accuracy:  0.526, Validation Accuracy:  0.552, Loss:  0.850
    Epoch   3 Batch   28/269 - Train Accuracy:  0.441, Validation Accuracy:  0.509, Loss:  0.889
    Epoch   3 Batch   29/269 - Train Accuracy:  0.528, Validation Accuracy:  0.570, Loss:  0.989
    Epoch   3 Batch   30/269 - Train Accuracy:  0.543, Validation Accuracy:  0.546, Loss:  0.851
    Epoch   3 Batch   31/269 - Train Accuracy:  0.441, Validation Accuracy:  0.442, Loss:  0.977
    Epoch   3 Batch   32/269 - Train Accuracy:  0.406, Validation Accuracy:  0.442, Loss:  1.746
    Epoch   3 Batch   33/269 - Train Accuracy:  0.447, Validation Accuracy:  0.443, Loss:  2.745
    Epoch   3 Batch   34/269 - Train Accuracy:  0.446, Validation Accuracy:  0.450, Loss:  1.583
    Epoch   3 Batch   35/269 - Train Accuracy:  0.476, Validation Accuracy:  0.497, Loss:  1.442
    Epoch   3 Batch   36/269 - Train Accuracy:  0.488, Validation Accuracy:  0.512, Loss:  1.160
    Epoch   3 Batch   37/269 - Train Accuracy:  0.464, Validation Accuracy:  0.491, Loss:  1.113
    Epoch   3 Batch   38/269 - Train Accuracy:  0.474, Validation Accuracy:  0.502, Loss:  1.203
    Epoch   3 Batch   39/269 - Train Accuracy:  0.506, Validation Accuracy:  0.528, Loss:  1.211
    Epoch   3 Batch   40/269 - Train Accuracy:  0.497, Validation Accuracy:  0.537, Loss:  1.128
    Epoch   3 Batch   41/269 - Train Accuracy:  0.518, Validation Accuracy:  0.542, Loss:  1.029
    Epoch   3 Batch   42/269 - Train Accuracy:  0.560, Validation Accuracy:  0.541, Loss:  0.954
    Epoch   3 Batch   43/269 - Train Accuracy:  0.517, Validation Accuracy:  0.545, Loss:  1.029
    Epoch   3 Batch   44/269 - Train Accuracy:  0.552, Validation Accuracy:  0.562, Loss:  0.941
    Epoch   3 Batch   45/269 - Train Accuracy:  0.522, Validation Accuracy:  0.568, Loss:  0.969
    Epoch   3 Batch   46/269 - Train Accuracy:  0.530, Validation Accuracy:  0.556, Loss:  0.943
    Epoch   3 Batch   47/269 - Train Accuracy:  0.575, Validation Accuracy:  0.549, Loss:  0.852
    Epoch   3 Batch   48/269 - Train Accuracy:  0.551, Validation Accuracy:  0.556, Loss:  0.856
    Epoch   3 Batch   49/269 - Train Accuracy:  0.540, Validation Accuracy:  0.566, Loss:  0.879
    Epoch   3 Batch   50/269 - Train Accuracy:  0.520, Validation Accuracy:  0.562, Loss:  0.881
    Epoch   3 Batch   51/269 - Train Accuracy:  0.549, Validation Accuracy:  0.563, Loss:  0.836
    Epoch   3 Batch   52/269 - Train Accuracy:  0.561, Validation Accuracy:  0.565, Loss:  0.796
    Epoch   3 Batch   53/269 - Train Accuracy:  0.547, Validation Accuracy:  0.573, Loss:  0.844
    Epoch   3 Batch   54/269 - Train Accuracy:  0.551, Validation Accuracy:  0.576, Loss:  0.821
    Epoch   3 Batch   55/269 - Train Accuracy:  0.564, Validation Accuracy:  0.572, Loss:  0.792
    Epoch   3 Batch   56/269 - Train Accuracy:  0.563, Validation Accuracy:  0.568, Loss:  0.789
    Epoch   3 Batch   57/269 - Train Accuracy:  0.568, Validation Accuracy:  0.573, Loss:  0.781
    Epoch   3 Batch   58/269 - Train Accuracy:  0.567, Validation Accuracy:  0.579, Loss:  0.764
    Epoch   3 Batch   59/269 - Train Accuracy:  0.577, Validation Accuracy:  0.584, Loss:  0.732
    Epoch   3 Batch   60/269 - Train Accuracy:  0.584, Validation Accuracy:  0.585, Loss:  0.723
    Epoch   3 Batch   61/269 - Train Accuracy:  0.596, Validation Accuracy:  0.590, Loss:  0.693
    Epoch   3 Batch   62/269 - Train Accuracy:  0.596, Validation Accuracy:  0.588, Loss:  0.707
    Epoch   3 Batch   63/269 - Train Accuracy:  0.569, Validation Accuracy:  0.590, Loss:  0.753
    Epoch   3 Batch   64/269 - Train Accuracy:  0.581, Validation Accuracy:  0.595, Loss:  0.716
    Epoch   3 Batch   65/269 - Train Accuracy:  0.587, Validation Accuracy:  0.597, Loss:  0.713
    Epoch   3 Batch   66/269 - Train Accuracy:  0.598, Validation Accuracy:  0.597, Loss:  0.684
    Epoch   3 Batch   67/269 - Train Accuracy:  0.590, Validation Accuracy:  0.593, Loss:  0.728
    Epoch   3 Batch   68/269 - Train Accuracy:  0.581, Validation Accuracy:  0.594, Loss:  0.704
    Epoch   3 Batch   69/269 - Train Accuracy:  0.556, Validation Accuracy:  0.598, Loss:  0.771
    Epoch   3 Batch   70/269 - Train Accuracy:  0.597, Validation Accuracy:  0.600, Loss:  0.691
    Epoch   3 Batch   71/269 - Train Accuracy:  0.582, Validation Accuracy:  0.603, Loss:  0.713
    Epoch   3 Batch   72/269 - Train Accuracy:  0.601, Validation Accuracy:  0.603, Loss:  0.670
    Epoch   3 Batch   73/269 - Train Accuracy:  0.603, Validation Accuracy:  0.604, Loss:  0.691
    Epoch   3 Batch   74/269 - Train Accuracy:  0.586, Validation Accuracy:  0.608, Loss:  0.687
    Epoch   3 Batch   75/269 - Train Accuracy:  0.598, Validation Accuracy:  0.608, Loss:  0.678
    Epoch   3 Batch   76/269 - Train Accuracy:  0.594, Validation Accuracy:  0.613, Loss:  0.678
    Epoch   3 Batch   77/269 - Train Accuracy:  0.632, Validation Accuracy:  0.616, Loss:  0.663
    Epoch   3 Batch   78/269 - Train Accuracy:  0.616, Validation Accuracy:  0.614, Loss:  0.660
    Epoch   3 Batch   79/269 - Train Accuracy:  0.602, Validation Accuracy:  0.610, Loss:  0.651
    Epoch   3 Batch   80/269 - Train Accuracy:  0.615, Validation Accuracy:  0.616, Loss:  0.656
    Epoch   3 Batch   81/269 - Train Accuracy:  0.630, Validation Accuracy:  0.617, Loss:  0.667
    Epoch   3 Batch   82/269 - Train Accuracy:  0.622, Validation Accuracy:  0.619, Loss:  0.628
    Epoch   3 Batch   83/269 - Train Accuracy:  0.606, Validation Accuracy:  0.607, Loss:  0.678
    Epoch   3 Batch   84/269 - Train Accuracy:  0.614, Validation Accuracy:  0.610, Loss:  0.637
    Epoch   3 Batch   85/269 - Train Accuracy:  0.610, Validation Accuracy:  0.611, Loss:  0.645
    Epoch   3 Batch   86/269 - Train Accuracy:  0.592, Validation Accuracy:  0.612, Loss:  0.628
    Epoch   3 Batch   87/269 - Train Accuracy:  0.586, Validation Accuracy:  0.610, Loss:  0.674
    Epoch   3 Batch   88/269 - Train Accuracy:  0.603, Validation Accuracy:  0.609, Loss:  0.628
    Epoch   3 Batch   89/269 - Train Accuracy:  0.641, Validation Accuracy:  0.621, Loss:  0.626
    Epoch   3 Batch   90/269 - Train Accuracy:  0.584, Validation Accuracy:  0.622, Loss:  0.674
    Epoch   3 Batch   91/269 - Train Accuracy:  0.619, Validation Accuracy:  0.618, Loss:  0.607
    Epoch   3 Batch   92/269 - Train Accuracy:  0.619, Validation Accuracy:  0.625, Loss:  0.617
    Epoch   3 Batch   93/269 - Train Accuracy:  0.629, Validation Accuracy:  0.628, Loss:  0.600
    Epoch   3 Batch   94/269 - Train Accuracy:  0.624, Validation Accuracy:  0.625, Loss:  0.626
    Epoch   3 Batch   95/269 - Train Accuracy:  0.613, Validation Accuracy:  0.625, Loss:  0.628
    Epoch   3 Batch   96/269 - Train Accuracy:  0.625, Validation Accuracy:  0.624, Loss:  0.612
    Epoch   3 Batch   97/269 - Train Accuracy:  0.603, Validation Accuracy:  0.626, Loss:  0.607
    Epoch   3 Batch   98/269 - Train Accuracy:  0.620, Validation Accuracy:  0.623, Loss:  0.612
    Epoch   3 Batch   99/269 - Train Accuracy:  0.619, Validation Accuracy:  0.624, Loss:  0.630
    Epoch   3 Batch  100/269 - Train Accuracy:  0.639, Validation Accuracy:  0.633, Loss:  0.608
    Epoch   3 Batch  101/269 - Train Accuracy:  0.612, Validation Accuracy:  0.637, Loss:  0.641
    Epoch   3 Batch  102/269 - Train Accuracy:  0.626, Validation Accuracy:  0.634, Loss:  0.600
    Epoch   3 Batch  103/269 - Train Accuracy:  0.629, Validation Accuracy:  0.636, Loss:  0.600
    Epoch   3 Batch  104/269 - Train Accuracy:  0.614, Validation Accuracy:  0.634, Loss:  0.601
    Epoch   3 Batch  105/269 - Train Accuracy:  0.619, Validation Accuracy:  0.630, Loss:  0.611
    Epoch   3 Batch  106/269 - Train Accuracy:  0.630, Validation Accuracy:  0.631, Loss:  0.586
    Epoch   3 Batch  107/269 - Train Accuracy:  0.612, Validation Accuracy:  0.637, Loss:  0.627
    Epoch   3 Batch  108/269 - Train Accuracy:  0.628, Validation Accuracy:  0.638, Loss:  0.590
    Epoch   3 Batch  109/269 - Train Accuracy:  0.610, Validation Accuracy:  0.629, Loss:  0.590
    Epoch   3 Batch  110/269 - Train Accuracy:  0.622, Validation Accuracy:  0.622, Loss:  0.581
    Epoch   3 Batch  111/269 - Train Accuracy:  0.597, Validation Accuracy:  0.625, Loss:  0.621
    Epoch   3 Batch  112/269 - Train Accuracy:  0.624, Validation Accuracy:  0.631, Loss:  0.596
    Epoch   3 Batch  113/269 - Train Accuracy:  0.633, Validation Accuracy:  0.626, Loss:  0.562
    Epoch   3 Batch  114/269 - Train Accuracy:  0.619, Validation Accuracy:  0.632, Loss:  0.578
    Epoch   3 Batch  115/269 - Train Accuracy:  0.614, Validation Accuracy:  0.631, Loss:  0.611
    Epoch   3 Batch  116/269 - Train Accuracy:  0.620, Validation Accuracy:  0.631, Loss:  0.587
    Epoch   3 Batch  117/269 - Train Accuracy:  0.620, Validation Accuracy:  0.631, Loss:  0.581
    Epoch   3 Batch  118/269 - Train Accuracy:  0.649, Validation Accuracy:  0.638, Loss:  0.569
    Epoch   3 Batch  119/269 - Train Accuracy:  0.621, Validation Accuracy:  0.645, Loss:  0.606
    Epoch   3 Batch  120/269 - Train Accuracy:  0.631, Validation Accuracy:  0.643, Loss:  0.597
    Epoch   3 Batch  121/269 - Train Accuracy:  0.646, Validation Accuracy:  0.644, Loss:  0.570
    Epoch   3 Batch  122/269 - Train Accuracy:  0.641, Validation Accuracy:  0.650, Loss:  0.569
    Epoch   3 Batch  123/269 - Train Accuracy:  0.614, Validation Accuracy:  0.648, Loss:  0.589
    Epoch   3 Batch  124/269 - Train Accuracy:  0.624, Validation Accuracy:  0.643, Loss:  0.561
    Epoch   3 Batch  125/269 - Train Accuracy:  0.631, Validation Accuracy:  0.646, Loss:  0.565
    Epoch   3 Batch  126/269 - Train Accuracy:  0.639, Validation Accuracy:  0.648, Loss:  0.568
    Epoch   3 Batch  127/269 - Train Accuracy:  0.621, Validation Accuracy:  0.648, Loss:  0.590
    Epoch   3 Batch  128/269 - Train Accuracy:  0.644, Validation Accuracy:  0.641, Loss:  0.566
    Epoch   3 Batch  129/269 - Train Accuracy:  0.629, Validation Accuracy:  0.645, Loss:  0.563
    Epoch   3 Batch  130/269 - Train Accuracy:  0.614, Validation Accuracy:  0.646, Loss:  0.580
    Epoch   3 Batch  131/269 - Train Accuracy:  0.622, Validation Accuracy:  0.643, Loss:  0.580
    Epoch   3 Batch  132/269 - Train Accuracy:  0.629, Validation Accuracy:  0.646, Loss:  0.563
    Epoch   3 Batch  133/269 - Train Accuracy:  0.643, Validation Accuracy:  0.645, Loss:  0.554
    Epoch   3 Batch  134/269 - Train Accuracy:  0.608, Validation Accuracy:  0.643, Loss:  0.581
    Epoch   3 Batch  135/269 - Train Accuracy:  0.620, Validation Accuracy:  0.647, Loss:  0.602
    Epoch   3 Batch  136/269 - Train Accuracy:  0.615, Validation Accuracy:  0.647, Loss:  0.600
    Epoch   3 Batch  137/269 - Train Accuracy:  0.642, Validation Accuracy:  0.650, Loss:  0.588
    Epoch   3 Batch  138/269 - Train Accuracy:  0.625, Validation Accuracy:  0.647, Loss:  0.574
    Epoch   3 Batch  139/269 - Train Accuracy:  0.654, Validation Accuracy:  0.650, Loss:  0.541
    Epoch   3 Batch  140/269 - Train Accuracy:  0.640, Validation Accuracy:  0.648, Loss:  0.573
    Epoch   3 Batch  141/269 - Train Accuracy:  0.645, Validation Accuracy:  0.648, Loss:  0.568
    Epoch   3 Batch  142/269 - Train Accuracy:  0.640, Validation Accuracy:  0.652, Loss:  0.538
    Epoch   3 Batch  143/269 - Train Accuracy:  0.649, Validation Accuracy:  0.654, Loss:  0.547
    Epoch   3 Batch  144/269 - Train Accuracy:  0.649, Validation Accuracy:  0.646, Loss:  0.518
    Epoch   3 Batch  145/269 - Train Accuracy:  0.643, Validation Accuracy:  0.651, Loss:  0.543
    Epoch   3 Batch  146/269 - Train Accuracy:  0.652, Validation Accuracy:  0.652, Loss:  0.540
    Epoch   3 Batch  147/269 - Train Accuracy:  0.653, Validation Accuracy:  0.654, Loss:  0.526
    Epoch   3 Batch  148/269 - Train Accuracy:  0.634, Validation Accuracy:  0.650, Loss:  0.551
    Epoch   3 Batch  149/269 - Train Accuracy:  0.640, Validation Accuracy:  0.646, Loss:  0.558
    Epoch   3 Batch  150/269 - Train Accuracy:  0.638, Validation Accuracy:  0.647, Loss:  0.541
    Epoch   3 Batch  151/269 - Train Accuracy:  0.669, Validation Accuracy:  0.646, Loss:  0.520
    Epoch   3 Batch  152/269 - Train Accuracy:  0.637, Validation Accuracy:  0.648, Loss:  0.545
    Epoch   3 Batch  153/269 - Train Accuracy:  0.659, Validation Accuracy:  0.654, Loss:  0.535
    Epoch   3 Batch  154/269 - Train Accuracy:  0.628, Validation Accuracy:  0.654, Loss:  0.548
    Epoch   3 Batch  155/269 - Train Accuracy:  0.659, Validation Accuracy:  0.654, Loss:  0.513
    Epoch   3 Batch  156/269 - Train Accuracy:  0.632, Validation Accuracy:  0.654, Loss:  0.576
    Epoch   3 Batch  157/269 - Train Accuracy:  0.651, Validation Accuracy:  0.655, Loss:  0.539
    Epoch   3 Batch  158/269 - Train Accuracy:  0.644, Validation Accuracy:  0.657, Loss:  0.537
    Epoch   3 Batch  159/269 - Train Accuracy:  0.651, Validation Accuracy:  0.655, Loss:  0.538
    Epoch   3 Batch  160/269 - Train Accuracy:  0.663, Validation Accuracy:  0.657, Loss:  0.534
    Epoch   3 Batch  161/269 - Train Accuracy:  0.642, Validation Accuracy:  0.662, Loss:  0.534
    Epoch   3 Batch  162/269 - Train Accuracy:  0.652, Validation Accuracy:  0.657, Loss:  0.537
    Epoch   3 Batch  163/269 - Train Accuracy:  0.646, Validation Accuracy:  0.657, Loss:  0.537
    Epoch   3 Batch  164/269 - Train Accuracy:  0.657, Validation Accuracy:  0.657, Loss:  0.532
    Epoch   3 Batch  165/269 - Train Accuracy:  0.617, Validation Accuracy:  0.660, Loss:  0.543
    Epoch   3 Batch  166/269 - Train Accuracy:  0.666, Validation Accuracy:  0.658, Loss:  0.510
    Epoch   3 Batch  167/269 - Train Accuracy:  0.644, Validation Accuracy:  0.652, Loss:  0.531
    Epoch   3 Batch  168/269 - Train Accuracy:  0.647, Validation Accuracy:  0.654, Loss:  0.535
    Epoch   3 Batch  169/269 - Train Accuracy:  0.645, Validation Accuracy:  0.658, Loss:  0.532
    Epoch   3 Batch  170/269 - Train Accuracy:  0.649, Validation Accuracy:  0.657, Loss:  0.524
    Epoch   3 Batch  171/269 - Train Accuracy:  0.634, Validation Accuracy:  0.659, Loss:  0.552
    Epoch   3 Batch  172/269 - Train Accuracy:  0.642, Validation Accuracy:  0.655, Loss:  0.535
    Epoch   3 Batch  173/269 - Train Accuracy:  0.648, Validation Accuracy:  0.656, Loss:  0.516
    Epoch   3 Batch  174/269 - Train Accuracy:  0.639, Validation Accuracy:  0.653, Loss:  0.529
    Epoch   3 Batch  175/269 - Train Accuracy:  0.655, Validation Accuracy:  0.655, Loss:  0.543
    Epoch   3 Batch  176/269 - Train Accuracy:  0.641, Validation Accuracy:  0.658, Loss:  0.559
    Epoch   3 Batch  177/269 - Train Accuracy:  0.643, Validation Accuracy:  0.657, Loss:  0.509
    Epoch   3 Batch  178/269 - Train Accuracy:  0.638, Validation Accuracy:  0.663, Loss:  0.542
    Epoch   3 Batch  179/269 - Train Accuracy:  0.661, Validation Accuracy:  0.663, Loss:  0.516
    Epoch   3 Batch  180/269 - Train Accuracy:  0.650, Validation Accuracy:  0.660, Loss:  0.514
    Epoch   3 Batch  181/269 - Train Accuracy:  0.638, Validation Accuracy:  0.662, Loss:  0.523
    Epoch   3 Batch  182/269 - Train Accuracy:  0.668, Validation Accuracy:  0.660, Loss:  0.520
    Epoch   3 Batch  183/269 - Train Accuracy:  0.703, Validation Accuracy:  0.655, Loss:  0.452
    Epoch   3 Batch  184/269 - Train Accuracy:  0.639, Validation Accuracy:  0.657, Loss:  0.537
    Epoch   3 Batch  185/269 - Train Accuracy:  0.663, Validation Accuracy:  0.657, Loss:  0.511
    Epoch   3 Batch  186/269 - Train Accuracy:  0.634, Validation Accuracy:  0.655, Loss:  0.537
    Epoch   3 Batch  187/269 - Train Accuracy:  0.654, Validation Accuracy:  0.656, Loss:  0.518
    Epoch   3 Batch  188/269 - Train Accuracy:  0.658, Validation Accuracy:  0.664, Loss:  0.507
    Epoch   3 Batch  189/269 - Train Accuracy:  0.661, Validation Accuracy:  0.664, Loss:  0.504
    Epoch   3 Batch  190/269 - Train Accuracy:  0.654, Validation Accuracy:  0.658, Loss:  0.505
    Epoch   3 Batch  191/269 - Train Accuracy:  0.662, Validation Accuracy:  0.662, Loss:  0.508
    Epoch   3 Batch  192/269 - Train Accuracy:  0.657, Validation Accuracy:  0.664, Loss:  0.514
    Epoch   3 Batch  193/269 - Train Accuracy:  0.648, Validation Accuracy:  0.659, Loss:  0.510
    Epoch   3 Batch  194/269 - Train Accuracy:  0.665, Validation Accuracy:  0.659, Loss:  0.517
    Epoch   3 Batch  195/269 - Train Accuracy:  0.654, Validation Accuracy:  0.657, Loss:  0.507
    Epoch   3 Batch  196/269 - Train Accuracy:  0.630, Validation Accuracy:  0.660, Loss:  0.507
    Epoch   3 Batch  197/269 - Train Accuracy:  0.632, Validation Accuracy:  0.665, Loss:  0.539
    Epoch   3 Batch  198/269 - Train Accuracy:  0.648, Validation Accuracy:  0.664, Loss:  0.538
    Epoch   3 Batch  199/269 - Train Accuracy:  0.639, Validation Accuracy:  0.660, Loss:  0.527
    Epoch   3 Batch  200/269 - Train Accuracy:  0.657, Validation Accuracy:  0.669, Loss:  0.530
    Epoch   3 Batch  201/269 - Train Accuracy:  0.657, Validation Accuracy:  0.668, Loss:  0.509
    Epoch   3 Batch  202/269 - Train Accuracy:  0.655, Validation Accuracy:  0.662, Loss:  0.507
    Epoch   3 Batch  203/269 - Train Accuracy:  0.643, Validation Accuracy:  0.662, Loss:  0.548
    Epoch   3 Batch  204/269 - Train Accuracy:  0.636, Validation Accuracy:  0.662, Loss:  0.534
    Epoch   3 Batch  205/269 - Train Accuracy:  0.639, Validation Accuracy:  0.657, Loss:  0.505
    Epoch   3 Batch  206/269 - Train Accuracy:  0.644, Validation Accuracy:  0.660, Loss:  0.526
    Epoch   3 Batch  207/269 - Train Accuracy:  0.670, Validation Accuracy:  0.661, Loss:  0.492
    Epoch   3 Batch  208/269 - Train Accuracy:  0.644, Validation Accuracy:  0.664, Loss:  0.527
    Epoch   3 Batch  209/269 - Train Accuracy:  0.647, Validation Accuracy:  0.663, Loss:  0.509
    Epoch   3 Batch  210/269 - Train Accuracy:  0.664, Validation Accuracy:  0.662, Loss:  0.494
    Epoch   3 Batch  211/269 - Train Accuracy:  0.661, Validation Accuracy:  0.665, Loss:  0.513
    Epoch   3 Batch  212/269 - Train Accuracy:  0.668, Validation Accuracy:  0.659, Loss:  0.503
    Epoch   3 Batch  213/269 - Train Accuracy:  0.659, Validation Accuracy:  0.665, Loss:  0.503
    Epoch   3 Batch  214/269 - Train Accuracy:  0.669, Validation Accuracy:  0.668, Loss:  0.505
    Epoch   3 Batch  215/269 - Train Accuracy:  0.686, Validation Accuracy:  0.664, Loss:  0.486
    Epoch   3 Batch  216/269 - Train Accuracy:  0.633, Validation Accuracy:  0.661, Loss:  0.541
    Epoch   3 Batch  217/269 - Train Accuracy:  0.632, Validation Accuracy:  0.667, Loss:  0.526
    Epoch   3 Batch  218/269 - Train Accuracy:  0.646, Validation Accuracy:  0.661, Loss:  0.518
    Epoch   3 Batch  219/269 - Train Accuracy:  0.643, Validation Accuracy:  0.657, Loss:  0.523
    Epoch   3 Batch  220/269 - Train Accuracy:  0.656, Validation Accuracy:  0.661, Loss:  0.468
    Epoch   3 Batch  221/269 - Train Accuracy:  0.683, Validation Accuracy:  0.665, Loss:  0.501
    Epoch   3 Batch  222/269 - Train Accuracy:  0.660, Validation Accuracy:  0.656, Loss:  0.491
    Epoch   3 Batch  223/269 - Train Accuracy:  0.650, Validation Accuracy:  0.653, Loss:  0.490
    Epoch   3 Batch  224/269 - Train Accuracy:  0.649, Validation Accuracy:  0.659, Loss:  0.518
    Epoch   3 Batch  225/269 - Train Accuracy:  0.651, Validation Accuracy:  0.661, Loss:  0.508
    Epoch   3 Batch  226/269 - Train Accuracy:  0.650, Validation Accuracy:  0.665, Loss:  0.502
    Epoch   3 Batch  227/269 - Train Accuracy:  0.713, Validation Accuracy:  0.667, Loss:  0.458
    Epoch   3 Batch  228/269 - Train Accuracy:  0.653, Validation Accuracy:  0.657, Loss:  0.489
    Epoch   3 Batch  229/269 - Train Accuracy:  0.653, Validation Accuracy:  0.649, Loss:  0.493
    Epoch   3 Batch  230/269 - Train Accuracy:  0.650, Validation Accuracy:  0.667, Loss:  0.495
    Epoch   3 Batch  231/269 - Train Accuracy:  0.630, Validation Accuracy:  0.662, Loss:  0.525
    Epoch   3 Batch  232/269 - Train Accuracy:  0.629, Validation Accuracy:  0.660, Loss:  0.525
    Epoch   3 Batch  233/269 - Train Accuracy:  0.666, Validation Accuracy:  0.664, Loss:  0.490
    Epoch   3 Batch  234/269 - Train Accuracy:  0.663, Validation Accuracy:  0.664, Loss:  0.488
    Epoch   3 Batch  235/269 - Train Accuracy:  0.670, Validation Accuracy:  0.677, Loss:  0.474
    Epoch   3 Batch  236/269 - Train Accuracy:  0.651, Validation Accuracy:  0.671, Loss:  0.486
    Epoch   3 Batch  237/269 - Train Accuracy:  0.640, Validation Accuracy:  0.658, Loss:  0.500
    Epoch   3 Batch  238/269 - Train Accuracy:  0.684, Validation Accuracy:  0.659, Loss:  0.482
    Epoch   3 Batch  239/269 - Train Accuracy:  0.671, Validation Accuracy:  0.674, Loss:  0.490
    Epoch   3 Batch  240/269 - Train Accuracy:  0.676, Validation Accuracy:  0.669, Loss:  0.443
    Epoch   3 Batch  241/269 - Train Accuracy:  0.660, Validation Accuracy:  0.667, Loss:  0.501
    Epoch   3 Batch  242/269 - Train Accuracy:  0.645, Validation Accuracy:  0.661, Loss:  0.485
    Epoch   3 Batch  243/269 - Train Accuracy:  0.683, Validation Accuracy:  0.662, Loss:  0.476
    Epoch   3 Batch  244/269 - Train Accuracy:  0.650, Validation Accuracy:  0.665, Loss:  0.494
    Epoch   3 Batch  245/269 - Train Accuracy:  0.650, Validation Accuracy:  0.662, Loss:  0.509
    Epoch   3 Batch  246/269 - Train Accuracy:  0.643, Validation Accuracy:  0.660, Loss:  0.487
    Epoch   3 Batch  247/269 - Train Accuracy:  0.656, Validation Accuracy:  0.662, Loss:  0.501
    Epoch   3 Batch  248/269 - Train Accuracy:  0.663, Validation Accuracy:  0.665, Loss:  0.475
    Epoch   3 Batch  249/269 - Train Accuracy:  0.688, Validation Accuracy:  0.669, Loss:  0.455
    Epoch   3 Batch  250/269 - Train Accuracy:  0.658, Validation Accuracy:  0.671, Loss:  0.487
    Epoch   3 Batch  251/269 - Train Accuracy:  0.678, Validation Accuracy:  0.669, Loss:  0.475
    Epoch   3 Batch  252/269 - Train Accuracy:  0.666, Validation Accuracy:  0.663, Loss:  0.476
    Epoch   3 Batch  253/269 - Train Accuracy:  0.658, Validation Accuracy:  0.665, Loss:  0.493
    Epoch   3 Batch  254/269 - Train Accuracy:  0.664, Validation Accuracy:  0.661, Loss:  0.482
    Epoch   3 Batch  255/269 - Train Accuracy:  0.688, Validation Accuracy:  0.666, Loss:  0.465
    Epoch   3 Batch  256/269 - Train Accuracy:  0.652, Validation Accuracy:  0.668, Loss:  0.485
    Epoch   3 Batch  257/269 - Train Accuracy:  0.638, Validation Accuracy:  0.663, Loss:  0.493
    Epoch   3 Batch  258/269 - Train Accuracy:  0.660, Validation Accuracy:  0.661, Loss:  0.483
    Epoch   3 Batch  259/269 - Train Accuracy:  0.664, Validation Accuracy:  0.664, Loss:  0.488
    Epoch   3 Batch  260/269 - Train Accuracy:  0.641, Validation Accuracy:  0.664, Loss:  0.506
    Epoch   3 Batch  261/269 - Train Accuracy:  0.642, Validation Accuracy:  0.665, Loss:  0.506
    Epoch   3 Batch  262/269 - Train Accuracy:  0.667, Validation Accuracy:  0.659, Loss:  0.481
    Epoch   3 Batch  263/269 - Train Accuracy:  0.662, Validation Accuracy:  0.663, Loss:  0.490
    Epoch   3 Batch  264/269 - Train Accuracy:  0.638, Validation Accuracy:  0.672, Loss:  0.511
    Epoch   3 Batch  265/269 - Train Accuracy:  0.653, Validation Accuracy:  0.667, Loss:  0.492
    Epoch   3 Batch  266/269 - Train Accuracy:  0.673, Validation Accuracy:  0.660, Loss:  0.473
    Epoch   3 Batch  267/269 - Train Accuracy:  0.670, Validation Accuracy:  0.670, Loss:  0.480
    Epoch   4 Batch    0/269 - Train Accuracy:  0.654, Validation Accuracy:  0.677, Loss:  0.499
    Epoch   4 Batch    1/269 - Train Accuracy:  0.643, Validation Accuracy:  0.675, Loss:  0.486
    Epoch   4 Batch    2/269 - Train Accuracy:  0.648, Validation Accuracy:  0.672, Loss:  0.481
    Epoch   4 Batch    3/269 - Train Accuracy:  0.658, Validation Accuracy:  0.673, Loss:  0.481
    Epoch   4 Batch    4/269 - Train Accuracy:  0.631, Validation Accuracy:  0.663, Loss:  0.501
    Epoch   4 Batch    5/269 - Train Accuracy:  0.641, Validation Accuracy:  0.666, Loss:  0.497
    Epoch   4 Batch    6/269 - Train Accuracy:  0.672, Validation Accuracy:  0.669, Loss:  0.464
    Epoch   4 Batch    7/269 - Train Accuracy:  0.658, Validation Accuracy:  0.660, Loss:  0.467
    Epoch   4 Batch    8/269 - Train Accuracy:  0.634, Validation Accuracy:  0.657, Loss:  0.499
    Epoch   4 Batch    9/269 - Train Accuracy:  0.637, Validation Accuracy:  0.669, Loss:  0.503
    Epoch   4 Batch   10/269 - Train Accuracy:  0.650, Validation Accuracy:  0.667, Loss:  0.575
    Epoch   4 Batch   11/269 - Train Accuracy:  0.603, Validation Accuracy:  0.634, Loss:  0.504
    Epoch   4 Batch   12/269 - Train Accuracy:  0.639, Validation Accuracy:  0.668, Loss:  0.607
    Epoch   4 Batch   13/269 - Train Accuracy:  0.655, Validation Accuracy:  0.634, Loss:  0.472
    Epoch   4 Batch   14/269 - Train Accuracy:  0.637, Validation Accuracy:  0.661, Loss:  0.523
    Epoch   4 Batch   15/269 - Train Accuracy:  0.647, Validation Accuracy:  0.663, Loss:  0.485
    Epoch   4 Batch   16/269 - Train Accuracy:  0.654, Validation Accuracy:  0.659, Loss:  0.497
    Epoch   4 Batch   17/269 - Train Accuracy:  0.653, Validation Accuracy:  0.659, Loss:  0.502
    Epoch   4 Batch   18/269 - Train Accuracy:  0.659, Validation Accuracy:  0.665, Loss:  0.501
    Epoch   4 Batch   19/269 - Train Accuracy:  0.691, Validation Accuracy:  0.663, Loss:  0.463
    Epoch   4 Batch   20/269 - Train Accuracy:  0.633, Validation Accuracy:  0.663, Loss:  0.505
    Epoch   4 Batch   21/269 - Train Accuracy:  0.645, Validation Accuracy:  0.660, Loss:  0.525
    Epoch   4 Batch   22/269 - Train Accuracy:  0.669, Validation Accuracy:  0.658, Loss:  0.473
    Epoch   4 Batch   23/269 - Train Accuracy:  0.668, Validation Accuracy:  0.666, Loss:  0.475
    Epoch   4 Batch   24/269 - Train Accuracy:  0.646, Validation Accuracy:  0.662, Loss:  0.499
    Epoch   4 Batch   25/269 - Train Accuracy:  0.635, Validation Accuracy:  0.674, Loss:  0.507
    Epoch   4 Batch   26/269 - Train Accuracy:  0.686, Validation Accuracy:  0.664, Loss:  0.444
    Epoch   4 Batch   27/269 - Train Accuracy:  0.663, Validation Accuracy:  0.665, Loss:  0.468
    Epoch   4 Batch   28/269 - Train Accuracy:  0.620, Validation Accuracy:  0.672, Loss:  0.508
    Epoch   4 Batch   29/269 - Train Accuracy:  0.658, Validation Accuracy:  0.674, Loss:  0.486
    Epoch   4 Batch   30/269 - Train Accuracy:  0.664, Validation Accuracy:  0.672, Loss:  0.463
    Epoch   4 Batch   31/269 - Train Accuracy:  0.675, Validation Accuracy:  0.675, Loss:  0.452
    Epoch   4 Batch   32/269 - Train Accuracy:  0.661, Validation Accuracy:  0.680, Loss:  0.455
    Epoch   4 Batch   33/269 - Train Accuracy:  0.684, Validation Accuracy:  0.676, Loss:  0.445
    Epoch   4 Batch   34/269 - Train Accuracy:  0.688, Validation Accuracy:  0.679, Loss:  0.460
    Epoch   4 Batch   35/269 - Train Accuracy:  0.665, Validation Accuracy:  0.678, Loss:  0.487
    Epoch   4 Batch   36/269 - Train Accuracy:  0.660, Validation Accuracy:  0.671, Loss:  0.458
    Epoch   4 Batch   37/269 - Train Accuracy:  0.683, Validation Accuracy:  0.677, Loss:  0.450
    Epoch   4 Batch   38/269 - Train Accuracy:  0.657, Validation Accuracy:  0.671, Loss:  0.452
    Epoch   4 Batch   39/269 - Train Accuracy:  0.663, Validation Accuracy:  0.678, Loss:  0.460
    Epoch   4 Batch   40/269 - Train Accuracy:  0.654, Validation Accuracy:  0.678, Loss:  0.482
    Epoch   4 Batch   41/269 - Train Accuracy:  0.669, Validation Accuracy:  0.683, Loss:  0.466
    Epoch   4 Batch   42/269 - Train Accuracy:  0.691, Validation Accuracy:  0.681, Loss:  0.434
    Epoch   4 Batch   43/269 - Train Accuracy:  0.660, Validation Accuracy:  0.676, Loss:  0.467
    Epoch   4 Batch   44/269 - Train Accuracy:  0.675, Validation Accuracy:  0.674, Loss:  0.457
    Epoch   4 Batch   45/269 - Train Accuracy:  0.655, Validation Accuracy:  0.679, Loss:  0.472
    Epoch   4 Batch   46/269 - Train Accuracy:  0.678, Validation Accuracy:  0.680, Loss:  0.462
    Epoch   4 Batch   47/269 - Train Accuracy:  0.694, Validation Accuracy:  0.679, Loss:  0.419
    Epoch   4 Batch   48/269 - Train Accuracy:  0.675, Validation Accuracy:  0.680, Loss:  0.438
    Epoch   4 Batch   49/269 - Train Accuracy:  0.654, Validation Accuracy:  0.673, Loss:  0.463
    Epoch   4 Batch   50/269 - Train Accuracy:  0.655, Validation Accuracy:  0.674, Loss:  0.475
    Epoch   4 Batch   51/269 - Train Accuracy:  0.668, Validation Accuracy:  0.679, Loss:  0.444
    Epoch   4 Batch   52/269 - Train Accuracy:  0.683, Validation Accuracy:  0.679, Loss:  0.430
    Epoch   4 Batch   53/269 - Train Accuracy:  0.660, Validation Accuracy:  0.679, Loss:  0.475
    Epoch   4 Batch   54/269 - Train Accuracy:  0.685, Validation Accuracy:  0.680, Loss:  0.462
    Epoch   4 Batch   55/269 - Train Accuracy:  0.675, Validation Accuracy:  0.681, Loss:  0.446
    Epoch   4 Batch   56/269 - Train Accuracy:  0.688, Validation Accuracy:  0.677, Loss:  0.443
    Epoch   4 Batch   57/269 - Train Accuracy:  0.682, Validation Accuracy:  0.675, Loss:  0.454
    Epoch   4 Batch   58/269 - Train Accuracy:  0.684, Validation Accuracy:  0.682, Loss:  0.437
    Epoch   4 Batch   59/269 - Train Accuracy:  0.684, Validation Accuracy:  0.680, Loss:  0.424
    Epoch   4 Batch   60/269 - Train Accuracy:  0.686, Validation Accuracy:  0.679, Loss:  0.425
    Epoch   4 Batch   61/269 - Train Accuracy:  0.686, Validation Accuracy:  0.680, Loss:  0.414
    Epoch   4 Batch   62/269 - Train Accuracy:  0.689, Validation Accuracy:  0.675, Loss:  0.432
    Epoch   4 Batch   63/269 - Train Accuracy:  0.679, Validation Accuracy:  0.683, Loss:  0.452
    Epoch   4 Batch   64/269 - Train Accuracy:  0.676, Validation Accuracy:  0.684, Loss:  0.425
    Epoch   4 Batch   65/269 - Train Accuracy:  0.668, Validation Accuracy:  0.686, Loss:  0.438
    Epoch   4 Batch   66/269 - Train Accuracy:  0.668, Validation Accuracy:  0.669, Loss:  0.418
    Epoch   4 Batch   67/269 - Train Accuracy:  0.682, Validation Accuracy:  0.671, Loss:  0.456
    Epoch   4 Batch   68/269 - Train Accuracy:  0.663, Validation Accuracy:  0.673, Loss:  0.441
    Epoch   4 Batch   69/269 - Train Accuracy:  0.640, Validation Accuracy:  0.674, Loss:  0.476
    Epoch   4 Batch   70/269 - Train Accuracy:  0.683, Validation Accuracy:  0.672, Loss:  0.440
    Epoch   4 Batch   71/269 - Train Accuracy:  0.678, Validation Accuracy:  0.676, Loss:  0.458
    Epoch   4 Batch   72/269 - Train Accuracy:  0.680, Validation Accuracy:  0.678, Loss:  0.434
    Epoch   4 Batch   73/269 - Train Accuracy:  0.678, Validation Accuracy:  0.677, Loss:  0.441
    Epoch   4 Batch   74/269 - Train Accuracy:  0.678, Validation Accuracy:  0.678, Loss:  0.434
    Epoch   4 Batch   75/269 - Train Accuracy:  0.684, Validation Accuracy:  0.681, Loss:  0.427
    Epoch   4 Batch   76/269 - Train Accuracy:  0.670, Validation Accuracy:  0.683, Loss:  0.436
    Epoch   4 Batch   77/269 - Train Accuracy:  0.703, Validation Accuracy:  0.684, Loss:  0.432
    Epoch   4 Batch   78/269 - Train Accuracy:  0.692, Validation Accuracy:  0.682, Loss:  0.427
    Epoch   4 Batch   79/269 - Train Accuracy:  0.692, Validation Accuracy:  0.683, Loss:  0.429
    Epoch   4 Batch   80/269 - Train Accuracy:  0.694, Validation Accuracy:  0.686, Loss:  0.436
    Epoch   4 Batch   81/269 - Train Accuracy:  0.687, Validation Accuracy:  0.686, Loss:  0.441
    Epoch   4 Batch   82/269 - Train Accuracy:  0.694, Validation Accuracy:  0.677, Loss:  0.406
    Epoch   4 Batch   83/269 - Train Accuracy:  0.682, Validation Accuracy:  0.683, Loss:  0.445
    Epoch   4 Batch   84/269 - Train Accuracy:  0.694, Validation Accuracy:  0.681, Loss:  0.422
    Epoch   4 Batch   85/269 - Train Accuracy:  0.691, Validation Accuracy:  0.683, Loss:  0.427
    Epoch   4 Batch   86/269 - Train Accuracy:  0.666, Validation Accuracy:  0.682, Loss:  0.414
    Epoch   4 Batch   87/269 - Train Accuracy:  0.653, Validation Accuracy:  0.686, Loss:  0.456
    Epoch   4 Batch   88/269 - Train Accuracy:  0.676, Validation Accuracy:  0.682, Loss:  0.423
    Epoch   4 Batch   89/269 - Train Accuracy:  0.698, Validation Accuracy:  0.684, Loss:  0.422
    Epoch   4 Batch   90/269 - Train Accuracy:  0.653, Validation Accuracy:  0.679, Loss:  0.451
    Epoch   4 Batch   91/269 - Train Accuracy:  0.693, Validation Accuracy:  0.683, Loss:  0.409
    Epoch   4 Batch   92/269 - Train Accuracy:  0.676, Validation Accuracy:  0.680, Loss:  0.415
    Epoch   4 Batch   93/269 - Train Accuracy:  0.692, Validation Accuracy:  0.680, Loss:  0.411
    Epoch   4 Batch   94/269 - Train Accuracy:  0.678, Validation Accuracy:  0.677, Loss:  0.432
    Epoch   4 Batch   95/269 - Train Accuracy:  0.687, Validation Accuracy:  0.682, Loss:  0.428
    Epoch   4 Batch   96/269 - Train Accuracy:  0.683, Validation Accuracy:  0.680, Loss:  0.422
    Epoch   4 Batch   97/269 - Train Accuracy:  0.681, Validation Accuracy:  0.688, Loss:  0.423
    Epoch   4 Batch   98/269 - Train Accuracy:  0.691, Validation Accuracy:  0.683, Loss:  0.425
    Epoch   4 Batch   99/269 - Train Accuracy:  0.685, Validation Accuracy:  0.691, Loss:  0.435
    Epoch   4 Batch  100/269 - Train Accuracy:  0.717, Validation Accuracy:  0.689, Loss:  0.416
    Epoch   4 Batch  101/269 - Train Accuracy:  0.662, Validation Accuracy:  0.684, Loss:  0.448
    Epoch   4 Batch  102/269 - Train Accuracy:  0.689, Validation Accuracy:  0.691, Loss:  0.420
    Epoch   4 Batch  103/269 - Train Accuracy:  0.686, Validation Accuracy:  0.692, Loss:  0.418
    Epoch   4 Batch  104/269 - Train Accuracy:  0.670, Validation Accuracy:  0.690, Loss:  0.419
    Epoch   4 Batch  105/269 - Train Accuracy:  0.682, Validation Accuracy:  0.681, Loss:  0.429
    Epoch   4 Batch  106/269 - Train Accuracy:  0.674, Validation Accuracy:  0.683, Loss:  0.404
    Epoch   4 Batch  107/269 - Train Accuracy:  0.662, Validation Accuracy:  0.678, Loss:  0.435
    Epoch   4 Batch  108/269 - Train Accuracy:  0.680, Validation Accuracy:  0.678, Loss:  0.423
    Epoch   4 Batch  109/269 - Train Accuracy:  0.668, Validation Accuracy:  0.677, Loss:  0.418
    Epoch   4 Batch  110/269 - Train Accuracy:  0.695, Validation Accuracy:  0.681, Loss:  0.410
    Epoch   4 Batch  111/269 - Train Accuracy:  0.664, Validation Accuracy:  0.681, Loss:  0.439
    Epoch   4 Batch  112/269 - Train Accuracy:  0.680, Validation Accuracy:  0.680, Loss:  0.418
    Epoch   4 Batch  113/269 - Train Accuracy:  0.692, Validation Accuracy:  0.683, Loss:  0.399
    Epoch   4 Batch  114/269 - Train Accuracy:  0.695, Validation Accuracy:  0.684, Loss:  0.413
    Epoch   4 Batch  115/269 - Train Accuracy:  0.674, Validation Accuracy:  0.688, Loss:  0.440
    Epoch   4 Batch  116/269 - Train Accuracy:  0.686, Validation Accuracy:  0.694, Loss:  0.418
    Epoch   4 Batch  117/269 - Train Accuracy:  0.677, Validation Accuracy:  0.693, Loss:  0.410
    Epoch   4 Batch  118/269 - Train Accuracy:  0.714, Validation Accuracy:  0.691, Loss:  0.401
    Epoch   4 Batch  119/269 - Train Accuracy:  0.672, Validation Accuracy:  0.687, Loss:  0.431
    Epoch   4 Batch  120/269 - Train Accuracy:  0.699, Validation Accuracy:  0.688, Loss:  0.421
    Epoch   4 Batch  121/269 - Train Accuracy:  0.690, Validation Accuracy:  0.684, Loss:  0.410
    Epoch   4 Batch  122/269 - Train Accuracy:  0.702, Validation Accuracy:  0.689, Loss:  0.407
    Epoch   4 Batch  123/269 - Train Accuracy:  0.687, Validation Accuracy:  0.686, Loss:  0.425
    Epoch   4 Batch  124/269 - Train Accuracy:  0.678, Validation Accuracy:  0.690, Loss:  0.404
    Epoch   4 Batch  125/269 - Train Accuracy:  0.685, Validation Accuracy:  0.691, Loss:  0.402
    Epoch   4 Batch  126/269 - Train Accuracy:  0.673, Validation Accuracy:  0.680, Loss:  0.410
    Epoch   4 Batch  127/269 - Train Accuracy:  0.686, Validation Accuracy:  0.670, Loss:  0.420
    Epoch   4 Batch  128/269 - Train Accuracy:  0.697, Validation Accuracy:  0.679, Loss:  0.409
    Epoch   4 Batch  129/269 - Train Accuracy:  0.685, Validation Accuracy:  0.689, Loss:  0.399
    Epoch   4 Batch  130/269 - Train Accuracy:  0.661, Validation Accuracy:  0.684, Loss:  0.421
    Epoch   4 Batch  131/269 - Train Accuracy:  0.684, Validation Accuracy:  0.693, Loss:  0.424
    Epoch   4 Batch  132/269 - Train Accuracy:  0.696, Validation Accuracy:  0.688, Loss:  0.413
    Epoch   4 Batch  133/269 - Train Accuracy:  0.690, Validation Accuracy:  0.686, Loss:  0.393
    Epoch   4 Batch  134/269 - Train Accuracy:  0.668, Validation Accuracy:  0.685, Loss:  0.420
    Epoch   4 Batch  135/269 - Train Accuracy:  0.674, Validation Accuracy:  0.695, Loss:  0.427
    Epoch   4 Batch  136/269 - Train Accuracy:  0.674, Validation Accuracy:  0.695, Loss:  0.430
    Epoch   4 Batch  137/269 - Train Accuracy:  0.688, Validation Accuracy:  0.698, Loss:  0.426
    Epoch   4 Batch  138/269 - Train Accuracy:  0.695, Validation Accuracy:  0.696, Loss:  0.420
    Epoch   4 Batch  139/269 - Train Accuracy:  0.698, Validation Accuracy:  0.698, Loss:  0.391
    Epoch   4 Batch  140/269 - Train Accuracy:  0.706, Validation Accuracy:  0.695, Loss:  0.418
    Epoch   4 Batch  141/269 - Train Accuracy:  0.711, Validation Accuracy:  0.701, Loss:  0.415
    Epoch   4 Batch  142/269 - Train Accuracy:  0.689, Validation Accuracy:  0.697, Loss:  0.382
    Epoch   4 Batch  143/269 - Train Accuracy:  0.710, Validation Accuracy:  0.699, Loss:  0.397
    Epoch   4 Batch  144/269 - Train Accuracy:  0.705, Validation Accuracy:  0.697, Loss:  0.376
    Epoch   4 Batch  145/269 - Train Accuracy:  0.702, Validation Accuracy:  0.704, Loss:  0.397
    Epoch   4 Batch  146/269 - Train Accuracy:  0.698, Validation Accuracy:  0.695, Loss:  0.391
    Epoch   4 Batch  147/269 - Train Accuracy:  0.714, Validation Accuracy:  0.684, Loss:  0.382
    Epoch   4 Batch  148/269 - Train Accuracy:  0.682, Validation Accuracy:  0.694, Loss:  0.406
    Epoch   4 Batch  149/269 - Train Accuracy:  0.681, Validation Accuracy:  0.692, Loss:  0.404
    Epoch   4 Batch  150/269 - Train Accuracy:  0.701, Validation Accuracy:  0.686, Loss:  0.399
    Epoch   4 Batch  151/269 - Train Accuracy:  0.711, Validation Accuracy:  0.683, Loss:  0.384
    Epoch   4 Batch  152/269 - Train Accuracy:  0.696, Validation Accuracy:  0.692, Loss:  0.403
    Epoch   4 Batch  153/269 - Train Accuracy:  0.715, Validation Accuracy:  0.693, Loss:  0.387
    Epoch   4 Batch  154/269 - Train Accuracy:  0.695, Validation Accuracy:  0.698, Loss:  0.400
    Epoch   4 Batch  155/269 - Train Accuracy:  0.718, Validation Accuracy:  0.697, Loss:  0.381
    Epoch   4 Batch  156/269 - Train Accuracy:  0.682, Validation Accuracy:  0.697, Loss:  0.413
    Epoch   4 Batch  157/269 - Train Accuracy:  0.700, Validation Accuracy:  0.698, Loss:  0.390
    Epoch   4 Batch  158/269 - Train Accuracy:  0.702, Validation Accuracy:  0.698, Loss:  0.405
    Epoch   4 Batch  159/269 - Train Accuracy:  0.695, Validation Accuracy:  0.703, Loss:  0.393
    Epoch   4 Batch  160/269 - Train Accuracy:  0.704, Validation Accuracy:  0.699, Loss:  0.392
    Epoch   4 Batch  161/269 - Train Accuracy:  0.701, Validation Accuracy:  0.705, Loss:  0.386
    Epoch   4 Batch  162/269 - Train Accuracy:  0.706, Validation Accuracy:  0.702, Loss:  0.387
    Epoch   4 Batch  163/269 - Train Accuracy:  0.720, Validation Accuracy:  0.700, Loss:  0.394
    Epoch   4 Batch  164/269 - Train Accuracy:  0.715, Validation Accuracy:  0.692, Loss:  0.391
    Epoch   4 Batch  165/269 - Train Accuracy:  0.688, Validation Accuracy:  0.686, Loss:  0.400
    Epoch   4 Batch  166/269 - Train Accuracy:  0.710, Validation Accuracy:  0.696, Loss:  0.380
    Epoch   4 Batch  167/269 - Train Accuracy:  0.712, Validation Accuracy:  0.701, Loss:  0.391
    Epoch   4 Batch  168/269 - Train Accuracy:  0.714, Validation Accuracy:  0.697, Loss:  0.389
    Epoch   4 Batch  169/269 - Train Accuracy:  0.697, Validation Accuracy:  0.697, Loss:  0.390
    Epoch   4 Batch  170/269 - Train Accuracy:  0.710, Validation Accuracy:  0.694, Loss:  0.378
    Epoch   4 Batch  171/269 - Train Accuracy:  0.696, Validation Accuracy:  0.693, Loss:  0.405
    Epoch   4 Batch  172/269 - Train Accuracy:  0.704, Validation Accuracy:  0.700, Loss:  0.401
    Epoch   4 Batch  173/269 - Train Accuracy:  0.713, Validation Accuracy:  0.695, Loss:  0.382
    Epoch   4 Batch  174/269 - Train Accuracy:  0.697, Validation Accuracy:  0.696, Loss:  0.391
    Epoch   4 Batch  175/269 - Train Accuracy:  0.708, Validation Accuracy:  0.700, Loss:  0.407
    Epoch   4 Batch  176/269 - Train Accuracy:  0.685, Validation Accuracy:  0.701, Loss:  0.413
    Epoch   4 Batch  177/269 - Train Accuracy:  0.703, Validation Accuracy:  0.700, Loss:  0.370
    Epoch   4 Batch  178/269 - Train Accuracy:  0.697, Validation Accuracy:  0.702, Loss:  0.393
    Epoch   4 Batch  179/269 - Train Accuracy:  0.698, Validation Accuracy:  0.704, Loss:  0.382
    Epoch   4 Batch  180/269 - Train Accuracy:  0.713, Validation Accuracy:  0.703, Loss:  0.373
    Epoch   4 Batch  181/269 - Train Accuracy:  0.700, Validation Accuracy:  0.706, Loss:  0.376
    Epoch   4 Batch  182/269 - Train Accuracy:  0.724, Validation Accuracy:  0.699, Loss:  0.377
    Epoch   4 Batch  183/269 - Train Accuracy:  0.746, Validation Accuracy:  0.702, Loss:  0.328
    Epoch   4 Batch  184/269 - Train Accuracy:  0.707, Validation Accuracy:  0.706, Loss:  0.393
    Epoch   4 Batch  185/269 - Train Accuracy:  0.730, Validation Accuracy:  0.704, Loss:  0.378
    Epoch   4 Batch  186/269 - Train Accuracy:  0.707, Validation Accuracy:  0.704, Loss:  0.388
    Epoch   4 Batch  187/269 - Train Accuracy:  0.708, Validation Accuracy:  0.709, Loss:  0.377
    Epoch   4 Batch  188/269 - Train Accuracy:  0.721, Validation Accuracy:  0.712, Loss:  0.372
    Epoch   4 Batch  189/269 - Train Accuracy:  0.718, Validation Accuracy:  0.710, Loss:  0.366
    Epoch   4 Batch  190/269 - Train Accuracy:  0.722, Validation Accuracy:  0.711, Loss:  0.366
    Epoch   4 Batch  191/269 - Train Accuracy:  0.721, Validation Accuracy:  0.706, Loss:  0.372
    Epoch   4 Batch  192/269 - Train Accuracy:  0.726, Validation Accuracy:  0.713, Loss:  0.377
    Epoch   4 Batch  193/269 - Train Accuracy:  0.720, Validation Accuracy:  0.712, Loss:  0.372
    Epoch   4 Batch  194/269 - Train Accuracy:  0.718, Validation Accuracy:  0.708, Loss:  0.373
    Epoch   4 Batch  195/269 - Train Accuracy:  0.725, Validation Accuracy:  0.710, Loss:  0.373
    Epoch   4 Batch  196/269 - Train Accuracy:  0.691, Validation Accuracy:  0.698, Loss:  0.367
    Epoch   4 Batch  197/269 - Train Accuracy:  0.701, Validation Accuracy:  0.704, Loss:  0.393
    Epoch   4 Batch  198/269 - Train Accuracy:  0.696, Validation Accuracy:  0.701, Loss:  0.394
    Epoch   4 Batch  199/269 - Train Accuracy:  0.711, Validation Accuracy:  0.696, Loss:  0.387
    Epoch   4 Batch  200/269 - Train Accuracy:  0.719, Validation Accuracy:  0.702, Loss:  0.386
    Epoch   4 Batch  201/269 - Train Accuracy:  0.728, Validation Accuracy:  0.702, Loss:  0.376
    Epoch   4 Batch  202/269 - Train Accuracy:  0.709, Validation Accuracy:  0.709, Loss:  0.370
    Epoch   4 Batch  203/269 - Train Accuracy:  0.721, Validation Accuracy:  0.705, Loss:  0.394
    Epoch   4 Batch  204/269 - Train Accuracy:  0.701, Validation Accuracy:  0.708, Loss:  0.388
    Epoch   4 Batch  205/269 - Train Accuracy:  0.723, Validation Accuracy:  0.705, Loss:  0.364
    Epoch   4 Batch  206/269 - Train Accuracy:  0.722, Validation Accuracy:  0.704, Loss:  0.386
    Epoch   4 Batch  207/269 - Train Accuracy:  0.728, Validation Accuracy:  0.706, Loss:  0.352
    Epoch   4 Batch  208/269 - Train Accuracy:  0.713, Validation Accuracy:  0.708, Loss:  0.379
    Epoch   4 Batch  209/269 - Train Accuracy:  0.726, Validation Accuracy:  0.715, Loss:  0.365
    Epoch   4 Batch  210/269 - Train Accuracy:  0.731, Validation Accuracy:  0.714, Loss:  0.356
    Epoch   4 Batch  211/269 - Train Accuracy:  0.714, Validation Accuracy:  0.711, Loss:  0.371
    Epoch   4 Batch  212/269 - Train Accuracy:  0.724, Validation Accuracy:  0.711, Loss:  0.369
    Epoch   4 Batch  213/269 - Train Accuracy:  0.722, Validation Accuracy:  0.715, Loss:  0.359
    Epoch   4 Batch  214/269 - Train Accuracy:  0.723, Validation Accuracy:  0.712, Loss:  0.367
    Epoch   4 Batch  215/269 - Train Accuracy:  0.739, Validation Accuracy:  0.708, Loss:  0.347
    Epoch   4 Batch  216/269 - Train Accuracy:  0.688, Validation Accuracy:  0.709, Loss:  0.393
    Epoch   4 Batch  217/269 - Train Accuracy:  0.698, Validation Accuracy:  0.713, Loss:  0.382
    Epoch   4 Batch  218/269 - Train Accuracy:  0.726, Validation Accuracy:  0.712, Loss:  0.370
    Epoch   4 Batch  219/269 - Train Accuracy:  0.699, Validation Accuracy:  0.711, Loss:  0.378
    Epoch   4 Batch  220/269 - Train Accuracy:  0.728, Validation Accuracy:  0.711, Loss:  0.345
    Epoch   4 Batch  221/269 - Train Accuracy:  0.746, Validation Accuracy:  0.715, Loss:  0.359
    Epoch   4 Batch  222/269 - Train Accuracy:  0.730, Validation Accuracy:  0.712, Loss:  0.353
    Epoch   4 Batch  223/269 - Train Accuracy:  0.730, Validation Accuracy:  0.711, Loss:  0.358
    Epoch   4 Batch  224/269 - Train Accuracy:  0.730, Validation Accuracy:  0.712, Loss:  0.378
    Epoch   4 Batch  225/269 - Train Accuracy:  0.716, Validation Accuracy:  0.709, Loss:  0.367
    Epoch   4 Batch  226/269 - Train Accuracy:  0.730, Validation Accuracy:  0.718, Loss:  0.367
    Epoch   4 Batch  227/269 - Train Accuracy:  0.767, Validation Accuracy:  0.711, Loss:  0.329
    Epoch   4 Batch  228/269 - Train Accuracy:  0.720, Validation Accuracy:  0.706, Loss:  0.360
    Epoch   4 Batch  229/269 - Train Accuracy:  0.728, Validation Accuracy:  0.707, Loss:  0.354
    Epoch   4 Batch  230/269 - Train Accuracy:  0.726, Validation Accuracy:  0.714, Loss:  0.358
    Epoch   4 Batch  231/269 - Train Accuracy:  0.688, Validation Accuracy:  0.704, Loss:  0.386
    Epoch   4 Batch  232/269 - Train Accuracy:  0.697, Validation Accuracy:  0.700, Loss:  0.376
    Epoch   4 Batch  233/269 - Train Accuracy:  0.741, Validation Accuracy:  0.715, Loss:  0.365
    Epoch   4 Batch  234/269 - Train Accuracy:  0.727, Validation Accuracy:  0.714, Loss:  0.359
    Epoch   4 Batch  235/269 - Train Accuracy:  0.729, Validation Accuracy:  0.713, Loss:  0.352
    Epoch   4 Batch  236/269 - Train Accuracy:  0.709, Validation Accuracy:  0.714, Loss:  0.354
    Epoch   4 Batch  237/269 - Train Accuracy:  0.710, Validation Accuracy:  0.721, Loss:  0.364
    Epoch   4 Batch  238/269 - Train Accuracy:  0.729, Validation Accuracy:  0.721, Loss:  0.349
    Epoch   4 Batch  239/269 - Train Accuracy:  0.734, Validation Accuracy:  0.717, Loss:  0.360
    Epoch   4 Batch  240/269 - Train Accuracy:  0.750, Validation Accuracy:  0.709, Loss:  0.317
    Epoch   4 Batch  241/269 - Train Accuracy:  0.724, Validation Accuracy:  0.719, Loss:  0.372
    Epoch   4 Batch  242/269 - Train Accuracy:  0.719, Validation Accuracy:  0.709, Loss:  0.354
    Epoch   4 Batch  243/269 - Train Accuracy:  0.753, Validation Accuracy:  0.708, Loss:  0.349
    Epoch   4 Batch  244/269 - Train Accuracy:  0.723, Validation Accuracy:  0.713, Loss:  0.363
    Epoch   4 Batch  245/269 - Train Accuracy:  0.707, Validation Accuracy:  0.704, Loss:  0.369
    Epoch   4 Batch  246/269 - Train Accuracy:  0.707, Validation Accuracy:  0.713, Loss:  0.360
    Epoch   4 Batch  247/269 - Train Accuracy:  0.732, Validation Accuracy:  0.712, Loss:  0.374
    Epoch   4 Batch  248/269 - Train Accuracy:  0.717, Validation Accuracy:  0.717, Loss:  0.345
    Epoch   4 Batch  249/269 - Train Accuracy:  0.752, Validation Accuracy:  0.709, Loss:  0.332
    Epoch   4 Batch  250/269 - Train Accuracy:  0.723, Validation Accuracy:  0.711, Loss:  0.358
    Epoch   4 Batch  251/269 - Train Accuracy:  0.761, Validation Accuracy:  0.714, Loss:  0.340
    Epoch   4 Batch  252/269 - Train Accuracy:  0.728, Validation Accuracy:  0.724, Loss:  0.343
    Epoch   4 Batch  253/269 - Train Accuracy:  0.720, Validation Accuracy:  0.713, Loss:  0.356
    Epoch   4 Batch  254/269 - Train Accuracy:  0.756, Validation Accuracy:  0.718, Loss:  0.347
    Epoch   4 Batch  255/269 - Train Accuracy:  0.754, Validation Accuracy:  0.723, Loss:  0.330
    Epoch   4 Batch  256/269 - Train Accuracy:  0.715, Validation Accuracy:  0.725, Loss:  0.354
    Epoch   4 Batch  257/269 - Train Accuracy:  0.709, Validation Accuracy:  0.718, Loss:  0.364
    Epoch   4 Batch  258/269 - Train Accuracy:  0.737, Validation Accuracy:  0.722, Loss:  0.346
    Epoch   4 Batch  259/269 - Train Accuracy:  0.738, Validation Accuracy:  0.721, Loss:  0.339
    Epoch   4 Batch  260/269 - Train Accuracy:  0.720, Validation Accuracy:  0.727, Loss:  0.367
    Epoch   4 Batch  261/269 - Train Accuracy:  0.720, Validation Accuracy:  0.726, Loss:  0.350
    Epoch   4 Batch  262/269 - Train Accuracy:  0.738, Validation Accuracy:  0.724, Loss:  0.337
    Epoch   4 Batch  263/269 - Train Accuracy:  0.725, Validation Accuracy:  0.730, Loss:  0.354
    Epoch   4 Batch  264/269 - Train Accuracy:  0.717, Validation Accuracy:  0.727, Loss:  0.362
    Epoch   4 Batch  265/269 - Train Accuracy:  0.731, Validation Accuracy:  0.727, Loss:  0.353
    Epoch   4 Batch  266/269 - Train Accuracy:  0.737, Validation Accuracy:  0.730, Loss:  0.334
    Epoch   4 Batch  267/269 - Train Accuracy:  0.737, Validation Accuracy:  0.722, Loss:  0.348
    Epoch   5 Batch    0/269 - Train Accuracy:  0.722, Validation Accuracy:  0.716, Loss:  0.359
    Epoch   5 Batch    1/269 - Train Accuracy:  0.716, Validation Accuracy:  0.724, Loss:  0.349
    Epoch   5 Batch    2/269 - Train Accuracy:  0.733, Validation Accuracy:  0.724, Loss:  0.342
    Epoch   5 Batch    3/269 - Train Accuracy:  0.720, Validation Accuracy:  0.724, Loss:  0.346
    Epoch   5 Batch    4/269 - Train Accuracy:  0.697, Validation Accuracy:  0.717, Loss:  0.364
    Epoch   5 Batch    5/269 - Train Accuracy:  0.712, Validation Accuracy:  0.715, Loss:  0.361
    Epoch   5 Batch    6/269 - Train Accuracy:  0.741, Validation Accuracy:  0.721, Loss:  0.346
    Epoch   5 Batch    7/269 - Train Accuracy:  0.738, Validation Accuracy:  0.717, Loss:  0.333
    Epoch   5 Batch    8/269 - Train Accuracy:  0.723, Validation Accuracy:  0.715, Loss:  0.363
    Epoch   5 Batch    9/269 - Train Accuracy:  0.715, Validation Accuracy:  0.722, Loss:  0.367
    Epoch   5 Batch   10/269 - Train Accuracy:  0.722, Validation Accuracy:  0.723, Loss:  0.347
    Epoch   5 Batch   11/269 - Train Accuracy:  0.712, Validation Accuracy:  0.719, Loss:  0.357
    Epoch   5 Batch   12/269 - Train Accuracy:  0.719, Validation Accuracy:  0.718, Loss:  0.356
    Epoch   5 Batch   13/269 - Train Accuracy:  0.743, Validation Accuracy:  0.720, Loss:  0.312
    Epoch   5 Batch   14/269 - Train Accuracy:  0.716, Validation Accuracy:  0.710, Loss:  0.352
    Epoch   5 Batch   15/269 - Train Accuracy:  0.732, Validation Accuracy:  0.717, Loss:  0.347
    Epoch   5 Batch   16/269 - Train Accuracy:  0.738, Validation Accuracy:  0.717, Loss:  0.340
    Epoch   5 Batch   17/269 - Train Accuracy:  0.731, Validation Accuracy:  0.720, Loss:  0.335
    Epoch   5 Batch   18/269 - Train Accuracy:  0.724, Validation Accuracy:  0.710, Loss:  0.361
    Epoch   5 Batch   19/269 - Train Accuracy:  0.765, Validation Accuracy:  0.713, Loss:  0.305
    Epoch   5 Batch   20/269 - Train Accuracy:  0.711, Validation Accuracy:  0.719, Loss:  0.355
    Epoch   5 Batch   21/269 - Train Accuracy:  0.721, Validation Accuracy:  0.726, Loss:  0.366
    Epoch   5 Batch   22/269 - Train Accuracy:  0.754, Validation Accuracy:  0.722, Loss:  0.325
    Epoch   5 Batch   23/269 - Train Accuracy:  0.741, Validation Accuracy:  0.727, Loss:  0.331
    Epoch   5 Batch   24/269 - Train Accuracy:  0.736, Validation Accuracy:  0.734, Loss:  0.342
    Epoch   5 Batch   25/269 - Train Accuracy:  0.723, Validation Accuracy:  0.724, Loss:  0.350
    Epoch   5 Batch   26/269 - Train Accuracy:  0.748, Validation Accuracy:  0.726, Loss:  0.309
    Epoch   5 Batch   27/269 - Train Accuracy:  0.728, Validation Accuracy:  0.727, Loss:  0.330
    Epoch   5 Batch   28/269 - Train Accuracy:  0.706, Validation Accuracy:  0.733, Loss:  0.357
    Epoch   5 Batch   29/269 - Train Accuracy:  0.740, Validation Accuracy:  0.727, Loss:  0.341
    Epoch   5 Batch   30/269 - Train Accuracy:  0.738, Validation Accuracy:  0.731, Loss:  0.325
    Epoch   5 Batch   31/269 - Train Accuracy:  0.736, Validation Accuracy:  0.728, Loss:  0.317
    Epoch   5 Batch   32/269 - Train Accuracy:  0.732, Validation Accuracy:  0.727, Loss:  0.323
    Epoch   5 Batch   33/269 - Train Accuracy:  0.742, Validation Accuracy:  0.731, Loss:  0.313
    Epoch   5 Batch   34/269 - Train Accuracy:  0.744, Validation Accuracy:  0.739, Loss:  0.324
    Epoch   5 Batch   35/269 - Train Accuracy:  0.723, Validation Accuracy:  0.729, Loss:  0.347
    Epoch   5 Batch   36/269 - Train Accuracy:  0.734, Validation Accuracy:  0.729, Loss:  0.328
    Epoch   5 Batch   37/269 - Train Accuracy:  0.748, Validation Accuracy:  0.729, Loss:  0.324
    Epoch   5 Batch   38/269 - Train Accuracy:  0.735, Validation Accuracy:  0.721, Loss:  0.318
    Epoch   5 Batch   39/269 - Train Accuracy:  0.755, Validation Accuracy:  0.723, Loss:  0.322
    Epoch   5 Batch   40/269 - Train Accuracy:  0.747, Validation Accuracy:  0.728, Loss:  0.338
    Epoch   5 Batch   41/269 - Train Accuracy:  0.736, Validation Accuracy:  0.731, Loss:  0.335
    Epoch   5 Batch   42/269 - Train Accuracy:  0.763, Validation Accuracy:  0.735, Loss:  0.298
    Epoch   5 Batch   43/269 - Train Accuracy:  0.740, Validation Accuracy:  0.728, Loss:  0.334
    Epoch   5 Batch   44/269 - Train Accuracy:  0.744, Validation Accuracy:  0.730, Loss:  0.325
    Epoch   5 Batch   45/269 - Train Accuracy:  0.724, Validation Accuracy:  0.726, Loss:  0.336
    Epoch   5 Batch   46/269 - Train Accuracy:  0.754, Validation Accuracy:  0.729, Loss:  0.329
    Epoch   5 Batch   47/269 - Train Accuracy:  0.771, Validation Accuracy:  0.730, Loss:  0.301
    Epoch   5 Batch   48/269 - Train Accuracy:  0.745, Validation Accuracy:  0.727, Loss:  0.309
    Epoch   5 Batch   49/269 - Train Accuracy:  0.723, Validation Accuracy:  0.727, Loss:  0.323
    Epoch   5 Batch   50/269 - Train Accuracy:  0.715, Validation Accuracy:  0.735, Loss:  0.341
    Epoch   5 Batch   51/269 - Train Accuracy:  0.736, Validation Accuracy:  0.738, Loss:  0.314
    Epoch   5 Batch   52/269 - Train Accuracy:  0.750, Validation Accuracy:  0.738, Loss:  0.305
    Epoch   5 Batch   53/269 - Train Accuracy:  0.734, Validation Accuracy:  0.736, Loss:  0.342
    Epoch   5 Batch   54/269 - Train Accuracy:  0.754, Validation Accuracy:  0.728, Loss:  0.326
    Epoch   5 Batch   55/269 - Train Accuracy:  0.755, Validation Accuracy:  0.734, Loss:  0.312
    Epoch   5 Batch   56/269 - Train Accuracy:  0.755, Validation Accuracy:  0.741, Loss:  0.315
    Epoch   5 Batch   57/269 - Train Accuracy:  0.759, Validation Accuracy:  0.740, Loss:  0.327
    Epoch   5 Batch   58/269 - Train Accuracy:  0.750, Validation Accuracy:  0.743, Loss:  0.308
    Epoch   5 Batch   59/269 - Train Accuracy:  0.760, Validation Accuracy:  0.739, Loss:  0.298
    Epoch   5 Batch   60/269 - Train Accuracy:  0.751, Validation Accuracy:  0.742, Loss:  0.301
    Epoch   5 Batch   61/269 - Train Accuracy:  0.768, Validation Accuracy:  0.742, Loss:  0.289
    Epoch   5 Batch   62/269 - Train Accuracy:  0.756, Validation Accuracy:  0.743, Loss:  0.310
    Epoch   5 Batch   63/269 - Train Accuracy:  0.753, Validation Accuracy:  0.739, Loss:  0.315
    Epoch   5 Batch   64/269 - Train Accuracy:  0.760, Validation Accuracy:  0.745, Loss:  0.301
    Epoch   5 Batch   65/269 - Train Accuracy:  0.739, Validation Accuracy:  0.749, Loss:  0.311
    Epoch   5 Batch   66/269 - Train Accuracy:  0.743, Validation Accuracy:  0.741, Loss:  0.305
    Epoch   5 Batch   67/269 - Train Accuracy:  0.754, Validation Accuracy:  0.737, Loss:  0.320
    Epoch   5 Batch   68/269 - Train Accuracy:  0.740, Validation Accuracy:  0.739, Loss:  0.312
    Epoch   5 Batch   69/269 - Train Accuracy:  0.717, Validation Accuracy:  0.736, Loss:  0.337
    Epoch   5 Batch   70/269 - Train Accuracy:  0.769, Validation Accuracy:  0.743, Loss:  0.307
    Epoch   5 Batch   71/269 - Train Accuracy:  0.756, Validation Accuracy:  0.750, Loss:  0.319
    Epoch   5 Batch   72/269 - Train Accuracy:  0.757, Validation Accuracy:  0.747, Loss:  0.308
    Epoch   5 Batch   73/269 - Train Accuracy:  0.757, Validation Accuracy:  0.737, Loss:  0.312
    Epoch   5 Batch   74/269 - Train Accuracy:  0.760, Validation Accuracy:  0.743, Loss:  0.310
    Epoch   5 Batch   75/269 - Train Accuracy:  0.749, Validation Accuracy:  0.743, Loss:  0.308
    Epoch   5 Batch   76/269 - Train Accuracy:  0.751, Validation Accuracy:  0.733, Loss:  0.302
    Epoch   5 Batch   77/269 - Train Accuracy:  0.767, Validation Accuracy:  0.735, Loss:  0.304
    Epoch   5 Batch   78/269 - Train Accuracy:  0.759, Validation Accuracy:  0.740, Loss:  0.301
    Epoch   5 Batch   79/269 - Train Accuracy:  0.752, Validation Accuracy:  0.743, Loss:  0.306
    Epoch   5 Batch   80/269 - Train Accuracy:  0.759, Validation Accuracy:  0.742, Loss:  0.308
    Epoch   5 Batch   81/269 - Train Accuracy:  0.756, Validation Accuracy:  0.746, Loss:  0.314
    Epoch   5 Batch   82/269 - Train Accuracy:  0.785, Validation Accuracy:  0.750, Loss:  0.286
    Epoch   5 Batch   83/269 - Train Accuracy:  0.752, Validation Accuracy:  0.750, Loss:  0.328
    Epoch   5 Batch   84/269 - Train Accuracy:  0.762, Validation Accuracy:  0.739, Loss:  0.295
    Epoch   5 Batch   85/269 - Train Accuracy:  0.759, Validation Accuracy:  0.741, Loss:  0.298
    Epoch   5 Batch   86/269 - Train Accuracy:  0.758, Validation Accuracy:  0.746, Loss:  0.287
    Epoch   5 Batch   87/269 - Train Accuracy:  0.744, Validation Accuracy:  0.748, Loss:  0.316
    Epoch   5 Batch   88/269 - Train Accuracy:  0.750, Validation Accuracy:  0.736, Loss:  0.298
    Epoch   5 Batch   89/269 - Train Accuracy:  0.773, Validation Accuracy:  0.739, Loss:  0.296
    Epoch   5 Batch   90/269 - Train Accuracy:  0.723, Validation Accuracy:  0.742, Loss:  0.315
    Epoch   5 Batch   91/269 - Train Accuracy:  0.752, Validation Accuracy:  0.741, Loss:  0.290
    Epoch   5 Batch   92/269 - Train Accuracy:  0.757, Validation Accuracy:  0.741, Loss:  0.283
    Epoch   5 Batch   93/269 - Train Accuracy:  0.757, Validation Accuracy:  0.750, Loss:  0.288
    Epoch   5 Batch   94/269 - Train Accuracy:  0.746, Validation Accuracy:  0.752, Loss:  0.313
    Epoch   5 Batch   95/269 - Train Accuracy:  0.755, Validation Accuracy:  0.744, Loss:  0.296
    Epoch   5 Batch   96/269 - Train Accuracy:  0.753, Validation Accuracy:  0.746, Loss:  0.300
    Epoch   5 Batch   97/269 - Train Accuracy:  0.754, Validation Accuracy:  0.744, Loss:  0.289
    Epoch   5 Batch   98/269 - Train Accuracy:  0.764, Validation Accuracy:  0.748, Loss:  0.297
    Epoch   5 Batch   99/269 - Train Accuracy:  0.750, Validation Accuracy:  0.746, Loss:  0.301
    Epoch   5 Batch  100/269 - Train Accuracy:  0.777, Validation Accuracy:  0.750, Loss:  0.289
    Epoch   5 Batch  101/269 - Train Accuracy:  0.737, Validation Accuracy:  0.753, Loss:  0.317
    Epoch   5 Batch  102/269 - Train Accuracy:  0.747, Validation Accuracy:  0.758, Loss:  0.291
    Epoch   5 Batch  103/269 - Train Accuracy:  0.756, Validation Accuracy:  0.754, Loss:  0.294
    Epoch   5 Batch  104/269 - Train Accuracy:  0.759, Validation Accuracy:  0.748, Loss:  0.297
    Epoch   5 Batch  105/269 - Train Accuracy:  0.750, Validation Accuracy:  0.759, Loss:  0.296
    Epoch   5 Batch  106/269 - Train Accuracy:  0.755, Validation Accuracy:  0.761, Loss:  0.286
    Epoch   5 Batch  107/269 - Train Accuracy:  0.734, Validation Accuracy:  0.755, Loss:  0.301
    Epoch   5 Batch  108/269 - Train Accuracy:  0.749, Validation Accuracy:  0.756, Loss:  0.292
    Epoch   5 Batch  109/269 - Train Accuracy:  0.734, Validation Accuracy:  0.753, Loss:  0.302
    Epoch   5 Batch  110/269 - Train Accuracy:  0.751, Validation Accuracy:  0.750, Loss:  0.275
    Epoch   5 Batch  111/269 - Train Accuracy:  0.753, Validation Accuracy:  0.753, Loss:  0.311
    Epoch   5 Batch  112/269 - Train Accuracy:  0.761, Validation Accuracy:  0.751, Loss:  0.292
    Epoch   5 Batch  113/269 - Train Accuracy:  0.749, Validation Accuracy:  0.752, Loss:  0.280
    Epoch   5 Batch  114/269 - Train Accuracy:  0.765, Validation Accuracy:  0.753, Loss:  0.288
    Epoch   5 Batch  115/269 - Train Accuracy:  0.742, Validation Accuracy:  0.757, Loss:  0.303
    Epoch   5 Batch  116/269 - Train Accuracy:  0.772, Validation Accuracy:  0.756, Loss:  0.289
    Epoch   5 Batch  117/269 - Train Accuracy:  0.754, Validation Accuracy:  0.754, Loss:  0.284
    Epoch   5 Batch  118/269 - Train Accuracy:  0.786, Validation Accuracy:  0.757, Loss:  0.281
    Epoch   5 Batch  119/269 - Train Accuracy:  0.755, Validation Accuracy:  0.759, Loss:  0.295
    Epoch   5 Batch  120/269 - Train Accuracy:  0.751, Validation Accuracy:  0.746, Loss:  0.296
    Epoch   5 Batch  121/269 - Train Accuracy:  0.765, Validation Accuracy:  0.751, Loss:  0.277
    Epoch   5 Batch  122/269 - Train Accuracy:  0.769, Validation Accuracy:  0.752, Loss:  0.282
    Epoch   5 Batch  123/269 - Train Accuracy:  0.768, Validation Accuracy:  0.757, Loss:  0.293
    Epoch   5 Batch  124/269 - Train Accuracy:  0.755, Validation Accuracy:  0.756, Loss:  0.271
    Epoch   5 Batch  125/269 - Train Accuracy:  0.774, Validation Accuracy:  0.755, Loss:  0.279
    Epoch   5 Batch  126/269 - Train Accuracy:  0.767, Validation Accuracy:  0.752, Loss:  0.283
    Epoch   5 Batch  127/269 - Train Accuracy:  0.770, Validation Accuracy:  0.748, Loss:  0.286
    Epoch   5 Batch  128/269 - Train Accuracy:  0.780, Validation Accuracy:  0.755, Loss:  0.287
    Epoch   5 Batch  129/269 - Train Accuracy:  0.758, Validation Accuracy:  0.756, Loss:  0.286
    Epoch   5 Batch  130/269 - Train Accuracy:  0.745, Validation Accuracy:  0.749, Loss:  0.286
    Epoch   5 Batch  131/269 - Train Accuracy:  0.753, Validation Accuracy:  0.762, Loss:  0.296
    Epoch   5 Batch  132/269 - Train Accuracy:  0.750, Validation Accuracy:  0.761, Loss:  0.290
    Epoch   5 Batch  133/269 - Train Accuracy:  0.772, Validation Accuracy:  0.744, Loss:  0.273
    Epoch   5 Batch  134/269 - Train Accuracy:  0.752, Validation Accuracy:  0.750, Loss:  0.300
    Epoch   5 Batch  135/269 - Train Accuracy:  0.757, Validation Accuracy:  0.752, Loss:  0.305
    Epoch   5 Batch  136/269 - Train Accuracy:  0.736, Validation Accuracy:  0.742, Loss:  0.303
    Epoch   5 Batch  137/269 - Train Accuracy:  0.772, Validation Accuracy:  0.763, Loss:  0.298
    Epoch   5 Batch  138/269 - Train Accuracy:  0.771, Validation Accuracy:  0.764, Loss:  0.287
    Epoch   5 Batch  139/269 - Train Accuracy:  0.757, Validation Accuracy:  0.752, Loss:  0.280
    Epoch   5 Batch  140/269 - Train Accuracy:  0.789, Validation Accuracy:  0.758, Loss:  0.298
    Epoch   5 Batch  141/269 - Train Accuracy:  0.768, Validation Accuracy:  0.762, Loss:  0.289
    Epoch   5 Batch  142/269 - Train Accuracy:  0.773, Validation Accuracy:  0.758, Loss:  0.278
    Epoch   5 Batch  143/269 - Train Accuracy:  0.785, Validation Accuracy:  0.754, Loss:  0.274
    Epoch   5 Batch  144/269 - Train Accuracy:  0.784, Validation Accuracy:  0.761, Loss:  0.264
    Epoch   5 Batch  145/269 - Train Accuracy:  0.767, Validation Accuracy:  0.755, Loss:  0.272
    Epoch   5 Batch  146/269 - Train Accuracy:  0.768, Validation Accuracy:  0.749, Loss:  0.276
    Epoch   5 Batch  147/269 - Train Accuracy:  0.794, Validation Accuracy:  0.749, Loss:  0.268
    Epoch   5 Batch  148/269 - Train Accuracy:  0.763, Validation Accuracy:  0.752, Loss:  0.280
    Epoch   5 Batch  149/269 - Train Accuracy:  0.759, Validation Accuracy:  0.755, Loss:  0.283
    Epoch   5 Batch  150/269 - Train Accuracy:  0.762, Validation Accuracy:  0.754, Loss:  0.273
    Epoch   5 Batch  151/269 - Train Accuracy:  0.773, Validation Accuracy:  0.751, Loss:  0.266
    Epoch   5 Batch  152/269 - Train Accuracy:  0.765, Validation Accuracy:  0.759, Loss:  0.278
    Epoch   5 Batch  153/269 - Train Accuracy:  0.771, Validation Accuracy:  0.755, Loss:  0.270
    Epoch   5 Batch  154/269 - Train Accuracy:  0.775, Validation Accuracy:  0.766, Loss:  0.281
    Epoch   5 Batch  155/269 - Train Accuracy:  0.781, Validation Accuracy:  0.766, Loss:  0.261
    Epoch   5 Batch  156/269 - Train Accuracy:  0.766, Validation Accuracy:  0.762, Loss:  0.290
    Epoch   5 Batch  157/269 - Train Accuracy:  0.785, Validation Accuracy:  0.768, Loss:  0.271
    Epoch   5 Batch  158/269 - Train Accuracy:  0.772, Validation Accuracy:  0.764, Loss:  0.276
    Epoch   5 Batch  159/269 - Train Accuracy:  0.760, Validation Accuracy:  0.771, Loss:  0.279
    Epoch   5 Batch  160/269 - Train Accuracy:  0.785, Validation Accuracy:  0.772, Loss:  0.265
    Epoch   5 Batch  161/269 - Train Accuracy:  0.769, Validation Accuracy:  0.760, Loss:  0.271
    Epoch   5 Batch  162/269 - Train Accuracy:  0.782, Validation Accuracy:  0.754, Loss:  0.272
    Epoch   5 Batch  163/269 - Train Accuracy:  0.779, Validation Accuracy:  0.758, Loss:  0.277
    Epoch   5 Batch  164/269 - Train Accuracy:  0.784, Validation Accuracy:  0.761, Loss:  0.263
    Epoch   5 Batch  165/269 - Train Accuracy:  0.768, Validation Accuracy:  0.761, Loss:  0.281
    Epoch   5 Batch  166/269 - Train Accuracy:  0.771, Validation Accuracy:  0.760, Loss:  0.264
    Epoch   5 Batch  167/269 - Train Accuracy:  0.781, Validation Accuracy:  0.752, Loss:  0.269
    Epoch   5 Batch  168/269 - Train Accuracy:  0.770, Validation Accuracy:  0.754, Loss:  0.273
    Epoch   5 Batch  169/269 - Train Accuracy:  0.770, Validation Accuracy:  0.765, Loss:  0.272
    Epoch   5 Batch  170/269 - Train Accuracy:  0.766, Validation Accuracy:  0.762, Loss:  0.261
    Epoch   5 Batch  171/269 - Train Accuracy:  0.778, Validation Accuracy:  0.766, Loss:  0.281
    Epoch   5 Batch  172/269 - Train Accuracy:  0.768, Validation Accuracy:  0.763, Loss:  0.279
    Epoch   5 Batch  173/269 - Train Accuracy:  0.777, Validation Accuracy:  0.771, Loss:  0.263
    Epoch   5 Batch  174/269 - Train Accuracy:  0.762, Validation Accuracy:  0.774, Loss:  0.266
    Epoch   5 Batch  175/269 - Train Accuracy:  0.780, Validation Accuracy:  0.769, Loss:  0.294
    Epoch   5 Batch  176/269 - Train Accuracy:  0.763, Validation Accuracy:  0.766, Loss:  0.281
    Epoch   5 Batch  177/269 - Train Accuracy:  0.782, Validation Accuracy:  0.767, Loss:  0.252
    Epoch   5 Batch  178/269 - Train Accuracy:  0.775, Validation Accuracy:  0.767, Loss:  0.269
    Epoch   5 Batch  179/269 - Train Accuracy:  0.760, Validation Accuracy:  0.767, Loss:  0.264
    Epoch   5 Batch  180/269 - Train Accuracy:  0.782, Validation Accuracy:  0.773, Loss:  0.255
    Epoch   5 Batch  181/269 - Train Accuracy:  0.761, Validation Accuracy:  0.767, Loss:  0.263
    Epoch   5 Batch  182/269 - Train Accuracy:  0.783, Validation Accuracy:  0.767, Loss:  0.260
    Epoch   5 Batch  183/269 - Train Accuracy:  0.817, Validation Accuracy:  0.769, Loss:  0.231
    Epoch   5 Batch  184/269 - Train Accuracy:  0.792, Validation Accuracy:  0.777, Loss:  0.266
    Epoch   5 Batch  185/269 - Train Accuracy:  0.794, Validation Accuracy:  0.780, Loss:  0.258
    Epoch   5 Batch  186/269 - Train Accuracy:  0.769, Validation Accuracy:  0.770, Loss:  0.259
    Epoch   5 Batch  187/269 - Train Accuracy:  0.774, Validation Accuracy:  0.772, Loss:  0.258
    Epoch   5 Batch  188/269 - Train Accuracy:  0.796, Validation Accuracy:  0.779, Loss:  0.256
    Epoch   5 Batch  189/269 - Train Accuracy:  0.774, Validation Accuracy:  0.781, Loss:  0.247
    Epoch   5 Batch  190/269 - Train Accuracy:  0.794, Validation Accuracy:  0.772, Loss:  0.246
    Epoch   5 Batch  191/269 - Train Accuracy:  0.783, Validation Accuracy:  0.773, Loss:  0.253
    Epoch   5 Batch  192/269 - Train Accuracy:  0.796, Validation Accuracy:  0.775, Loss:  0.259
    Epoch   5 Batch  193/269 - Train Accuracy:  0.786, Validation Accuracy:  0.769, Loss:  0.250
    Epoch   5 Batch  194/269 - Train Accuracy:  0.785, Validation Accuracy:  0.775, Loss:  0.261
    Epoch   5 Batch  195/269 - Train Accuracy:  0.778, Validation Accuracy:  0.776, Loss:  0.252
    Epoch   5 Batch  196/269 - Train Accuracy:  0.770, Validation Accuracy:  0.776, Loss:  0.248
    Epoch   5 Batch  197/269 - Train Accuracy:  0.783, Validation Accuracy:  0.768, Loss:  0.266
    Epoch   5 Batch  198/269 - Train Accuracy:  0.768, Validation Accuracy:  0.769, Loss:  0.258
    Epoch   5 Batch  199/269 - Train Accuracy:  0.777, Validation Accuracy:  0.776, Loss:  0.264
    Epoch   5 Batch  200/269 - Train Accuracy:  0.785, Validation Accuracy:  0.781, Loss:  0.257
    Epoch   5 Batch  201/269 - Train Accuracy:  0.782, Validation Accuracy:  0.778, Loss:  0.250
    Epoch   5 Batch  202/269 - Train Accuracy:  0.773, Validation Accuracy:  0.775, Loss:  0.255
    Epoch   5 Batch  203/269 - Train Accuracy:  0.773, Validation Accuracy:  0.776, Loss:  0.273
    Epoch   5 Batch  204/269 - Train Accuracy:  0.773, Validation Accuracy:  0.775, Loss:  0.263
    Epoch   5 Batch  205/269 - Train Accuracy:  0.786, Validation Accuracy:  0.775, Loss:  0.251
    Epoch   5 Batch  206/269 - Train Accuracy:  0.795, Validation Accuracy:  0.773, Loss:  0.259
    Epoch   5 Batch  207/269 - Train Accuracy:  0.779, Validation Accuracy:  0.772, Loss:  0.247
    Epoch   5 Batch  208/269 - Train Accuracy:  0.771, Validation Accuracy:  0.774, Loss:  0.266
    Epoch   5 Batch  209/269 - Train Accuracy:  0.798, Validation Accuracy:  0.778, Loss:  0.257
    Epoch   5 Batch  210/269 - Train Accuracy:  0.795, Validation Accuracy:  0.782, Loss:  0.246
    Epoch   5 Batch  211/269 - Train Accuracy:  0.777, Validation Accuracy:  0.772, Loss:  0.266
    Epoch   5 Batch  212/269 - Train Accuracy:  0.781, Validation Accuracy:  0.774, Loss:  0.265
    Epoch   5 Batch  213/269 - Train Accuracy:  0.791, Validation Accuracy:  0.781, Loss:  0.259
    Epoch   5 Batch  214/269 - Train Accuracy:  0.776, Validation Accuracy:  0.780, Loss:  0.253
    Epoch   5 Batch  215/269 - Train Accuracy:  0.790, Validation Accuracy:  0.774, Loss:  0.248
    Epoch   5 Batch  216/269 - Train Accuracy:  0.760, Validation Accuracy:  0.781, Loss:  0.291
    Epoch   5 Batch  217/269 - Train Accuracy:  0.767, Validation Accuracy:  0.777, Loss:  0.264
    Epoch   5 Batch  218/269 - Train Accuracy:  0.792, Validation Accuracy:  0.769, Loss:  0.254
    Epoch   5 Batch  219/269 - Train Accuracy:  0.781, Validation Accuracy:  0.772, Loss:  0.265
    Epoch   5 Batch  220/269 - Train Accuracy:  0.791, Validation Accuracy:  0.772, Loss:  0.235
    Epoch   5 Batch  221/269 - Train Accuracy:  0.798, Validation Accuracy:  0.771, Loss:  0.256
    Epoch   5 Batch  222/269 - Train Accuracy:  0.785, Validation Accuracy:  0.768, Loss:  0.246
    Epoch   5 Batch  223/269 - Train Accuracy:  0.786, Validation Accuracy:  0.773, Loss:  0.247
    Epoch   5 Batch  224/269 - Train Accuracy:  0.784, Validation Accuracy:  0.773, Loss:  0.260
    Epoch   5 Batch  225/269 - Train Accuracy:  0.786, Validation Accuracy:  0.766, Loss:  0.250
    Epoch   5 Batch  226/269 - Train Accuracy:  0.785, Validation Accuracy:  0.780, Loss:  0.257
    Epoch   5 Batch  227/269 - Train Accuracy:  0.801, Validation Accuracy:  0.774, Loss:  0.229
    Epoch   5 Batch  228/269 - Train Accuracy:  0.793, Validation Accuracy:  0.773, Loss:  0.244
    Epoch   5 Batch  229/269 - Train Accuracy:  0.775, Validation Accuracy:  0.767, Loss:  0.243
    Epoch   5 Batch  230/269 - Train Accuracy:  0.781, Validation Accuracy:  0.774, Loss:  0.244
    Epoch   5 Batch  231/269 - Train Accuracy:  0.758, Validation Accuracy:  0.769, Loss:  0.259
    Epoch   5 Batch  232/269 - Train Accuracy:  0.772, Validation Accuracy:  0.771, Loss:  0.254
    Epoch   5 Batch  233/269 - Train Accuracy:  0.803, Validation Accuracy:  0.774, Loss:  0.237
    Epoch   5 Batch  234/269 - Train Accuracy:  0.795, Validation Accuracy:  0.783, Loss:  0.242
    Epoch   5 Batch  235/269 - Train Accuracy:  0.795, Validation Accuracy:  0.782, Loss:  0.228
    Epoch   5 Batch  236/269 - Train Accuracy:  0.777, Validation Accuracy:  0.776, Loss:  0.232
    Epoch   5 Batch  237/269 - Train Accuracy:  0.782, Validation Accuracy:  0.786, Loss:  0.244
    Epoch   5 Batch  238/269 - Train Accuracy:  0.805, Validation Accuracy:  0.789, Loss:  0.235
    Epoch   5 Batch  239/269 - Train Accuracy:  0.794, Validation Accuracy:  0.779, Loss:  0.238
    Epoch   5 Batch  240/269 - Train Accuracy:  0.811, Validation Accuracy:  0.782, Loss:  0.215
    Epoch   5 Batch  241/269 - Train Accuracy:  0.780, Validation Accuracy:  0.786, Loss:  0.256
    Epoch   5 Batch  242/269 - Train Accuracy:  0.783, Validation Accuracy:  0.785, Loss:  0.231
    Epoch   5 Batch  243/269 - Train Accuracy:  0.812, Validation Accuracy:  0.782, Loss:  0.228
    Epoch   5 Batch  244/269 - Train Accuracy:  0.805, Validation Accuracy:  0.789, Loss:  0.239
    Epoch   5 Batch  245/269 - Train Accuracy:  0.793, Validation Accuracy:  0.778, Loss:  0.242
    Epoch   5 Batch  246/269 - Train Accuracy:  0.779, Validation Accuracy:  0.781, Loss:  0.235
    Epoch   5 Batch  247/269 - Train Accuracy:  0.791, Validation Accuracy:  0.785, Loss:  0.238
    Epoch   5 Batch  248/269 - Train Accuracy:  0.790, Validation Accuracy:  0.782, Loss:  0.232
    Epoch   5 Batch  249/269 - Train Accuracy:  0.822, Validation Accuracy:  0.779, Loss:  0.213
    Epoch   5 Batch  250/269 - Train Accuracy:  0.791, Validation Accuracy:  0.782, Loss:  0.236
    Epoch   5 Batch  251/269 - Train Accuracy:  0.838, Validation Accuracy:  0.784, Loss:  0.223
    Epoch   5 Batch  252/269 - Train Accuracy:  0.803, Validation Accuracy:  0.785, Loss:  0.219
    Epoch   5 Batch  253/269 - Train Accuracy:  0.789, Validation Accuracy:  0.785, Loss:  0.233
    Epoch   5 Batch  254/269 - Train Accuracy:  0.814, Validation Accuracy:  0.790, Loss:  0.232
    Epoch   5 Batch  255/269 - Train Accuracy:  0.810, Validation Accuracy:  0.795, Loss:  0.219
    Epoch   5 Batch  256/269 - Train Accuracy:  0.781, Validation Accuracy:  0.780, Loss:  0.229
    Epoch   5 Batch  257/269 - Train Accuracy:  0.777, Validation Accuracy:  0.790, Loss:  0.240
    Epoch   5 Batch  258/269 - Train Accuracy:  0.800, Validation Accuracy:  0.783, Loss:  0.224
    Epoch   5 Batch  259/269 - Train Accuracy:  0.799, Validation Accuracy:  0.784, Loss:  0.230
    Epoch   5 Batch  260/269 - Train Accuracy:  0.766, Validation Accuracy:  0.781, Loss:  0.248
    Epoch   5 Batch  261/269 - Train Accuracy:  0.795, Validation Accuracy:  0.796, Loss:  0.230
    Epoch   5 Batch  262/269 - Train Accuracy:  0.803, Validation Accuracy:  0.786, Loss:  0.221
    Epoch   5 Batch  263/269 - Train Accuracy:  0.798, Validation Accuracy:  0.779, Loss:  0.235
    Epoch   5 Batch  264/269 - Train Accuracy:  0.765, Validation Accuracy:  0.788, Loss:  0.248
    Epoch   5 Batch  265/269 - Train Accuracy:  0.789, Validation Accuracy:  0.775, Loss:  0.233
    Epoch   5 Batch  266/269 - Train Accuracy:  0.789, Validation Accuracy:  0.782, Loss:  0.224
    Epoch   5 Batch  267/269 - Train Accuracy:  0.796, Validation Accuracy:  0.789, Loss:  0.239
    Epoch   6 Batch    0/269 - Train Accuracy:  0.788, Validation Accuracy:  0.784, Loss:  0.240
    Epoch   6 Batch    1/269 - Train Accuracy:  0.771, Validation Accuracy:  0.784, Loss:  0.234
    Epoch   6 Batch    2/269 - Train Accuracy:  0.790, Validation Accuracy:  0.783, Loss:  0.241
    Epoch   6 Batch    3/269 - Train Accuracy:  0.789, Validation Accuracy:  0.792, Loss:  0.234
    Epoch   6 Batch    4/269 - Train Accuracy:  0.771, Validation Accuracy:  0.786, Loss:  0.237
    Epoch   6 Batch    5/269 - Train Accuracy:  0.765, Validation Accuracy:  0.789, Loss:  0.234
    Epoch   6 Batch    6/269 - Train Accuracy:  0.791, Validation Accuracy:  0.789, Loss:  0.222
    Epoch   6 Batch    7/269 - Train Accuracy:  0.812, Validation Accuracy:  0.789, Loss:  0.224
    Epoch   6 Batch    8/269 - Train Accuracy:  0.786, Validation Accuracy:  0.780, Loss:  0.235
    Epoch   6 Batch    9/269 - Train Accuracy:  0.786, Validation Accuracy:  0.798, Loss:  0.236
    Epoch   6 Batch   10/269 - Train Accuracy:  0.790, Validation Accuracy:  0.781, Loss:  0.230
    Epoch   6 Batch   11/269 - Train Accuracy:  0.782, Validation Accuracy:  0.789, Loss:  0.242
    Epoch   6 Batch   12/269 - Train Accuracy:  0.787, Validation Accuracy:  0.784, Loss:  0.239
    Epoch   6 Batch   13/269 - Train Accuracy:  0.811, Validation Accuracy:  0.782, Loss:  0.212
    Epoch   6 Batch   14/269 - Train Accuracy:  0.781, Validation Accuracy:  0.788, Loss:  0.233
    Epoch   6 Batch   15/269 - Train Accuracy:  0.803, Validation Accuracy:  0.798, Loss:  0.212
    Epoch   6 Batch   16/269 - Train Accuracy:  0.804, Validation Accuracy:  0.794, Loss:  0.219
    Epoch   6 Batch   17/269 - Train Accuracy:  0.798, Validation Accuracy:  0.790, Loss:  0.217
    Epoch   6 Batch   18/269 - Train Accuracy:  0.803, Validation Accuracy:  0.795, Loss:  0.230
    Epoch   6 Batch   19/269 - Train Accuracy:  0.831, Validation Accuracy:  0.786, Loss:  0.204
    Epoch   6 Batch   20/269 - Train Accuracy:  0.775, Validation Accuracy:  0.792, Loss:  0.231
    Epoch   6 Batch   21/269 - Train Accuracy:  0.795, Validation Accuracy:  0.794, Loss:  0.251
    Epoch   6 Batch   22/269 - Train Accuracy:  0.810, Validation Accuracy:  0.786, Loss:  0.215
    Epoch   6 Batch   23/269 - Train Accuracy:  0.803, Validation Accuracy:  0.785, Loss:  0.223
    Epoch   6 Batch   24/269 - Train Accuracy:  0.800, Validation Accuracy:  0.792, Loss:  0.231
    Epoch   6 Batch   25/269 - Train Accuracy:  0.777, Validation Accuracy:  0.787, Loss:  0.234
    Epoch   6 Batch   26/269 - Train Accuracy:  0.807, Validation Accuracy:  0.789, Loss:  0.202
    Epoch   6 Batch   27/269 - Train Accuracy:  0.784, Validation Accuracy:  0.786, Loss:  0.211
    Epoch   6 Batch   28/269 - Train Accuracy:  0.764, Validation Accuracy:  0.791, Loss:  0.240
    Epoch   6 Batch   29/269 - Train Accuracy:  0.809, Validation Accuracy:  0.794, Loss:  0.233
    Epoch   6 Batch   30/269 - Train Accuracy:  0.795, Validation Accuracy:  0.788, Loss:  0.219
    Epoch   6 Batch   31/269 - Train Accuracy:  0.802, Validation Accuracy:  0.787, Loss:  0.211
    Epoch   6 Batch   32/269 - Train Accuracy:  0.781, Validation Accuracy:  0.784, Loss:  0.218
    Epoch   6 Batch   33/269 - Train Accuracy:  0.807, Validation Accuracy:  0.786, Loss:  0.218
    Epoch   6 Batch   34/269 - Train Accuracy:  0.803, Validation Accuracy:  0.779, Loss:  0.217
    Epoch   6 Batch   35/269 - Train Accuracy:  0.784, Validation Accuracy:  0.783, Loss:  0.249
    Epoch   6 Batch   36/269 - Train Accuracy:  0.778, Validation Accuracy:  0.791, Loss:  0.223
    Epoch   6 Batch   37/269 - Train Accuracy:  0.804, Validation Accuracy:  0.789, Loss:  0.223
    Epoch   6 Batch   38/269 - Train Accuracy:  0.795, Validation Accuracy:  0.782, Loss:  0.220
    Epoch   6 Batch   39/269 - Train Accuracy:  0.807, Validation Accuracy:  0.784, Loss:  0.223
    Epoch   6 Batch   40/269 - Train Accuracy:  0.799, Validation Accuracy:  0.782, Loss:  0.235
    Epoch   6 Batch   41/269 - Train Accuracy:  0.787, Validation Accuracy:  0.787, Loss:  0.227
    Epoch   6 Batch   42/269 - Train Accuracy:  0.817, Validation Accuracy:  0.788, Loss:  0.204
    Epoch   6 Batch   43/269 - Train Accuracy:  0.788, Validation Accuracy:  0.781, Loss:  0.228
    Epoch   6 Batch   44/269 - Train Accuracy:  0.796, Validation Accuracy:  0.778, Loss:  0.226
    Epoch   6 Batch   45/269 - Train Accuracy:  0.766, Validation Accuracy:  0.778, Loss:  0.230
    Epoch   6 Batch   46/269 - Train Accuracy:  0.794, Validation Accuracy:  0.779, Loss:  0.233
    Epoch   6 Batch   47/269 - Train Accuracy:  0.813, Validation Accuracy:  0.781, Loss:  0.209
    Epoch   6 Batch   48/269 - Train Accuracy:  0.796, Validation Accuracy:  0.786, Loss:  0.215
    Epoch   6 Batch   49/269 - Train Accuracy:  0.782, Validation Accuracy:  0.786, Loss:  0.217
    Epoch   6 Batch   50/269 - Train Accuracy:  0.783, Validation Accuracy:  0.784, Loss:  0.236
    Epoch   6 Batch   51/269 - Train Accuracy:  0.798, Validation Accuracy:  0.778, Loss:  0.214
    Epoch   6 Batch   52/269 - Train Accuracy:  0.799, Validation Accuracy:  0.782, Loss:  0.196
    Epoch   6 Batch   53/269 - Train Accuracy:  0.789, Validation Accuracy:  0.787, Loss:  0.232
    Epoch   6 Batch   54/269 - Train Accuracy:  0.820, Validation Accuracy:  0.784, Loss:  0.213
    Epoch   6 Batch   55/269 - Train Accuracy:  0.813, Validation Accuracy:  0.795, Loss:  0.211
    Epoch   6 Batch   56/269 - Train Accuracy:  0.811, Validation Accuracy:  0.793, Loss:  0.215
    Epoch   6 Batch   57/269 - Train Accuracy:  0.799, Validation Accuracy:  0.792, Loss:  0.218
    Epoch   6 Batch   58/269 - Train Accuracy:  0.794, Validation Accuracy:  0.794, Loss:  0.210
    Epoch   6 Batch   59/269 - Train Accuracy:  0.819, Validation Accuracy:  0.792, Loss:  0.198
    Epoch   6 Batch   60/269 - Train Accuracy:  0.793, Validation Accuracy:  0.787, Loss:  0.198
    Epoch   6 Batch   61/269 - Train Accuracy:  0.815, Validation Accuracy:  0.803, Loss:  0.203
    Epoch   6 Batch   62/269 - Train Accuracy:  0.818, Validation Accuracy:  0.806, Loss:  0.210
    Epoch   6 Batch   63/269 - Train Accuracy:  0.797, Validation Accuracy:  0.790, Loss:  0.217
    Epoch   6 Batch   64/269 - Train Accuracy:  0.803, Validation Accuracy:  0.796, Loss:  0.198
    Epoch   6 Batch   65/269 - Train Accuracy:  0.803, Validation Accuracy:  0.802, Loss:  0.203
    Epoch   6 Batch   66/269 - Train Accuracy:  0.796, Validation Accuracy:  0.793, Loss:  0.208
    Epoch   6 Batch   67/269 - Train Accuracy:  0.785, Validation Accuracy:  0.795, Loss:  0.220
    Epoch   6 Batch   68/269 - Train Accuracy:  0.798, Validation Accuracy:  0.797, Loss:  0.218
    Epoch   6 Batch   69/269 - Train Accuracy:  0.772, Validation Accuracy:  0.790, Loss:  0.229
    Epoch   6 Batch   70/269 - Train Accuracy:  0.810, Validation Accuracy:  0.788, Loss:  0.212
    Epoch   6 Batch   71/269 - Train Accuracy:  0.802, Validation Accuracy:  0.798, Loss:  0.217
    Epoch   6 Batch   72/269 - Train Accuracy:  0.797, Validation Accuracy:  0.800, Loss:  0.222
    Epoch   6 Batch   73/269 - Train Accuracy:  0.806, Validation Accuracy:  0.801, Loss:  0.211
    Epoch   6 Batch   74/269 - Train Accuracy:  0.790, Validation Accuracy:  0.791, Loss:  0.214
    Epoch   6 Batch   75/269 - Train Accuracy:  0.806, Validation Accuracy:  0.794, Loss:  0.219
    Epoch   6 Batch   76/269 - Train Accuracy:  0.805, Validation Accuracy:  0.796, Loss:  0.210
    Epoch   6 Batch   77/269 - Train Accuracy:  0.815, Validation Accuracy:  0.790, Loss:  0.211
    Epoch   6 Batch   78/269 - Train Accuracy:  0.806, Validation Accuracy:  0.798, Loss:  0.223
    Epoch   6 Batch   79/269 - Train Accuracy:  0.796, Validation Accuracy:  0.798, Loss:  0.212
    Epoch   6 Batch   80/269 - Train Accuracy:  0.816, Validation Accuracy:  0.788, Loss:  0.207
    Epoch   6 Batch   81/269 - Train Accuracy:  0.799, Validation Accuracy:  0.789, Loss:  0.213
    Epoch   6 Batch   82/269 - Train Accuracy:  0.827, Validation Accuracy:  0.796, Loss:  0.192
    Epoch   6 Batch   83/269 - Train Accuracy:  0.806, Validation Accuracy:  0.791, Loss:  0.220
    Epoch   6 Batch   84/269 - Train Accuracy:  0.807, Validation Accuracy:  0.792, Loss:  0.208
    Epoch   6 Batch   85/269 - Train Accuracy:  0.815, Validation Accuracy:  0.781, Loss:  0.198
    Epoch   6 Batch   86/269 - Train Accuracy:  0.803, Validation Accuracy:  0.786, Loss:  0.200
    Epoch   6 Batch   87/269 - Train Accuracy:  0.786, Validation Accuracy:  0.791, Loss:  0.218
    Epoch   6 Batch   88/269 - Train Accuracy:  0.807, Validation Accuracy:  0.791, Loss:  0.196
    Epoch   6 Batch   89/269 - Train Accuracy:  0.818, Validation Accuracy:  0.789, Loss:  0.203
    Epoch   6 Batch   90/269 - Train Accuracy:  0.778, Validation Accuracy:  0.793, Loss:  0.221
    Epoch   6 Batch   91/269 - Train Accuracy:  0.806, Validation Accuracy:  0.795, Loss:  0.201
    Epoch   6 Batch   92/269 - Train Accuracy:  0.816, Validation Accuracy:  0.801, Loss:  0.189
    Epoch   6 Batch   93/269 - Train Accuracy:  0.807, Validation Accuracy:  0.792, Loss:  0.198
    Epoch   6 Batch   94/269 - Train Accuracy:  0.804, Validation Accuracy:  0.797, Loss:  0.218
    Epoch   6 Batch   95/269 - Train Accuracy:  0.813, Validation Accuracy:  0.795, Loss:  0.195
    Epoch   6 Batch   96/269 - Train Accuracy:  0.800, Validation Accuracy:  0.794, Loss:  0.209
    Epoch   6 Batch   97/269 - Train Accuracy:  0.796, Validation Accuracy:  0.799, Loss:  0.200
    Epoch   6 Batch   98/269 - Train Accuracy:  0.812, Validation Accuracy:  0.802, Loss:  0.200
    Epoch   6 Batch   99/269 - Train Accuracy:  0.803, Validation Accuracy:  0.803, Loss:  0.206
    Epoch   6 Batch  100/269 - Train Accuracy:  0.848, Validation Accuracy:  0.801, Loss:  0.201
    Epoch   6 Batch  101/269 - Train Accuracy:  0.772, Validation Accuracy:  0.792, Loss:  0.211
    Epoch   6 Batch  102/269 - Train Accuracy:  0.803, Validation Accuracy:  0.797, Loss:  0.197
    Epoch   6 Batch  103/269 - Train Accuracy:  0.804, Validation Accuracy:  0.796, Loss:  0.210
    Epoch   6 Batch  104/269 - Train Accuracy:  0.802, Validation Accuracy:  0.794, Loss:  0.199
    Epoch   6 Batch  105/269 - Train Accuracy:  0.812, Validation Accuracy:  0.800, Loss:  0.211
    Epoch   6 Batch  106/269 - Train Accuracy:  0.799, Validation Accuracy:  0.798, Loss:  0.193
    Epoch   6 Batch  107/269 - Train Accuracy:  0.795, Validation Accuracy:  0.786, Loss:  0.204
    Epoch   6 Batch  108/269 - Train Accuracy:  0.807, Validation Accuracy:  0.802, Loss:  0.201
    Epoch   6 Batch  109/269 - Train Accuracy:  0.773, Validation Accuracy:  0.799, Loss:  0.207
    Epoch   6 Batch  110/269 - Train Accuracy:  0.796, Validation Accuracy:  0.786, Loss:  0.201
    Epoch   6 Batch  111/269 - Train Accuracy:  0.790, Validation Accuracy:  0.790, Loss:  0.223
    Epoch   6 Batch  112/269 - Train Accuracy:  0.806, Validation Accuracy:  0.801, Loss:  0.204
    Epoch   6 Batch  113/269 - Train Accuracy:  0.786, Validation Accuracy:  0.787, Loss:  0.196
    Epoch   6 Batch  114/269 - Train Accuracy:  0.808, Validation Accuracy:  0.794, Loss:  0.203
    Epoch   6 Batch  115/269 - Train Accuracy:  0.785, Validation Accuracy:  0.803, Loss:  0.209
    Epoch   6 Batch  116/269 - Train Accuracy:  0.827, Validation Accuracy:  0.805, Loss:  0.199
    Epoch   6 Batch  117/269 - Train Accuracy:  0.796, Validation Accuracy:  0.801, Loss:  0.193
    Epoch   6 Batch  118/269 - Train Accuracy:  0.831, Validation Accuracy:  0.805, Loss:  0.186
    Epoch   6 Batch  119/269 - Train Accuracy:  0.798, Validation Accuracy:  0.798, Loss:  0.209
    Epoch   6 Batch  120/269 - Train Accuracy:  0.804, Validation Accuracy:  0.803, Loss:  0.209
    Epoch   6 Batch  121/269 - Train Accuracy:  0.814, Validation Accuracy:  0.799, Loss:  0.193
    Epoch   6 Batch  122/269 - Train Accuracy:  0.812, Validation Accuracy:  0.803, Loss:  0.192
    Epoch   6 Batch  123/269 - Train Accuracy:  0.814, Validation Accuracy:  0.800, Loss:  0.196
    Epoch   6 Batch  124/269 - Train Accuracy:  0.809, Validation Accuracy:  0.802, Loss:  0.180
    Epoch   6 Batch  125/269 - Train Accuracy:  0.814, Validation Accuracy:  0.804, Loss:  0.183
    Epoch   6 Batch  126/269 - Train Accuracy:  0.803, Validation Accuracy:  0.796, Loss:  0.191
    Epoch   6 Batch  127/269 - Train Accuracy:  0.796, Validation Accuracy:  0.781, Loss:  0.198
    Epoch   6 Batch  128/269 - Train Accuracy:  0.807, Validation Accuracy:  0.801, Loss:  0.198
    Epoch   6 Batch  129/269 - Train Accuracy:  0.806, Validation Accuracy:  0.798, Loss:  0.187
    Epoch   6 Batch  130/269 - Train Accuracy:  0.797, Validation Accuracy:  0.802, Loss:  0.194
    Epoch   6 Batch  131/269 - Train Accuracy:  0.795, Validation Accuracy:  0.806, Loss:  0.202
    Epoch   6 Batch  132/269 - Train Accuracy:  0.798, Validation Accuracy:  0.809, Loss:  0.201
    Epoch   6 Batch  133/269 - Train Accuracy:  0.826, Validation Accuracy:  0.804, Loss:  0.178
    Epoch   6 Batch  134/269 - Train Accuracy:  0.810, Validation Accuracy:  0.810, Loss:  0.198
    Epoch   6 Batch  135/269 - Train Accuracy:  0.816, Validation Accuracy:  0.814, Loss:  0.194
    Epoch   6 Batch  136/269 - Train Accuracy:  0.801, Validation Accuracy:  0.811, Loss:  0.198
    Epoch   6 Batch  137/269 - Train Accuracy:  0.819, Validation Accuracy:  0.809, Loss:  0.195
    Epoch   6 Batch  138/269 - Train Accuracy:  0.824, Validation Accuracy:  0.813, Loss:  0.189
    Epoch   6 Batch  139/269 - Train Accuracy:  0.819, Validation Accuracy:  0.812, Loss:  0.180
    Epoch   6 Batch  140/269 - Train Accuracy:  0.838, Validation Accuracy:  0.812, Loss:  0.189
    Epoch   6 Batch  141/269 - Train Accuracy:  0.818, Validation Accuracy:  0.813, Loss:  0.196
    Epoch   6 Batch  142/269 - Train Accuracy:  0.811, Validation Accuracy:  0.812, Loss:  0.180
    Epoch   6 Batch  143/269 - Train Accuracy:  0.839, Validation Accuracy:  0.807, Loss:  0.176
    Epoch   6 Batch  144/269 - Train Accuracy:  0.833, Validation Accuracy:  0.811, Loss:  0.165
    Epoch   6 Batch  145/269 - Train Accuracy:  0.822, Validation Accuracy:  0.812, Loss:  0.174
    Epoch   6 Batch  146/269 - Train Accuracy:  0.820, Validation Accuracy:  0.805, Loss:  0.187
    Epoch   6 Batch  147/269 - Train Accuracy:  0.828, Validation Accuracy:  0.799, Loss:  0.179
    Epoch   6 Batch  148/269 - Train Accuracy:  0.824, Validation Accuracy:  0.808, Loss:  0.180
    Epoch   6 Batch  149/269 - Train Accuracy:  0.794, Validation Accuracy:  0.802, Loss:  0.198
    Epoch   6 Batch  150/269 - Train Accuracy:  0.818, Validation Accuracy:  0.801, Loss:  0.180
    Epoch   6 Batch  151/269 - Train Accuracy:  0.835, Validation Accuracy:  0.815, Loss:  0.181
    Epoch   6 Batch  152/269 - Train Accuracy:  0.806, Validation Accuracy:  0.817, Loss:  0.184
    Epoch   6 Batch  153/269 - Train Accuracy:  0.836, Validation Accuracy:  0.808, Loss:  0.178
    Epoch   6 Batch  154/269 - Train Accuracy:  0.830, Validation Accuracy:  0.805, Loss:  0.173
    Epoch   6 Batch  155/269 - Train Accuracy:  0.832, Validation Accuracy:  0.812, Loss:  0.179
    Epoch   6 Batch  156/269 - Train Accuracy:  0.819, Validation Accuracy:  0.817, Loss:  0.187
    Epoch   6 Batch  157/269 - Train Accuracy:  0.820, Validation Accuracy:  0.812, Loss:  0.177
    Epoch   6 Batch  158/269 - Train Accuracy:  0.820, Validation Accuracy:  0.814, Loss:  0.186
    Epoch   6 Batch  159/269 - Train Accuracy:  0.805, Validation Accuracy:  0.814, Loss:  0.184
    Epoch   6 Batch  160/269 - Train Accuracy:  0.824, Validation Accuracy:  0.815, Loss:  0.182
    Epoch   6 Batch  161/269 - Train Accuracy:  0.806, Validation Accuracy:  0.813, Loss:  0.177
    Epoch   6 Batch  162/269 - Train Accuracy:  0.831, Validation Accuracy:  0.808, Loss:  0.179
    Epoch   6 Batch  163/269 - Train Accuracy:  0.831, Validation Accuracy:  0.804, Loss:  0.183
    Epoch   6 Batch  164/269 - Train Accuracy:  0.827, Validation Accuracy:  0.804, Loss:  0.180
    Epoch   6 Batch  165/269 - Train Accuracy:  0.806, Validation Accuracy:  0.805, Loss:  0.176
    Epoch   6 Batch  166/269 - Train Accuracy:  0.826, Validation Accuracy:  0.807, Loss:  0.174
    Epoch   6 Batch  167/269 - Train Accuracy:  0.829, Validation Accuracy:  0.801, Loss:  0.172
    Epoch   6 Batch  168/269 - Train Accuracy:  0.825, Validation Accuracy:  0.808, Loss:  0.179
    Epoch   6 Batch  169/269 - Train Accuracy:  0.802, Validation Accuracy:  0.807, Loss:  0.180
    Epoch   6 Batch  170/269 - Train Accuracy:  0.811, Validation Accuracy:  0.808, Loss:  0.176
    Epoch   6 Batch  171/269 - Train Accuracy:  0.828, Validation Accuracy:  0.810, Loss:  0.188
    Epoch   6 Batch  172/269 - Train Accuracy:  0.795, Validation Accuracy:  0.798, Loss:  0.189
    Epoch   6 Batch  173/269 - Train Accuracy:  0.838, Validation Accuracy:  0.812, Loss:  0.177
    Epoch   6 Batch  174/269 - Train Accuracy:  0.819, Validation Accuracy:  0.813, Loss:  0.174
    Epoch   6 Batch  175/269 - Train Accuracy:  0.806, Validation Accuracy:  0.812, Loss:  0.199
    Epoch   6 Batch  176/269 - Train Accuracy:  0.819, Validation Accuracy:  0.809, Loss:  0.194
    Epoch   6 Batch  177/269 - Train Accuracy:  0.827, Validation Accuracy:  0.797, Loss:  0.168
    Epoch   6 Batch  178/269 - Train Accuracy:  0.821, Validation Accuracy:  0.808, Loss:  0.182
    Epoch   6 Batch  179/269 - Train Accuracy:  0.813, Validation Accuracy:  0.810, Loss:  0.183
    Epoch   6 Batch  180/269 - Train Accuracy:  0.824, Validation Accuracy:  0.800, Loss:  0.168
    Epoch   6 Batch  181/269 - Train Accuracy:  0.795, Validation Accuracy:  0.803, Loss:  0.187
    Epoch   6 Batch  182/269 - Train Accuracy:  0.818, Validation Accuracy:  0.807, Loss:  0.178
    Epoch   6 Batch  183/269 - Train Accuracy:  0.854, Validation Accuracy:  0.812, Loss:  0.154
    Epoch   6 Batch  184/269 - Train Accuracy:  0.829, Validation Accuracy:  0.809, Loss:  0.178
    Epoch   6 Batch  185/269 - Train Accuracy:  0.828, Validation Accuracy:  0.809, Loss:  0.174
    Epoch   6 Batch  186/269 - Train Accuracy:  0.809, Validation Accuracy:  0.808, Loss:  0.173
    Epoch   6 Batch  187/269 - Train Accuracy:  0.816, Validation Accuracy:  0.818, Loss:  0.178
    Epoch   6 Batch  188/269 - Train Accuracy:  0.829, Validation Accuracy:  0.815, Loss:  0.166
    Epoch   6 Batch  189/269 - Train Accuracy:  0.816, Validation Accuracy:  0.817, Loss:  0.168
    Epoch   6 Batch  190/269 - Train Accuracy:  0.830, Validation Accuracy:  0.820, Loss:  0.170
    Epoch   6 Batch  191/269 - Train Accuracy:  0.826, Validation Accuracy:  0.818, Loss:  0.162
    Epoch   6 Batch  192/269 - Train Accuracy:  0.831, Validation Accuracy:  0.816, Loss:  0.171
    Epoch   6 Batch  193/269 - Train Accuracy:  0.836, Validation Accuracy:  0.814, Loss:  0.166
    Epoch   6 Batch  194/269 - Train Accuracy:  0.827, Validation Accuracy:  0.812, Loss:  0.168
    Epoch   6 Batch  195/269 - Train Accuracy:  0.818, Validation Accuracy:  0.810, Loss:  0.172
    Epoch   6 Batch  196/269 - Train Accuracy:  0.813, Validation Accuracy:  0.815, Loss:  0.174
    Epoch   6 Batch  197/269 - Train Accuracy:  0.816, Validation Accuracy:  0.816, Loss:  0.175
    Epoch   6 Batch  198/269 - Train Accuracy:  0.804, Validation Accuracy:  0.806, Loss:  0.182
    Epoch   6 Batch  199/269 - Train Accuracy:  0.828, Validation Accuracy:  0.812, Loss:  0.180
    Epoch   6 Batch  200/269 - Train Accuracy:  0.837, Validation Accuracy:  0.824, Loss:  0.171
    Epoch   6 Batch  201/269 - Train Accuracy:  0.825, Validation Accuracy:  0.818, Loss:  0.171
    Epoch   6 Batch  202/269 - Train Accuracy:  0.808, Validation Accuracy:  0.811, Loss:  0.174
    Epoch   6 Batch  203/269 - Train Accuracy:  0.830, Validation Accuracy:  0.806, Loss:  0.187
    Epoch   6 Batch  204/269 - Train Accuracy:  0.813, Validation Accuracy:  0.812, Loss:  0.187
    Epoch   6 Batch  205/269 - Train Accuracy:  0.829, Validation Accuracy:  0.813, Loss:  0.169
    Epoch   6 Batch  206/269 - Train Accuracy:  0.828, Validation Accuracy:  0.805, Loss:  0.177
    Epoch   6 Batch  207/269 - Train Accuracy:  0.826, Validation Accuracy:  0.813, Loss:  0.166
    Epoch   6 Batch  208/269 - Train Accuracy:  0.811, Validation Accuracy:  0.814, Loss:  0.187
    Epoch   6 Batch  209/269 - Train Accuracy:  0.831, Validation Accuracy:  0.809, Loss:  0.165
    Epoch   6 Batch  210/269 - Train Accuracy:  0.834, Validation Accuracy:  0.817, Loss:  0.167
    Epoch   6 Batch  211/269 - Train Accuracy:  0.824, Validation Accuracy:  0.819, Loss:  0.174
    Epoch   6 Batch  212/269 - Train Accuracy:  0.825, Validation Accuracy:  0.810, Loss:  0.175
    Epoch   6 Batch  213/269 - Train Accuracy:  0.825, Validation Accuracy:  0.822, Loss:  0.170
    Epoch   6 Batch  214/269 - Train Accuracy:  0.832, Validation Accuracy:  0.828, Loss:  0.180
    Epoch   6 Batch  215/269 - Train Accuracy:  0.842, Validation Accuracy:  0.817, Loss:  0.160
    Epoch   6 Batch  216/269 - Train Accuracy:  0.821, Validation Accuracy:  0.826, Loss:  0.200
    Epoch   6 Batch  217/269 - Train Accuracy:  0.811, Validation Accuracy:  0.824, Loss:  0.173
    Epoch   6 Batch  218/269 - Train Accuracy:  0.837, Validation Accuracy:  0.809, Loss:  0.172
    Epoch   6 Batch  219/269 - Train Accuracy:  0.817, Validation Accuracy:  0.817, Loss:  0.175
    Epoch   6 Batch  220/269 - Train Accuracy:  0.822, Validation Accuracy:  0.822, Loss:  0.153
    Epoch   6 Batch  221/269 - Train Accuracy:  0.839, Validation Accuracy:  0.811, Loss:  0.168
    Epoch   6 Batch  222/269 - Train Accuracy:  0.832, Validation Accuracy:  0.819, Loss:  0.160
    Epoch   6 Batch  223/269 - Train Accuracy:  0.823, Validation Accuracy:  0.825, Loss:  0.157
    Epoch   6 Batch  224/269 - Train Accuracy:  0.830, Validation Accuracy:  0.816, Loss:  0.179
    Epoch   6 Batch  225/269 - Train Accuracy:  0.816, Validation Accuracy:  0.812, Loss:  0.163
    Epoch   6 Batch  226/269 - Train Accuracy:  0.822, Validation Accuracy:  0.821, Loss:  0.174
    Epoch   6 Batch  227/269 - Train Accuracy:  0.846, Validation Accuracy:  0.820, Loss:  0.168
    Epoch   6 Batch  228/269 - Train Accuracy:  0.823, Validation Accuracy:  0.816, Loss:  0.161
    Epoch   6 Batch  229/269 - Train Accuracy:  0.829, Validation Accuracy:  0.822, Loss:  0.169
    Epoch   6 Batch  230/269 - Train Accuracy:  0.822, Validation Accuracy:  0.816, Loss:  0.160
    Epoch   6 Batch  231/269 - Train Accuracy:  0.799, Validation Accuracy:  0.812, Loss:  0.178
    Epoch   6 Batch  232/269 - Train Accuracy:  0.816, Validation Accuracy:  0.815, Loss:  0.165
    Epoch   6 Batch  233/269 - Train Accuracy:  0.837, Validation Accuracy:  0.820, Loss:  0.171
    Epoch   6 Batch  234/269 - Train Accuracy:  0.831, Validation Accuracy:  0.817, Loss:  0.163
    Epoch   6 Batch  235/269 - Train Accuracy:  0.841, Validation Accuracy:  0.813, Loss:  0.151
    Epoch   6 Batch  236/269 - Train Accuracy:  0.811, Validation Accuracy:  0.820, Loss:  0.164
    Epoch   6 Batch  237/269 - Train Accuracy:  0.823, Validation Accuracy:  0.823, Loss:  0.165
    Epoch   6 Batch  238/269 - Train Accuracy:  0.829, Validation Accuracy:  0.821, Loss:  0.159
    Epoch   6 Batch  239/269 - Train Accuracy:  0.832, Validation Accuracy:  0.809, Loss:  0.163
    Epoch   6 Batch  240/269 - Train Accuracy:  0.840, Validation Accuracy:  0.813, Loss:  0.149
    Epoch   6 Batch  241/269 - Train Accuracy:  0.817, Validation Accuracy:  0.807, Loss:  0.178
    Epoch   6 Batch  242/269 - Train Accuracy:  0.813, Validation Accuracy:  0.811, Loss:  0.153
    Epoch   6 Batch  243/269 - Train Accuracy:  0.851, Validation Accuracy:  0.810, Loss:  0.155
    Epoch   6 Batch  244/269 - Train Accuracy:  0.851, Validation Accuracy:  0.815, Loss:  0.169
    Epoch   6 Batch  245/269 - Train Accuracy:  0.824, Validation Accuracy:  0.823, Loss:  0.169
    Epoch   6 Batch  246/269 - Train Accuracy:  0.807, Validation Accuracy:  0.822, Loss:  0.158
    Epoch   6 Batch  247/269 - Train Accuracy:  0.825, Validation Accuracy:  0.820, Loss:  0.163
    Epoch   6 Batch  248/269 - Train Accuracy:  0.821, Validation Accuracy:  0.814, Loss:  0.153
    Epoch   6 Batch  249/269 - Train Accuracy:  0.854, Validation Accuracy:  0.809, Loss:  0.143
    Epoch   6 Batch  250/269 - Train Accuracy:  0.836, Validation Accuracy:  0.805, Loss:  0.151
    Epoch   6 Batch  251/269 - Train Accuracy:  0.862, Validation Accuracy:  0.813, Loss:  0.151
    Epoch   6 Batch  252/269 - Train Accuracy:  0.833, Validation Accuracy:  0.813, Loss:  0.146
    Epoch   6 Batch  253/269 - Train Accuracy:  0.830, Validation Accuracy:  0.817, Loss:  0.159
    Epoch   6 Batch  254/269 - Train Accuracy:  0.837, Validation Accuracy:  0.820, Loss:  0.155
    Epoch   6 Batch  255/269 - Train Accuracy:  0.838, Validation Accuracy:  0.818, Loss:  0.156
    Epoch   6 Batch  256/269 - Train Accuracy:  0.813, Validation Accuracy:  0.815, Loss:  0.161
    Epoch   6 Batch  257/269 - Train Accuracy:  0.807, Validation Accuracy:  0.816, Loss:  0.172
    Epoch   6 Batch  258/269 - Train Accuracy:  0.827, Validation Accuracy:  0.816, Loss:  0.164
    Epoch   6 Batch  259/269 - Train Accuracy:  0.840, Validation Accuracy:  0.819, Loss:  0.164
    Epoch   6 Batch  260/269 - Train Accuracy:  0.808, Validation Accuracy:  0.823, Loss:  0.167
    Epoch   6 Batch  261/269 - Train Accuracy:  0.830, Validation Accuracy:  0.826, Loss:  0.168
    Epoch   6 Batch  262/269 - Train Accuracy:  0.831, Validation Accuracy:  0.826, Loss:  0.149
    Epoch   6 Batch  263/269 - Train Accuracy:  0.837, Validation Accuracy:  0.821, Loss:  0.163
    Epoch   6 Batch  264/269 - Train Accuracy:  0.798, Validation Accuracy:  0.823, Loss:  0.169
    Epoch   6 Batch  265/269 - Train Accuracy:  0.829, Validation Accuracy:  0.815, Loss:  0.164
    Epoch   6 Batch  266/269 - Train Accuracy:  0.822, Validation Accuracy:  0.821, Loss:  0.150
    Epoch   6 Batch  267/269 - Train Accuracy:  0.832, Validation Accuracy:  0.825, Loss:  0.167
    Epoch   7 Batch    0/269 - Train Accuracy:  0.813, Validation Accuracy:  0.817, Loss:  0.167
    Epoch   7 Batch    1/269 - Train Accuracy:  0.818, Validation Accuracy:  0.816, Loss:  0.158
    Epoch   7 Batch    2/269 - Train Accuracy:  0.825, Validation Accuracy:  0.819, Loss:  0.161
    Epoch   7 Batch    3/269 - Train Accuracy:  0.831, Validation Accuracy:  0.826, Loss:  0.158
    Epoch   7 Batch    4/269 - Train Accuracy:  0.817, Validation Accuracy:  0.824, Loss:  0.155
    Epoch   7 Batch    5/269 - Train Accuracy:  0.797, Validation Accuracy:  0.822, Loss:  0.158
    Epoch   7 Batch    6/269 - Train Accuracy:  0.830, Validation Accuracy:  0.828, Loss:  0.150
    Epoch   7 Batch    7/269 - Train Accuracy:  0.840, Validation Accuracy:  0.834, Loss:  0.154
    Epoch   7 Batch    8/269 - Train Accuracy:  0.826, Validation Accuracy:  0.831, Loss:  0.158
    Epoch   7 Batch    9/269 - Train Accuracy:  0.821, Validation Accuracy:  0.830, Loss:  0.161
    Epoch   7 Batch   10/269 - Train Accuracy:  0.827, Validation Accuracy:  0.824, Loss:  0.152
    Epoch   7 Batch   11/269 - Train Accuracy:  0.820, Validation Accuracy:  0.821, Loss:  0.164
    Epoch   7 Batch   12/269 - Train Accuracy:  0.825, Validation Accuracy:  0.822, Loss:  0.171
    Epoch   7 Batch   13/269 - Train Accuracy:  0.852, Validation Accuracy:  0.819, Loss:  0.137
    Epoch   7 Batch   14/269 - Train Accuracy:  0.811, Validation Accuracy:  0.814, Loss:  0.155
    Epoch   7 Batch   15/269 - Train Accuracy:  0.841, Validation Accuracy:  0.825, Loss:  0.145
    Epoch   7 Batch   16/269 - Train Accuracy:  0.834, Validation Accuracy:  0.823, Loss:  0.160
    Epoch   7 Batch   17/269 - Train Accuracy:  0.832, Validation Accuracy:  0.819, Loss:  0.145
    Epoch   7 Batch   18/269 - Train Accuracy:  0.832, Validation Accuracy:  0.824, Loss:  0.158
    Epoch   7 Batch   19/269 - Train Accuracy:  0.865, Validation Accuracy:  0.829, Loss:  0.145
    Epoch   7 Batch   20/269 - Train Accuracy:  0.829, Validation Accuracy:  0.824, Loss:  0.160
    Epoch   7 Batch   21/269 - Train Accuracy:  0.827, Validation Accuracy:  0.827, Loss:  0.164
    Epoch   7 Batch   22/269 - Train Accuracy:  0.840, Validation Accuracy:  0.824, Loss:  0.147
    Epoch   7 Batch   23/269 - Train Accuracy:  0.834, Validation Accuracy:  0.831, Loss:  0.164
    Epoch   7 Batch   24/269 - Train Accuracy:  0.837, Validation Accuracy:  0.829, Loss:  0.151
    Epoch   7 Batch   25/269 - Train Accuracy:  0.822, Validation Accuracy:  0.831, Loss:  0.166
    Epoch   7 Batch   26/269 - Train Accuracy:  0.839, Validation Accuracy:  0.827, Loss:  0.146
    Epoch   7 Batch   27/269 - Train Accuracy:  0.829, Validation Accuracy:  0.833, Loss:  0.159
    Epoch   7 Batch   28/269 - Train Accuracy:  0.807, Validation Accuracy:  0.823, Loss:  0.167
    Epoch   7 Batch   29/269 - Train Accuracy:  0.840, Validation Accuracy:  0.827, Loss:  0.161
    Epoch   7 Batch   30/269 - Train Accuracy:  0.838, Validation Accuracy:  0.828, Loss:  0.151
    Epoch   7 Batch   31/269 - Train Accuracy:  0.838, Validation Accuracy:  0.827, Loss:  0.146
    Epoch   7 Batch   32/269 - Train Accuracy:  0.826, Validation Accuracy:  0.823, Loss:  0.157
    Epoch   7 Batch   33/269 - Train Accuracy:  0.833, Validation Accuracy:  0.821, Loss:  0.146
    Epoch   7 Batch   34/269 - Train Accuracy:  0.828, Validation Accuracy:  0.811, Loss:  0.159
    Epoch   7 Batch   35/269 - Train Accuracy:  0.820, Validation Accuracy:  0.813, Loss:  0.190
    Epoch   7 Batch   36/269 - Train Accuracy:  0.807, Validation Accuracy:  0.811, Loss:  0.153
    Epoch   7 Batch   37/269 - Train Accuracy:  0.839, Validation Accuracy:  0.816, Loss:  0.163
    Epoch   7 Batch   38/269 - Train Accuracy:  0.825, Validation Accuracy:  0.811, Loss:  0.160
    Epoch   7 Batch   39/269 - Train Accuracy:  0.834, Validation Accuracy:  0.817, Loss:  0.168
    Epoch   7 Batch   40/269 - Train Accuracy:  0.822, Validation Accuracy:  0.813, Loss:  0.168
    Epoch   7 Batch   41/269 - Train Accuracy:  0.804, Validation Accuracy:  0.803, Loss:  0.167
    Epoch   7 Batch   42/269 - Train Accuracy:  0.844, Validation Accuracy:  0.811, Loss:  0.162
    Epoch   7 Batch   43/269 - Train Accuracy:  0.819, Validation Accuracy:  0.798, Loss:  0.159
    Epoch   7 Batch   44/269 - Train Accuracy:  0.826, Validation Accuracy:  0.814, Loss:  0.176
    Epoch   7 Batch   45/269 - Train Accuracy:  0.818, Validation Accuracy:  0.814, Loss:  0.155
    Epoch   7 Batch   46/269 - Train Accuracy:  0.824, Validation Accuracy:  0.819, Loss:  0.172
    Epoch   7 Batch   47/269 - Train Accuracy:  0.840, Validation Accuracy:  0.817, Loss:  0.148
    Epoch   7 Batch   48/269 - Train Accuracy:  0.821, Validation Accuracy:  0.809, Loss:  0.169
    Epoch   7 Batch   49/269 - Train Accuracy:  0.811, Validation Accuracy:  0.800, Loss:  0.163
    Epoch   7 Batch   50/269 - Train Accuracy:  0.808, Validation Accuracy:  0.815, Loss:  0.185
    Epoch   7 Batch   51/269 - Train Accuracy:  0.807, Validation Accuracy:  0.808, Loss:  0.178
    Epoch   7 Batch   52/269 - Train Accuracy:  0.815, Validation Accuracy:  0.805, Loss:  0.173
    Epoch   7 Batch   53/269 - Train Accuracy:  0.791, Validation Accuracy:  0.790, Loss:  0.172
    Epoch   7 Batch   54/269 - Train Accuracy:  0.822, Validation Accuracy:  0.804, Loss:  0.177
    Epoch   7 Batch   55/269 - Train Accuracy:  0.815, Validation Accuracy:  0.801, Loss:  0.164
    Epoch   7 Batch   56/269 - Train Accuracy:  0.829, Validation Accuracy:  0.811, Loss:  0.186
    Epoch   7 Batch   57/269 - Train Accuracy:  0.819, Validation Accuracy:  0.811, Loss:  0.177
    Epoch   7 Batch   58/269 - Train Accuracy:  0.804, Validation Accuracy:  0.806, Loss:  0.164
    Epoch   7 Batch   59/269 - Train Accuracy:  0.840, Validation Accuracy:  0.813, Loss:  0.149
    Epoch   7 Batch   60/269 - Train Accuracy:  0.812, Validation Accuracy:  0.820, Loss:  0.159
    Epoch   7 Batch   61/269 - Train Accuracy:  0.835, Validation Accuracy:  0.820, Loss:  0.145
    Epoch   7 Batch   62/269 - Train Accuracy:  0.826, Validation Accuracy:  0.819, Loss:  0.165
    Epoch   7 Batch   63/269 - Train Accuracy:  0.813, Validation Accuracy:  0.813, Loss:  0.164
    Epoch   7 Batch   64/269 - Train Accuracy:  0.826, Validation Accuracy:  0.826, Loss:  0.153
    Epoch   7 Batch   65/269 - Train Accuracy:  0.820, Validation Accuracy:  0.818, Loss:  0.154
    Epoch   7 Batch   66/269 - Train Accuracy:  0.801, Validation Accuracy:  0.814, Loss:  0.159
    Epoch   7 Batch   67/269 - Train Accuracy:  0.821, Validation Accuracy:  0.817, Loss:  0.169
    Epoch   7 Batch   68/269 - Train Accuracy:  0.822, Validation Accuracy:  0.813, Loss:  0.164
    Epoch   7 Batch   69/269 - Train Accuracy:  0.796, Validation Accuracy:  0.811, Loss:  0.181
    Epoch   7 Batch   70/269 - Train Accuracy:  0.827, Validation Accuracy:  0.824, Loss:  0.157
    Epoch   7 Batch   71/269 - Train Accuracy:  0.831, Validation Accuracy:  0.823, Loss:  0.168
    Epoch   7 Batch   72/269 - Train Accuracy:  0.821, Validation Accuracy:  0.820, Loss:  0.164
    Epoch   7 Batch   73/269 - Train Accuracy:  0.831, Validation Accuracy:  0.821, Loss:  0.164
    Epoch   7 Batch   74/269 - Train Accuracy:  0.824, Validation Accuracy:  0.820, Loss:  0.157
    Epoch   7 Batch   75/269 - Train Accuracy:  0.823, Validation Accuracy:  0.822, Loss:  0.162
    Epoch   7 Batch   76/269 - Train Accuracy:  0.821, Validation Accuracy:  0.821, Loss:  0.153
    Epoch   7 Batch   77/269 - Train Accuracy:  0.840, Validation Accuracy:  0.824, Loss:  0.162
    Epoch   7 Batch   78/269 - Train Accuracy:  0.829, Validation Accuracy:  0.818, Loss:  0.151
    Epoch   7 Batch   79/269 - Train Accuracy:  0.824, Validation Accuracy:  0.814, Loss:  0.160
    Epoch   7 Batch   80/269 - Train Accuracy:  0.835, Validation Accuracy:  0.811, Loss:  0.152
    Epoch   7 Batch   81/269 - Train Accuracy:  0.818, Validation Accuracy:  0.812, Loss:  0.160
    Epoch   7 Batch   82/269 - Train Accuracy:  0.843, Validation Accuracy:  0.817, Loss:  0.140
    Epoch   7 Batch   83/269 - Train Accuracy:  0.832, Validation Accuracy:  0.817, Loss:  0.171
    Epoch   7 Batch   84/269 - Train Accuracy:  0.836, Validation Accuracy:  0.816, Loss:  0.155
    Epoch   7 Batch   85/269 - Train Accuracy:  0.840, Validation Accuracy:  0.818, Loss:  0.148
    Epoch   7 Batch   86/269 - Train Accuracy:  0.818, Validation Accuracy:  0.822, Loss:  0.149
    Epoch   7 Batch   87/269 - Train Accuracy:  0.815, Validation Accuracy:  0.820, Loss:  0.167
    Epoch   7 Batch   88/269 - Train Accuracy:  0.834, Validation Accuracy:  0.818, Loss:  0.141
    Epoch   7 Batch   89/269 - Train Accuracy:  0.842, Validation Accuracy:  0.820, Loss:  0.140
    Epoch   7 Batch   90/269 - Train Accuracy:  0.811, Validation Accuracy:  0.821, Loss:  0.155
    Epoch   7 Batch   91/269 - Train Accuracy:  0.821, Validation Accuracy:  0.828, Loss:  0.140
    Epoch   7 Batch   92/269 - Train Accuracy:  0.845, Validation Accuracy:  0.828, Loss:  0.136
    Epoch   7 Batch   93/269 - Train Accuracy:  0.835, Validation Accuracy:  0.824, Loss:  0.137
    Epoch   7 Batch   94/269 - Train Accuracy:  0.832, Validation Accuracy:  0.833, Loss:  0.162
    Epoch   7 Batch   95/269 - Train Accuracy:  0.844, Validation Accuracy:  0.826, Loss:  0.138
    Epoch   7 Batch   96/269 - Train Accuracy:  0.812, Validation Accuracy:  0.826, Loss:  0.157
    Epoch   7 Batch   97/269 - Train Accuracy:  0.834, Validation Accuracy:  0.824, Loss:  0.140
    Epoch   7 Batch   98/269 - Train Accuracy:  0.834, Validation Accuracy:  0.822, Loss:  0.152
    Epoch   7 Batch   99/269 - Train Accuracy:  0.824, Validation Accuracy:  0.828, Loss:  0.147
    Epoch   7 Batch  100/269 - Train Accuracy:  0.862, Validation Accuracy:  0.827, Loss:  0.141
    Epoch   7 Batch  101/269 - Train Accuracy:  0.814, Validation Accuracy:  0.826, Loss:  0.155
    Epoch   7 Batch  102/269 - Train Accuracy:  0.835, Validation Accuracy:  0.829, Loss:  0.143
    Epoch   7 Batch  103/269 - Train Accuracy:  0.822, Validation Accuracy:  0.827, Loss:  0.151
    Epoch   7 Batch  104/269 - Train Accuracy:  0.830, Validation Accuracy:  0.828, Loss:  0.150
    Epoch   7 Batch  105/269 - Train Accuracy:  0.826, Validation Accuracy:  0.824, Loss:  0.146
    Epoch   7 Batch  106/269 - Train Accuracy:  0.825, Validation Accuracy:  0.828, Loss:  0.141
    Epoch   7 Batch  107/269 - Train Accuracy:  0.826, Validation Accuracy:  0.837, Loss:  0.142
    Epoch   7 Batch  108/269 - Train Accuracy:  0.836, Validation Accuracy:  0.826, Loss:  0.139
    Epoch   7 Batch  109/269 - Train Accuracy:  0.816, Validation Accuracy:  0.827, Loss:  0.151
    Epoch   7 Batch  110/269 - Train Accuracy:  0.826, Validation Accuracy:  0.822, Loss:  0.126
    Epoch   7 Batch  111/269 - Train Accuracy:  0.821, Validation Accuracy:  0.826, Loss:  0.152
    Epoch   7 Batch  112/269 - Train Accuracy:  0.831, Validation Accuracy:  0.822, Loss:  0.143
    Epoch   7 Batch  113/269 - Train Accuracy:  0.837, Validation Accuracy:  0.826, Loss:  0.138
    Epoch   7 Batch  114/269 - Train Accuracy:  0.837, Validation Accuracy:  0.826, Loss:  0.141
    Epoch   7 Batch  115/269 - Train Accuracy:  0.816, Validation Accuracy:  0.820, Loss:  0.141
    Epoch   7 Batch  116/269 - Train Accuracy:  0.846, Validation Accuracy:  0.823, Loss:  0.139
    Epoch   7 Batch  117/269 - Train Accuracy:  0.816, Validation Accuracy:  0.819, Loss:  0.138
    Epoch   7 Batch  118/269 - Train Accuracy:  0.857, Validation Accuracy:  0.814, Loss:  0.129
    Epoch   7 Batch  119/269 - Train Accuracy:  0.833, Validation Accuracy:  0.825, Loss:  0.157
    Epoch   7 Batch  120/269 - Train Accuracy:  0.842, Validation Accuracy:  0.827, Loss:  0.143
    Epoch   7 Batch  121/269 - Train Accuracy:  0.847, Validation Accuracy:  0.825, Loss:  0.136
    Epoch   7 Batch  122/269 - Train Accuracy:  0.835, Validation Accuracy:  0.822, Loss:  0.138
    Epoch   7 Batch  123/269 - Train Accuracy:  0.820, Validation Accuracy:  0.829, Loss:  0.147
    Epoch   7 Batch  124/269 - Train Accuracy:  0.825, Validation Accuracy:  0.833, Loss:  0.132
    Epoch   7 Batch  125/269 - Train Accuracy:  0.829, Validation Accuracy:  0.822, Loss:  0.129
    Epoch   7 Batch  126/269 - Train Accuracy:  0.826, Validation Accuracy:  0.827, Loss:  0.147
    Epoch   7 Batch  127/269 - Train Accuracy:  0.823, Validation Accuracy:  0.828, Loss:  0.145
    Epoch   7 Batch  128/269 - Train Accuracy:  0.831, Validation Accuracy:  0.832, Loss:  0.145
    Epoch   7 Batch  129/269 - Train Accuracy:  0.815, Validation Accuracy:  0.826, Loss:  0.137
    Epoch   7 Batch  130/269 - Train Accuracy:  0.817, Validation Accuracy:  0.830, Loss:  0.144
    Epoch   7 Batch  131/269 - Train Accuracy:  0.816, Validation Accuracy:  0.822, Loss:  0.148
    Epoch   7 Batch  132/269 - Train Accuracy:  0.797, Validation Accuracy:  0.822, Loss:  0.147
    Epoch   7 Batch  133/269 - Train Accuracy:  0.844, Validation Accuracy:  0.828, Loss:  0.127
    Epoch   7 Batch  134/269 - Train Accuracy:  0.816, Validation Accuracy:  0.822, Loss:  0.144
    Epoch   7 Batch  135/269 - Train Accuracy:  0.831, Validation Accuracy:  0.827, Loss:  0.145
    Epoch   7 Batch  136/269 - Train Accuracy:  0.818, Validation Accuracy:  0.826, Loss:  0.145
    Epoch   7 Batch  137/269 - Train Accuracy:  0.837, Validation Accuracy:  0.821, Loss:  0.151
    Epoch   7 Batch  138/269 - Train Accuracy:  0.830, Validation Accuracy:  0.828, Loss:  0.146
    Epoch   7 Batch  139/269 - Train Accuracy:  0.839, Validation Accuracy:  0.826, Loss:  0.133
    Epoch   7 Batch  140/269 - Train Accuracy:  0.853, Validation Accuracy:  0.826, Loss:  0.145
    Epoch   7 Batch  141/269 - Train Accuracy:  0.833, Validation Accuracy:  0.832, Loss:  0.149
    Epoch   7 Batch  142/269 - Train Accuracy:  0.845, Validation Accuracy:  0.826, Loss:  0.132
    Epoch   7 Batch  143/269 - Train Accuracy:  0.860, Validation Accuracy:  0.825, Loss:  0.133
    Epoch   7 Batch  144/269 - Train Accuracy:  0.842, Validation Accuracy:  0.827, Loss:  0.132
    Epoch   7 Batch  145/269 - Train Accuracy:  0.835, Validation Accuracy:  0.831, Loss:  0.136
    Epoch   7 Batch  146/269 - Train Accuracy:  0.850, Validation Accuracy:  0.835, Loss:  0.132
    Epoch   7 Batch  147/269 - Train Accuracy:  0.851, Validation Accuracy:  0.835, Loss:  0.138
    Epoch   7 Batch  148/269 - Train Accuracy:  0.836, Validation Accuracy:  0.836, Loss:  0.136
    Epoch   7 Batch  149/269 - Train Accuracy:  0.821, Validation Accuracy:  0.838, Loss:  0.142
    Epoch   7 Batch  150/269 - Train Accuracy:  0.840, Validation Accuracy:  0.828, Loss:  0.139
    Epoch   7 Batch  151/269 - Train Accuracy:  0.852, Validation Accuracy:  0.832, Loss:  0.136
    Epoch   7 Batch  152/269 - Train Accuracy:  0.824, Validation Accuracy:  0.833, Loss:  0.140
    Epoch   7 Batch  153/269 - Train Accuracy:  0.840, Validation Accuracy:  0.831, Loss:  0.130
    Epoch   7 Batch  154/269 - Train Accuracy:  0.846, Validation Accuracy:  0.829, Loss:  0.132
    Epoch   7 Batch  155/269 - Train Accuracy:  0.849, Validation Accuracy:  0.832, Loss:  0.131
    Epoch   7 Batch  156/269 - Train Accuracy:  0.842, Validation Accuracy:  0.833, Loss:  0.146
    Epoch   7 Batch  157/269 - Train Accuracy:  0.850, Validation Accuracy:  0.830, Loss:  0.134
    Epoch   7 Batch  158/269 - Train Accuracy:  0.843, Validation Accuracy:  0.838, Loss:  0.143
    Epoch   7 Batch  159/269 - Train Accuracy:  0.831, Validation Accuracy:  0.834, Loss:  0.138
    Epoch   7 Batch  160/269 - Train Accuracy:  0.841, Validation Accuracy:  0.831, Loss:  0.141
    Epoch   7 Batch  161/269 - Train Accuracy:  0.832, Validation Accuracy:  0.826, Loss:  0.131
    Epoch   7 Batch  162/269 - Train Accuracy:  0.847, Validation Accuracy:  0.830, Loss:  0.137
    Epoch   7 Batch  163/269 - Train Accuracy:  0.857, Validation Accuracy:  0.829, Loss:  0.148
    Epoch   7 Batch  164/269 - Train Accuracy:  0.837, Validation Accuracy:  0.835, Loss:  0.135
    Epoch   7 Batch  165/269 - Train Accuracy:  0.827, Validation Accuracy:  0.828, Loss:  0.131
    Epoch   7 Batch  166/269 - Train Accuracy:  0.851, Validation Accuracy:  0.835, Loss:  0.128
    Epoch   7 Batch  167/269 - Train Accuracy:  0.847, Validation Accuracy:  0.829, Loss:  0.127
    Epoch   7 Batch  168/269 - Train Accuracy:  0.831, Validation Accuracy:  0.829, Loss:  0.134
    Epoch   7 Batch  169/269 - Train Accuracy:  0.837, Validation Accuracy:  0.834, Loss:  0.135
    Epoch   7 Batch  170/269 - Train Accuracy:  0.834, Validation Accuracy:  0.826, Loss:  0.124
    Epoch   7 Batch  171/269 - Train Accuracy:  0.835, Validation Accuracy:  0.824, Loss:  0.139
    Epoch   7 Batch  172/269 - Train Accuracy:  0.829, Validation Accuracy:  0.832, Loss:  0.142
    Epoch   7 Batch  173/269 - Train Accuracy:  0.848, Validation Accuracy:  0.835, Loss:  0.123
    Epoch   7 Batch  174/269 - Train Accuracy:  0.831, Validation Accuracy:  0.835, Loss:  0.126
    Epoch   7 Batch  175/269 - Train Accuracy:  0.839, Validation Accuracy:  0.842, Loss:  0.149
    Epoch   7 Batch  176/269 - Train Accuracy:  0.834, Validation Accuracy:  0.833, Loss:  0.141
    Epoch   7 Batch  177/269 - Train Accuracy:  0.847, Validation Accuracy:  0.829, Loss:  0.125
    Epoch   7 Batch  178/269 - Train Accuracy:  0.842, Validation Accuracy:  0.835, Loss:  0.133
    Epoch   7 Batch  179/269 - Train Accuracy:  0.834, Validation Accuracy:  0.828, Loss:  0.123
    Epoch   7 Batch  180/269 - Train Accuracy:  0.839, Validation Accuracy:  0.830, Loss:  0.130
    Epoch   7 Batch  181/269 - Train Accuracy:  0.819, Validation Accuracy:  0.827, Loss:  0.137
    Epoch   7 Batch  182/269 - Train Accuracy:  0.828, Validation Accuracy:  0.830, Loss:  0.132
    Epoch   7 Batch  183/269 - Train Accuracy:  0.864, Validation Accuracy:  0.841, Loss:  0.120
    Epoch   7 Batch  184/269 - Train Accuracy:  0.853, Validation Accuracy:  0.837, Loss:  0.127
    Epoch   7 Batch  185/269 - Train Accuracy:  0.841, Validation Accuracy:  0.840, Loss:  0.130
    Epoch   7 Batch  186/269 - Train Accuracy:  0.816, Validation Accuracy:  0.827, Loss:  0.127
    Epoch   7 Batch  187/269 - Train Accuracy:  0.835, Validation Accuracy:  0.837, Loss:  0.135
    Epoch   7 Batch  188/269 - Train Accuracy:  0.851, Validation Accuracy:  0.834, Loss:  0.122
    Epoch   7 Batch  189/269 - Train Accuracy:  0.840, Validation Accuracy:  0.836, Loss:  0.130
    Epoch   7 Batch  190/269 - Train Accuracy:  0.846, Validation Accuracy:  0.834, Loss:  0.122
    Epoch   7 Batch  191/269 - Train Accuracy:  0.843, Validation Accuracy:  0.828, Loss:  0.126
    Epoch   7 Batch  192/269 - Train Accuracy:  0.843, Validation Accuracy:  0.830, Loss:  0.134
    Epoch   7 Batch  193/269 - Train Accuracy:  0.838, Validation Accuracy:  0.819, Loss:  0.124
    Epoch   7 Batch  194/269 - Train Accuracy:  0.841, Validation Accuracy:  0.830, Loss:  0.141
    Epoch   7 Batch  195/269 - Train Accuracy:  0.822, Validation Accuracy:  0.829, Loss:  0.127
    Epoch   7 Batch  196/269 - Train Accuracy:  0.815, Validation Accuracy:  0.824, Loss:  0.134
    Epoch   7 Batch  197/269 - Train Accuracy:  0.827, Validation Accuracy:  0.829, Loss:  0.140
    Epoch   7 Batch  198/269 - Train Accuracy:  0.819, Validation Accuracy:  0.829, Loss:  0.130
    Epoch   7 Batch  199/269 - Train Accuracy:  0.844, Validation Accuracy:  0.829, Loss:  0.139
    Epoch   7 Batch  200/269 - Train Accuracy:  0.845, Validation Accuracy:  0.830, Loss:  0.136
    Epoch   7 Batch  201/269 - Train Accuracy:  0.840, Validation Accuracy:  0.835, Loss:  0.135
    Epoch   7 Batch  202/269 - Train Accuracy:  0.845, Validation Accuracy:  0.836, Loss:  0.132
    Epoch   7 Batch  203/269 - Train Accuracy:  0.849, Validation Accuracy:  0.828, Loss:  0.137
    Epoch   7 Batch  204/269 - Train Accuracy:  0.836, Validation Accuracy:  0.836, Loss:  0.134
    Epoch   7 Batch  205/269 - Train Accuracy:  0.839, Validation Accuracy:  0.831, Loss:  0.128
    Epoch   7 Batch  206/269 - Train Accuracy:  0.836, Validation Accuracy:  0.836, Loss:  0.136
    Epoch   7 Batch  207/269 - Train Accuracy:  0.854, Validation Accuracy:  0.841, Loss:  0.123
    Epoch   7 Batch  208/269 - Train Accuracy:  0.834, Validation Accuracy:  0.837, Loss:  0.128
    Epoch   7 Batch  209/269 - Train Accuracy:  0.850, Validation Accuracy:  0.843, Loss:  0.123
    Epoch   7 Batch  210/269 - Train Accuracy:  0.853, Validation Accuracy:  0.834, Loss:  0.125
    Epoch   7 Batch  211/269 - Train Accuracy:  0.851, Validation Accuracy:  0.842, Loss:  0.128
    Epoch   7 Batch  212/269 - Train Accuracy:  0.850, Validation Accuracy:  0.838, Loss:  0.129
    Epoch   7 Batch  213/269 - Train Accuracy:  0.849, Validation Accuracy:  0.840, Loss:  0.121
    Epoch   7 Batch  214/269 - Train Accuracy:  0.848, Validation Accuracy:  0.838, Loss:  0.126
    Epoch   7 Batch  215/269 - Train Accuracy:  0.856, Validation Accuracy:  0.839, Loss:  0.116
    Epoch   7 Batch  216/269 - Train Accuracy:  0.837, Validation Accuracy:  0.843, Loss:  0.142
    Epoch   7 Batch  217/269 - Train Accuracy:  0.838, Validation Accuracy:  0.841, Loss:  0.122
    Epoch   7 Batch  218/269 - Train Accuracy:  0.852, Validation Accuracy:  0.841, Loss:  0.120
    Epoch   7 Batch  219/269 - Train Accuracy:  0.842, Validation Accuracy:  0.846, Loss:  0.126
    Epoch   7 Batch  220/269 - Train Accuracy:  0.843, Validation Accuracy:  0.847, Loss:  0.114
    Epoch   7 Batch  221/269 - Train Accuracy:  0.864, Validation Accuracy:  0.837, Loss:  0.117
    Epoch   7 Batch  222/269 - Train Accuracy:  0.859, Validation Accuracy:  0.844, Loss:  0.109
    Epoch   7 Batch  223/269 - Train Accuracy:  0.853, Validation Accuracy:  0.849, Loss:  0.109
    Epoch   7 Batch  224/269 - Train Accuracy:  0.858, Validation Accuracy:  0.848, Loss:  0.127
    Epoch   7 Batch  225/269 - Train Accuracy:  0.846, Validation Accuracy:  0.853, Loss:  0.110
    Epoch   7 Batch  226/269 - Train Accuracy:  0.847, Validation Accuracy:  0.847, Loss:  0.120
    Epoch   7 Batch  227/269 - Train Accuracy:  0.875, Validation Accuracy:  0.849, Loss:  0.114
    Epoch   7 Batch  228/269 - Train Accuracy:  0.849, Validation Accuracy:  0.849, Loss:  0.115
    Epoch   7 Batch  229/269 - Train Accuracy:  0.846, Validation Accuracy:  0.846, Loss:  0.119
    Epoch   7 Batch  230/269 - Train Accuracy:  0.844, Validation Accuracy:  0.846, Loss:  0.114
    Epoch   7 Batch  231/269 - Train Accuracy:  0.826, Validation Accuracy:  0.844, Loss:  0.130
    Epoch   7 Batch  232/269 - Train Accuracy:  0.834, Validation Accuracy:  0.847, Loss:  0.124
    Epoch   7 Batch  233/269 - Train Accuracy:  0.858, Validation Accuracy:  0.847, Loss:  0.124
    Epoch   7 Batch  234/269 - Train Accuracy:  0.839, Validation Accuracy:  0.843, Loss:  0.118
    Epoch   7 Batch  235/269 - Train Accuracy:  0.861, Validation Accuracy:  0.839, Loss:  0.104
    Epoch   7 Batch  236/269 - Train Accuracy:  0.827, Validation Accuracy:  0.842, Loss:  0.115
    Epoch   7 Batch  237/269 - Train Accuracy:  0.848, Validation Accuracy:  0.838, Loss:  0.119
    Epoch   7 Batch  238/269 - Train Accuracy:  0.846, Validation Accuracy:  0.846, Loss:  0.113
    Epoch   7 Batch  239/269 - Train Accuracy:  0.859, Validation Accuracy:  0.843, Loss:  0.112
    Epoch   7 Batch  240/269 - Train Accuracy:  0.861, Validation Accuracy:  0.839, Loss:  0.102
    Epoch   7 Batch  241/269 - Train Accuracy:  0.833, Validation Accuracy:  0.840, Loss:  0.123
    Epoch   7 Batch  242/269 - Train Accuracy:  0.834, Validation Accuracy:  0.843, Loss:  0.109
    Epoch   7 Batch  243/269 - Train Accuracy:  0.869, Validation Accuracy:  0.840, Loss:  0.112
    Epoch   7 Batch  244/269 - Train Accuracy:  0.867, Validation Accuracy:  0.834, Loss:  0.120
    Epoch   7 Batch  245/269 - Train Accuracy:  0.837, Validation Accuracy:  0.839, Loss:  0.122
    Epoch   7 Batch  246/269 - Train Accuracy:  0.830, Validation Accuracy:  0.844, Loss:  0.118
    Epoch   7 Batch  247/269 - Train Accuracy:  0.850, Validation Accuracy:  0.843, Loss:  0.112
    Epoch   7 Batch  248/269 - Train Accuracy:  0.836, Validation Accuracy:  0.844, Loss:  0.110
    Epoch   7 Batch  249/269 - Train Accuracy:  0.876, Validation Accuracy:  0.835, Loss:  0.104
    Epoch   7 Batch  250/269 - Train Accuracy:  0.845, Validation Accuracy:  0.840, Loss:  0.108
    Epoch   7 Batch  251/269 - Train Accuracy:  0.879, Validation Accuracy:  0.839, Loss:  0.105
    Epoch   7 Batch  252/269 - Train Accuracy:  0.858, Validation Accuracy:  0.840, Loss:  0.103
    Epoch   7 Batch  253/269 - Train Accuracy:  0.837, Validation Accuracy:  0.842, Loss:  0.120
    Epoch   7 Batch  254/269 - Train Accuracy:  0.849, Validation Accuracy:  0.840, Loss:  0.112
    Epoch   7 Batch  255/269 - Train Accuracy:  0.852, Validation Accuracy:  0.843, Loss:  0.110
    Epoch   7 Batch  256/269 - Train Accuracy:  0.830, Validation Accuracy:  0.848, Loss:  0.113
    Epoch   7 Batch  257/269 - Train Accuracy:  0.824, Validation Accuracy:  0.847, Loss:  0.124
    Epoch   7 Batch  258/269 - Train Accuracy:  0.841, Validation Accuracy:  0.840, Loss:  0.110
    Epoch   7 Batch  259/269 - Train Accuracy:  0.843, Validation Accuracy:  0.839, Loss:  0.117
    Epoch   7 Batch  260/269 - Train Accuracy:  0.839, Validation Accuracy:  0.839, Loss:  0.125
    Epoch   7 Batch  261/269 - Train Accuracy:  0.840, Validation Accuracy:  0.854, Loss:  0.111
    Epoch   7 Batch  262/269 - Train Accuracy:  0.852, Validation Accuracy:  0.844, Loss:  0.111
    Epoch   7 Batch  263/269 - Train Accuracy:  0.847, Validation Accuracy:  0.842, Loss:  0.119
    Epoch   7 Batch  264/269 - Train Accuracy:  0.827, Validation Accuracy:  0.842, Loss:  0.127
    Epoch   7 Batch  265/269 - Train Accuracy:  0.853, Validation Accuracy:  0.840, Loss:  0.110
    Epoch   7 Batch  266/269 - Train Accuracy:  0.840, Validation Accuracy:  0.835, Loss:  0.107
    Epoch   7 Batch  267/269 - Train Accuracy:  0.850, Validation Accuracy:  0.844, Loss:  0.120
    Epoch   8 Batch    0/269 - Train Accuracy:  0.847, Validation Accuracy:  0.839, Loss:  0.122
    Epoch   8 Batch    1/269 - Train Accuracy:  0.836, Validation Accuracy:  0.840, Loss:  0.112
    Epoch   8 Batch    2/269 - Train Accuracy:  0.851, Validation Accuracy:  0.837, Loss:  0.116
    Epoch   8 Batch    3/269 - Train Accuracy:  0.845, Validation Accuracy:  0.840, Loss:  0.112
    Epoch   8 Batch    4/269 - Train Accuracy:  0.834, Validation Accuracy:  0.838, Loss:  0.106
    Epoch   8 Batch    5/269 - Train Accuracy:  0.819, Validation Accuracy:  0.841, Loss:  0.116
    Epoch   8 Batch    6/269 - Train Accuracy:  0.845, Validation Accuracy:  0.842, Loss:  0.105
    Epoch   8 Batch    7/269 - Train Accuracy:  0.859, Validation Accuracy:  0.845, Loss:  0.108
    Epoch   8 Batch    8/269 - Train Accuracy:  0.859, Validation Accuracy:  0.843, Loss:  0.115
    Epoch   8 Batch    9/269 - Train Accuracy:  0.846, Validation Accuracy:  0.848, Loss:  0.112
    Epoch   8 Batch   10/269 - Train Accuracy:  0.865, Validation Accuracy:  0.844, Loss:  0.108
    Epoch   8 Batch   11/269 - Train Accuracy:  0.848, Validation Accuracy:  0.848, Loss:  0.115
    Epoch   8 Batch   12/269 - Train Accuracy:  0.853, Validation Accuracy:  0.846, Loss:  0.120
    Epoch   8 Batch   13/269 - Train Accuracy:  0.858, Validation Accuracy:  0.839, Loss:  0.096
    Epoch   8 Batch   14/269 - Train Accuracy:  0.844, Validation Accuracy:  0.843, Loss:  0.111
    Epoch   8 Batch   15/269 - Train Accuracy:  0.854, Validation Accuracy:  0.838, Loss:  0.095
    Epoch   8 Batch   16/269 - Train Accuracy:  0.854, Validation Accuracy:  0.843, Loss:  0.112
    Epoch   8 Batch   17/269 - Train Accuracy:  0.854, Validation Accuracy:  0.845, Loss:  0.099
    Epoch   8 Batch   18/269 - Train Accuracy:  0.842, Validation Accuracy:  0.842, Loss:  0.110
    Epoch   8 Batch   19/269 - Train Accuracy:  0.888, Validation Accuracy:  0.839, Loss:  0.099
    Epoch   8 Batch   20/269 - Train Accuracy:  0.833, Validation Accuracy:  0.835, Loss:  0.110
    Epoch   8 Batch   21/269 - Train Accuracy:  0.858, Validation Accuracy:  0.846, Loss:  0.124
    Epoch   8 Batch   22/269 - Train Accuracy:  0.869, Validation Accuracy:  0.846, Loss:  0.101
    Epoch   8 Batch   23/269 - Train Accuracy:  0.855, Validation Accuracy:  0.846, Loss:  0.113
    Epoch   8 Batch   24/269 - Train Accuracy:  0.849, Validation Accuracy:  0.848, Loss:  0.110
    Epoch   8 Batch   25/269 - Train Accuracy:  0.846, Validation Accuracy:  0.847, Loss:  0.119
    Epoch   8 Batch   26/269 - Train Accuracy:  0.854, Validation Accuracy:  0.850, Loss:  0.099
    Epoch   8 Batch   27/269 - Train Accuracy:  0.845, Validation Accuracy:  0.844, Loss:  0.107
    Epoch   8 Batch   28/269 - Train Accuracy:  0.832, Validation Accuracy:  0.846, Loss:  0.115
    Epoch   8 Batch   29/269 - Train Accuracy:  0.866, Validation Accuracy:  0.849, Loss:  0.115
    Epoch   8 Batch   30/269 - Train Accuracy:  0.854, Validation Accuracy:  0.851, Loss:  0.105
    Epoch   8 Batch   31/269 - Train Accuracy:  0.856, Validation Accuracy:  0.846, Loss:  0.099
    Epoch   8 Batch   32/269 - Train Accuracy:  0.849, Validation Accuracy:  0.838, Loss:  0.106
    Epoch   8 Batch   33/269 - Train Accuracy:  0.860, Validation Accuracy:  0.845, Loss:  0.101
    Epoch   8 Batch   34/269 - Train Accuracy:  0.853, Validation Accuracy:  0.849, Loss:  0.102
    Epoch   8 Batch   35/269 - Train Accuracy:  0.847, Validation Accuracy:  0.845, Loss:  0.127
    Epoch   8 Batch   36/269 - Train Accuracy:  0.830, Validation Accuracy:  0.847, Loss:  0.111
    Epoch   8 Batch   37/269 - Train Accuracy:  0.867, Validation Accuracy:  0.840, Loss:  0.106
    Epoch   8 Batch   38/269 - Train Accuracy:  0.847, Validation Accuracy:  0.841, Loss:  0.105
    Epoch   8 Batch   39/269 - Train Accuracy:  0.860, Validation Accuracy:  0.844, Loss:  0.110
    Epoch   8 Batch   40/269 - Train Accuracy:  0.855, Validation Accuracy:  0.842, Loss:  0.112
    Epoch   8 Batch   41/269 - Train Accuracy:  0.840, Validation Accuracy:  0.843, Loss:  0.107
    Epoch   8 Batch   42/269 - Train Accuracy:  0.862, Validation Accuracy:  0.844, Loss:  0.100
    Epoch   8 Batch   43/269 - Train Accuracy:  0.846, Validation Accuracy:  0.847, Loss:  0.116
    Epoch   8 Batch   44/269 - Train Accuracy:  0.856, Validation Accuracy:  0.851, Loss:  0.117
    Epoch   8 Batch   45/269 - Train Accuracy:  0.847, Validation Accuracy:  0.847, Loss:  0.113
    Epoch   8 Batch   46/269 - Train Accuracy:  0.857, Validation Accuracy:  0.843, Loss:  0.108
    Epoch   8 Batch   47/269 - Train Accuracy:  0.859, Validation Accuracy:  0.854, Loss:  0.101
    Epoch   8 Batch   48/269 - Train Accuracy:  0.852, Validation Accuracy:  0.844, Loss:  0.101
    Epoch   8 Batch   49/269 - Train Accuracy:  0.846, Validation Accuracy:  0.842, Loss:  0.106
    Epoch   8 Batch   50/269 - Train Accuracy:  0.840, Validation Accuracy:  0.839, Loss:  0.118
    Epoch   8 Batch   51/269 - Train Accuracy:  0.847, Validation Accuracy:  0.847, Loss:  0.112
    Epoch   8 Batch   52/269 - Train Accuracy:  0.846, Validation Accuracy:  0.840, Loss:  0.095
    Epoch   8 Batch   53/269 - Train Accuracy:  0.832, Validation Accuracy:  0.839, Loss:  0.115
    Epoch   8 Batch   54/269 - Train Accuracy:  0.867, Validation Accuracy:  0.843, Loss:  0.103
    Epoch   8 Batch   55/269 - Train Accuracy:  0.855, Validation Accuracy:  0.843, Loss:  0.107
    Epoch   8 Batch   56/269 - Train Accuracy:  0.848, Validation Accuracy:  0.846, Loss:  0.112
    Epoch   8 Batch   57/269 - Train Accuracy:  0.852, Validation Accuracy:  0.842, Loss:  0.113
    Epoch   8 Batch   58/269 - Train Accuracy:  0.835, Validation Accuracy:  0.836, Loss:  0.105
    Epoch   8 Batch   59/269 - Train Accuracy:  0.868, Validation Accuracy:  0.847, Loss:  0.098
    Epoch   8 Batch   60/269 - Train Accuracy:  0.845, Validation Accuracy:  0.844, Loss:  0.102
    Epoch   8 Batch   61/269 - Train Accuracy:  0.860, Validation Accuracy:  0.847, Loss:  0.099
    Epoch   8 Batch   62/269 - Train Accuracy:  0.848, Validation Accuracy:  0.842, Loss:  0.109
    Epoch   8 Batch   63/269 - Train Accuracy:  0.861, Validation Accuracy:  0.847, Loss:  0.114
    Epoch   8 Batch   64/269 - Train Accuracy:  0.847, Validation Accuracy:  0.846, Loss:  0.102
    Epoch   8 Batch   65/269 - Train Accuracy:  0.852, Validation Accuracy:  0.843, Loss:  0.105
    Epoch   8 Batch   66/269 - Train Accuracy:  0.832, Validation Accuracy:  0.841, Loss:  0.106
    Epoch   8 Batch   67/269 - Train Accuracy:  0.843, Validation Accuracy:  0.843, Loss:  0.115
    Epoch   8 Batch   68/269 - Train Accuracy:  0.844, Validation Accuracy:  0.837, Loss:  0.110
    Epoch   8 Batch   69/269 - Train Accuracy:  0.824, Validation Accuracy:  0.840, Loss:  0.123
    Epoch   8 Batch   70/269 - Train Accuracy:  0.858, Validation Accuracy:  0.835, Loss:  0.105
    Epoch   8 Batch   71/269 - Train Accuracy:  0.854, Validation Accuracy:  0.850, Loss:  0.120
    Epoch   8 Batch   72/269 - Train Accuracy:  0.847, Validation Accuracy:  0.845, Loss:  0.115
    Epoch   8 Batch   73/269 - Train Accuracy:  0.854, Validation Accuracy:  0.844, Loss:  0.115
    Epoch   8 Batch   74/269 - Train Accuracy:  0.849, Validation Accuracy:  0.847, Loss:  0.108
    Epoch   8 Batch   75/269 - Train Accuracy:  0.846, Validation Accuracy:  0.849, Loss:  0.119
    Epoch   8 Batch   76/269 - Train Accuracy:  0.857, Validation Accuracy:  0.850, Loss:  0.102
    Epoch   8 Batch   77/269 - Train Accuracy:  0.863, Validation Accuracy:  0.847, Loss:  0.106
    Epoch   8 Batch   78/269 - Train Accuracy:  0.856, Validation Accuracy:  0.848, Loss:  0.106
    Epoch   8 Batch   79/269 - Train Accuracy:  0.852, Validation Accuracy:  0.847, Loss:  0.106
    Epoch   8 Batch   80/269 - Train Accuracy:  0.862, Validation Accuracy:  0.850, Loss:  0.114
    Epoch   8 Batch   81/269 - Train Accuracy:  0.837, Validation Accuracy:  0.846, Loss:  0.120
    Epoch   8 Batch   82/269 - Train Accuracy:  0.871, Validation Accuracy:  0.840, Loss:  0.108
    Epoch   8 Batch   83/269 - Train Accuracy:  0.852, Validation Accuracy:  0.837, Loss:  0.128
    Epoch   8 Batch   84/269 - Train Accuracy:  0.855, Validation Accuracy:  0.848, Loss:  0.119
    Epoch   8 Batch   85/269 - Train Accuracy:  0.855, Validation Accuracy:  0.849, Loss:  0.114
    Epoch   8 Batch   86/269 - Train Accuracy:  0.848, Validation Accuracy:  0.843, Loss:  0.109
    Epoch   8 Batch   87/269 - Train Accuracy:  0.838, Validation Accuracy:  0.832, Loss:  0.123
    Epoch   8 Batch   88/269 - Train Accuracy:  0.857, Validation Accuracy:  0.837, Loss:  0.114
    Epoch   8 Batch   89/269 - Train Accuracy:  0.859, Validation Accuracy:  0.842, Loss:  0.107
    Epoch   8 Batch   90/269 - Train Accuracy:  0.842, Validation Accuracy:  0.849, Loss:  0.121
    Epoch   8 Batch   91/269 - Train Accuracy:  0.845, Validation Accuracy:  0.852, Loss:  0.113
    Epoch   8 Batch   92/269 - Train Accuracy:  0.857, Validation Accuracy:  0.847, Loss:  0.101
    Epoch   8 Batch   93/269 - Train Accuracy:  0.850, Validation Accuracy:  0.842, Loss:  0.120
    Epoch   8 Batch   94/269 - Train Accuracy:  0.852, Validation Accuracy:  0.841, Loss:  0.130
    Epoch   8 Batch   95/269 - Train Accuracy:  0.861, Validation Accuracy:  0.845, Loss:  0.116
    Epoch   8 Batch   96/269 - Train Accuracy:  0.820, Validation Accuracy:  0.838, Loss:  0.125
    Epoch   8 Batch   97/269 - Train Accuracy:  0.853, Validation Accuracy:  0.836, Loss:  0.111
    Epoch   8 Batch   98/269 - Train Accuracy:  0.859, Validation Accuracy:  0.845, Loss:  0.115
    Epoch   8 Batch   99/269 - Train Accuracy:  0.838, Validation Accuracy:  0.846, Loss:  0.113
    Epoch   8 Batch  100/269 - Train Accuracy:  0.875, Validation Accuracy:  0.844, Loss:  0.115
    Epoch   8 Batch  101/269 - Train Accuracy:  0.829, Validation Accuracy:  0.843, Loss:  0.113
    Epoch   8 Batch  102/269 - Train Accuracy:  0.852, Validation Accuracy:  0.848, Loss:  0.105
    Epoch   8 Batch  103/269 - Train Accuracy:  0.844, Validation Accuracy:  0.848, Loss:  0.113
    Epoch   8 Batch  104/269 - Train Accuracy:  0.845, Validation Accuracy:  0.844, Loss:  0.106
    Epoch   8 Batch  105/269 - Train Accuracy:  0.855, Validation Accuracy:  0.850, Loss:  0.109
    Epoch   8 Batch  106/269 - Train Accuracy:  0.841, Validation Accuracy:  0.850, Loss:  0.095
    Epoch   8 Batch  107/269 - Train Accuracy:  0.855, Validation Accuracy:  0.851, Loss:  0.106
    Epoch   8 Batch  108/269 - Train Accuracy:  0.861, Validation Accuracy:  0.849, Loss:  0.100
    Epoch   8 Batch  109/269 - Train Accuracy:  0.825, Validation Accuracy:  0.848, Loss:  0.111
    Epoch   8 Batch  110/269 - Train Accuracy:  0.839, Validation Accuracy:  0.834, Loss:  0.093
    Epoch   8 Batch  111/269 - Train Accuracy:  0.828, Validation Accuracy:  0.838, Loss:  0.113
    Epoch   8 Batch  112/269 - Train Accuracy:  0.844, Validation Accuracy:  0.836, Loss:  0.106
    Epoch   8 Batch  113/269 - Train Accuracy:  0.857, Validation Accuracy:  0.837, Loss:  0.102
    Epoch   8 Batch  114/269 - Train Accuracy:  0.842, Validation Accuracy:  0.841, Loss:  0.103
    Epoch   8 Batch  115/269 - Train Accuracy:  0.858, Validation Accuracy:  0.841, Loss:  0.109
    Epoch   8 Batch  116/269 - Train Accuracy:  0.871, Validation Accuracy:  0.850, Loss:  0.100
    Epoch   8 Batch  117/269 - Train Accuracy:  0.842, Validation Accuracy:  0.851, Loss:  0.099
    Epoch   8 Batch  118/269 - Train Accuracy:  0.876, Validation Accuracy:  0.847, Loss:  0.090
    Epoch   8 Batch  119/269 - Train Accuracy:  0.861, Validation Accuracy:  0.854, Loss:  0.112
    Epoch   8 Batch  120/269 - Train Accuracy:  0.856, Validation Accuracy:  0.851, Loss:  0.105
    Epoch   8 Batch  121/269 - Train Accuracy:  0.865, Validation Accuracy:  0.850, Loss:  0.096
    Epoch   8 Batch  122/269 - Train Accuracy:  0.867, Validation Accuracy:  0.850, Loss:  0.098
    Epoch   8 Batch  123/269 - Train Accuracy:  0.846, Validation Accuracy:  0.854, Loss:  0.099
    Epoch   8 Batch  124/269 - Train Accuracy:  0.849, Validation Accuracy:  0.851, Loss:  0.092
    Epoch   8 Batch  125/269 - Train Accuracy:  0.864, Validation Accuracy:  0.844, Loss:  0.094
    Epoch   8 Batch  126/269 - Train Accuracy:  0.837, Validation Accuracy:  0.842, Loss:  0.100
    Epoch   8 Batch  127/269 - Train Accuracy:  0.856, Validation Accuracy:  0.835, Loss:  0.104
    Epoch   8 Batch  128/269 - Train Accuracy:  0.858, Validation Accuracy:  0.835, Loss:  0.102
    Epoch   8 Batch  129/269 - Train Accuracy:  0.840, Validation Accuracy:  0.837, Loss:  0.104
    Epoch   8 Batch  130/269 - Train Accuracy:  0.845, Validation Accuracy:  0.839, Loss:  0.108
    Epoch   8 Batch  131/269 - Train Accuracy:  0.837, Validation Accuracy:  0.846, Loss:  0.112
    Epoch   8 Batch  132/269 - Train Accuracy:  0.835, Validation Accuracy:  0.849, Loss:  0.107
    Epoch   8 Batch  133/269 - Train Accuracy:  0.864, Validation Accuracy:  0.845, Loss:  0.093
    Epoch   8 Batch  134/269 - Train Accuracy:  0.859, Validation Accuracy:  0.842, Loss:  0.096
    Epoch   8 Batch  135/269 - Train Accuracy:  0.854, Validation Accuracy:  0.844, Loss:  0.105
    Epoch   8 Batch  136/269 - Train Accuracy:  0.848, Validation Accuracy:  0.841, Loss:  0.108
    Epoch   8 Batch  137/269 - Train Accuracy:  0.856, Validation Accuracy:  0.842, Loss:  0.110
    Epoch   8 Batch  138/269 - Train Accuracy:  0.851, Validation Accuracy:  0.843, Loss:  0.097
    Epoch   8 Batch  139/269 - Train Accuracy:  0.861, Validation Accuracy:  0.853, Loss:  0.098
    Epoch   8 Batch  140/269 - Train Accuracy:  0.867, Validation Accuracy:  0.841, Loss:  0.106
    Epoch   8 Batch  141/269 - Train Accuracy:  0.863, Validation Accuracy:  0.854, Loss:  0.110
    Epoch   8 Batch  142/269 - Train Accuracy:  0.823, Validation Accuracy:  0.817, Loss:  0.103
    Epoch   8 Batch  143/269 - Train Accuracy:  0.870, Validation Accuracy:  0.839, Loss:  0.129
    Epoch   8 Batch  144/269 - Train Accuracy:  0.854, Validation Accuracy:  0.812, Loss:  0.097
    Epoch   8 Batch  145/269 - Train Accuracy:  0.852, Validation Accuracy:  0.845, Loss:  0.120
    Epoch   8 Batch  146/269 - Train Accuracy:  0.840, Validation Accuracy:  0.849, Loss:  0.113
    Epoch   8 Batch  147/269 - Train Accuracy:  0.847, Validation Accuracy:  0.844, Loss:  0.111
    Epoch   8 Batch  148/269 - Train Accuracy:  0.858, Validation Accuracy:  0.849, Loss:  0.126
    Epoch   8 Batch  149/269 - Train Accuracy:  0.798, Validation Accuracy:  0.809, Loss:  0.116
    Epoch   8 Batch  150/269 - Train Accuracy:  0.849, Validation Accuracy:  0.829, Loss:  0.159
    Epoch   8 Batch  151/269 - Train Accuracy:  0.845, Validation Accuracy:  0.831, Loss:  0.123
    Epoch   8 Batch  152/269 - Train Accuracy:  0.823, Validation Accuracy:  0.818, Loss:  0.181
    Epoch   8 Batch  153/269 - Train Accuracy:  0.832, Validation Accuracy:  0.814, Loss:  0.137
    Epoch   8 Batch  154/269 - Train Accuracy:  0.837, Validation Accuracy:  0.818, Loss:  0.156
    Epoch   8 Batch  155/269 - Train Accuracy:  0.834, Validation Accuracy:  0.815, Loss:  0.137
    Epoch   8 Batch  156/269 - Train Accuracy:  0.834, Validation Accuracy:  0.832, Loss:  0.167
    Epoch   8 Batch  157/269 - Train Accuracy:  0.839, Validation Accuracy:  0.833, Loss:  0.142
    Epoch   8 Batch  158/269 - Train Accuracy:  0.832, Validation Accuracy:  0.817, Loss:  0.143
    Epoch   8 Batch  159/269 - Train Accuracy:  0.818, Validation Accuracy:  0.820, Loss:  0.147
    Epoch   8 Batch  160/269 - Train Accuracy:  0.794, Validation Accuracy:  0.798, Loss:  0.155
    Epoch   8 Batch  161/269 - Train Accuracy:  0.825, Validation Accuracy:  0.821, Loss:  0.199
    Epoch   8 Batch  162/269 - Train Accuracy:  0.798, Validation Accuracy:  0.794, Loss:  0.161
    Epoch   8 Batch  163/269 - Train Accuracy:  0.819, Validation Accuracy:  0.803, Loss:  0.180
    Epoch   8 Batch  164/269 - Train Accuracy:  0.823, Validation Accuracy:  0.804, Loss:  0.165
    Epoch   8 Batch  165/269 - Train Accuracy:  0.809, Validation Accuracy:  0.783, Loss:  0.165
    Epoch   8 Batch  166/269 - Train Accuracy:  0.836, Validation Accuracy:  0.796, Loss:  0.181
    Epoch   8 Batch  167/269 - Train Accuracy:  0.810, Validation Accuracy:  0.789, Loss:  0.158
    Epoch   8 Batch  168/269 - Train Accuracy:  0.824, Validation Accuracy:  0.809, Loss:  0.201
    Epoch   8 Batch  169/269 - Train Accuracy:  0.802, Validation Accuracy:  0.801, Loss:  0.166
    Epoch   8 Batch  170/269 - Train Accuracy:  0.808, Validation Accuracy:  0.799, Loss:  0.190
    Epoch   8 Batch  171/269 - Train Accuracy:  0.829, Validation Accuracy:  0.811, Loss:  0.172
    Epoch   8 Batch  172/269 - Train Accuracy:  0.829, Validation Accuracy:  0.825, Loss:  0.163
    Epoch   8 Batch  173/269 - Train Accuracy:  0.833, Validation Accuracy:  0.821, Loss:  0.142
    Epoch   8 Batch  174/269 - Train Accuracy:  0.814, Validation Accuracy:  0.819, Loss:  0.139
    Epoch   8 Batch  175/269 - Train Accuracy:  0.811, Validation Accuracy:  0.812, Loss:  0.159
    Epoch   8 Batch  176/269 - Train Accuracy:  0.829, Validation Accuracy:  0.819, Loss:  0.145
    Epoch   8 Batch  177/269 - Train Accuracy:  0.839, Validation Accuracy:  0.820, Loss:  0.126
    Epoch   8 Batch  178/269 - Train Accuracy:  0.835, Validation Accuracy:  0.820, Loss:  0.143
    Epoch   8 Batch  179/269 - Train Accuracy:  0.842, Validation Accuracy:  0.831, Loss:  0.134
    Epoch   8 Batch  180/269 - Train Accuracy:  0.841, Validation Accuracy:  0.833, Loss:  0.118
    Epoch   8 Batch  181/269 - Train Accuracy:  0.820, Validation Accuracy:  0.836, Loss:  0.133
    Epoch   8 Batch  182/269 - Train Accuracy:  0.848, Validation Accuracy:  0.826, Loss:  0.119
    Epoch   8 Batch  183/269 - Train Accuracy:  0.872, Validation Accuracy:  0.827, Loss:  0.108
    Epoch   8 Batch  184/269 - Train Accuracy:  0.848, Validation Accuracy:  0.825, Loss:  0.115
    Epoch   8 Batch  185/269 - Train Accuracy:  0.852, Validation Accuracy:  0.829, Loss:  0.116
    Epoch   8 Batch  186/269 - Train Accuracy:  0.833, Validation Accuracy:  0.835, Loss:  0.111
    Epoch   8 Batch  187/269 - Train Accuracy:  0.838, Validation Accuracy:  0.837, Loss:  0.110
    Epoch   8 Batch  188/269 - Train Accuracy:  0.863, Validation Accuracy:  0.846, Loss:  0.119
    Epoch   8 Batch  189/269 - Train Accuracy:  0.837, Validation Accuracy:  0.847, Loss:  0.106
    Epoch   8 Batch  190/269 - Train Accuracy:  0.848, Validation Accuracy:  0.846, Loss:  0.114
    Epoch   8 Batch  191/269 - Train Accuracy:  0.858, Validation Accuracy:  0.845, Loss:  0.114
    Epoch   8 Batch  192/269 - Train Accuracy:  0.856, Validation Accuracy:  0.845, Loss:  0.106
    Epoch   8 Batch  193/269 - Train Accuracy:  0.854, Validation Accuracy:  0.844, Loss:  0.104
    Epoch   8 Batch  194/269 - Train Accuracy:  0.852, Validation Accuracy:  0.851, Loss:  0.112
    Epoch   8 Batch  195/269 - Train Accuracy:  0.851, Validation Accuracy:  0.846, Loss:  0.105
    Epoch   8 Batch  196/269 - Train Accuracy:  0.835, Validation Accuracy:  0.845, Loss:  0.107
    Epoch   8 Batch  197/269 - Train Accuracy:  0.844, Validation Accuracy:  0.848, Loss:  0.116
    Epoch   8 Batch  198/269 - Train Accuracy:  0.837, Validation Accuracy:  0.846, Loss:  0.106
    Epoch   8 Batch  199/269 - Train Accuracy:  0.850, Validation Accuracy:  0.837, Loss:  0.112
    Epoch   8 Batch  200/269 - Train Accuracy:  0.865, Validation Accuracy:  0.838, Loss:  0.106
    Epoch   8 Batch  201/269 - Train Accuracy:  0.850, Validation Accuracy:  0.842, Loss:  0.106
    Epoch   8 Batch  202/269 - Train Accuracy:  0.848, Validation Accuracy:  0.845, Loss:  0.111
    Epoch   8 Batch  203/269 - Train Accuracy:  0.870, Validation Accuracy:  0.842, Loss:  0.105
    Epoch   8 Batch  204/269 - Train Accuracy:  0.851, Validation Accuracy:  0.845, Loss:  0.109
    Epoch   8 Batch  205/269 - Train Accuracy:  0.859, Validation Accuracy:  0.837, Loss:  0.099
    Epoch   8 Batch  206/269 - Train Accuracy:  0.853, Validation Accuracy:  0.842, Loss:  0.113
    Epoch   8 Batch  207/269 - Train Accuracy:  0.863, Validation Accuracy:  0.843, Loss:  0.096
    Epoch   8 Batch  208/269 - Train Accuracy:  0.850, Validation Accuracy:  0.843, Loss:  0.103
    Epoch   8 Batch  209/269 - Train Accuracy:  0.863, Validation Accuracy:  0.844, Loss:  0.096
    Epoch   8 Batch  210/269 - Train Accuracy:  0.859, Validation Accuracy:  0.847, Loss:  0.094
    Epoch   8 Batch  211/269 - Train Accuracy:  0.866, Validation Accuracy:  0.852, Loss:  0.101
    Epoch   8 Batch  212/269 - Train Accuracy:  0.863, Validation Accuracy:  0.855, Loss:  0.106
    Epoch   8 Batch  213/269 - Train Accuracy:  0.852, Validation Accuracy:  0.852, Loss:  0.099
    Epoch   8 Batch  214/269 - Train Accuracy:  0.859, Validation Accuracy:  0.852, Loss:  0.102
    Epoch   8 Batch  215/269 - Train Accuracy:  0.872, Validation Accuracy:  0.858, Loss:  0.094
    Epoch   8 Batch  216/269 - Train Accuracy:  0.845, Validation Accuracy:  0.858, Loss:  0.116
    Epoch   8 Batch  217/269 - Train Accuracy:  0.862, Validation Accuracy:  0.860, Loss:  0.097
    Epoch   8 Batch  218/269 - Train Accuracy:  0.861, Validation Accuracy:  0.856, Loss:  0.094
    Epoch   8 Batch  219/269 - Train Accuracy:  0.865, Validation Accuracy:  0.854, Loss:  0.097
    Epoch   8 Batch  220/269 - Train Accuracy:  0.860, Validation Accuracy:  0.854, Loss:  0.087
    Epoch   8 Batch  221/269 - Train Accuracy:  0.865, Validation Accuracy:  0.853, Loss:  0.093
    Epoch   8 Batch  222/269 - Train Accuracy:  0.878, Validation Accuracy:  0.853, Loss:  0.086
    Epoch   8 Batch  223/269 - Train Accuracy:  0.858, Validation Accuracy:  0.854, Loss:  0.086
    Epoch   8 Batch  224/269 - Train Accuracy:  0.860, Validation Accuracy:  0.851, Loss:  0.103
    Epoch   8 Batch  225/269 - Train Accuracy:  0.852, Validation Accuracy:  0.857, Loss:  0.086
    Epoch   8 Batch  226/269 - Train Accuracy:  0.858, Validation Accuracy:  0.860, Loss:  0.098
    Epoch   8 Batch  227/269 - Train Accuracy:  0.883, Validation Accuracy:  0.859, Loss:  0.098
    Epoch   8 Batch  228/269 - Train Accuracy:  0.862, Validation Accuracy:  0.858, Loss:  0.089
    Epoch   8 Batch  229/269 - Train Accuracy:  0.860, Validation Accuracy:  0.851, Loss:  0.090
    Epoch   8 Batch  230/269 - Train Accuracy:  0.862, Validation Accuracy:  0.855, Loss:  0.092
    Epoch   8 Batch  231/269 - Train Accuracy:  0.842, Validation Accuracy:  0.854, Loss:  0.095
    Epoch   8 Batch  232/269 - Train Accuracy:  0.856, Validation Accuracy:  0.852, Loss:  0.096
    Epoch   8 Batch  233/269 - Train Accuracy:  0.877, Validation Accuracy:  0.849, Loss:  0.093
    Epoch   8 Batch  234/269 - Train Accuracy:  0.859, Validation Accuracy:  0.853, Loss:  0.088
    Epoch   8 Batch  235/269 - Train Accuracy:  0.882, Validation Accuracy:  0.853, Loss:  0.078
    Epoch   8 Batch  236/269 - Train Accuracy:  0.859, Validation Accuracy:  0.857, Loss:  0.090
    Epoch   8 Batch  237/269 - Train Accuracy:  0.863, Validation Accuracy:  0.858, Loss:  0.090
    Epoch   8 Batch  238/269 - Train Accuracy:  0.855, Validation Accuracy:  0.856, Loss:  0.092
    Epoch   8 Batch  239/269 - Train Accuracy:  0.884, Validation Accuracy:  0.857, Loss:  0.085
    Epoch   8 Batch  240/269 - Train Accuracy:  0.878, Validation Accuracy:  0.860, Loss:  0.083
    Epoch   8 Batch  241/269 - Train Accuracy:  0.868, Validation Accuracy:  0.858, Loss:  0.097
    Epoch   8 Batch  242/269 - Train Accuracy:  0.858, Validation Accuracy:  0.855, Loss:  0.087
    Epoch   8 Batch  243/269 - Train Accuracy:  0.891, Validation Accuracy:  0.857, Loss:  0.083
    Epoch   8 Batch  244/269 - Train Accuracy:  0.880, Validation Accuracy:  0.857, Loss:  0.094
    Epoch   8 Batch  245/269 - Train Accuracy:  0.862, Validation Accuracy:  0.860, Loss:  0.093
    Epoch   8 Batch  246/269 - Train Accuracy:  0.849, Validation Accuracy:  0.855, Loss:  0.095
    Epoch   8 Batch  247/269 - Train Accuracy:  0.867, Validation Accuracy:  0.859, Loss:  0.097
    Epoch   8 Batch  248/269 - Train Accuracy:  0.849, Validation Accuracy:  0.851, Loss:  0.085
    Epoch   8 Batch  249/269 - Train Accuracy:  0.885, Validation Accuracy:  0.855, Loss:  0.094
    Epoch   8 Batch  250/269 - Train Accuracy:  0.862, Validation Accuracy:  0.853, Loss:  0.089
    Epoch   8 Batch  251/269 - Train Accuracy:  0.889, Validation Accuracy:  0.853, Loss:  0.098
    Epoch   8 Batch  252/269 - Train Accuracy:  0.866, Validation Accuracy:  0.851, Loss:  0.078
    Epoch   8 Batch  253/269 - Train Accuracy:  0.862, Validation Accuracy:  0.857, Loss:  0.105
    Epoch   8 Batch  254/269 - Train Accuracy:  0.860, Validation Accuracy:  0.858, Loss:  0.087
    Epoch   8 Batch  255/269 - Train Accuracy:  0.864, Validation Accuracy:  0.861, Loss:  0.089
    Epoch   8 Batch  256/269 - Train Accuracy:  0.847, Validation Accuracy:  0.862, Loss:  0.091
    Epoch   8 Batch  257/269 - Train Accuracy:  0.844, Validation Accuracy:  0.854, Loss:  0.101
    Epoch   8 Batch  258/269 - Train Accuracy:  0.863, Validation Accuracy:  0.851, Loss:  0.097
    Epoch   8 Batch  259/269 - Train Accuracy:  0.876, Validation Accuracy:  0.854, Loss:  0.095
    Epoch   8 Batch  260/269 - Train Accuracy:  0.852, Validation Accuracy:  0.865, Loss:  0.100
    Epoch   8 Batch  261/269 - Train Accuracy:  0.865, Validation Accuracy:  0.866, Loss:  0.089
    Epoch   8 Batch  262/269 - Train Accuracy:  0.873, Validation Accuracy:  0.869, Loss:  0.088
    Epoch   8 Batch  263/269 - Train Accuracy:  0.865, Validation Accuracy:  0.869, Loss:  0.094
    Epoch   8 Batch  264/269 - Train Accuracy:  0.846, Validation Accuracy:  0.862, Loss:  0.098
    Epoch   8 Batch  265/269 - Train Accuracy:  0.867, Validation Accuracy:  0.868, Loss:  0.097
    Epoch   8 Batch  266/269 - Train Accuracy:  0.869, Validation Accuracy:  0.863, Loss:  0.082
    Epoch   8 Batch  267/269 - Train Accuracy:  0.871, Validation Accuracy:  0.865, Loss:  0.095
    Model Trained and Saved


### Save Parameters
Save the `batch_size` and `save_path` parameters for inference.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()
```

## Sentence to Sequence
To feed a sentence into the model for translation, you first need to preprocess it.  Implement the function `sentence_to_seq()` to preprocess new sentences.

- Convert the sentence to lowercase
- Convert words into ids using `vocab_to_int`
 - Convert words not in the vocabulary, to the `<UNK>` word id.


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    # TODO: Implement Function
    return [vocab_to_int.get(word, vocab_to_int['<UNK>'])for word in sentence.lower().split()]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)
```

    Tests Passed


## Translate
This will translate `translate_sentence` from English to French.


```python
translate_sentence = 'he saw a old yellow truck .'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))
```

    Input
      Word Ids:      [119, 126, 97, 181, 185, 188, 87]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']
    
    Prediction
      Word Ids:      [19, 301, 79, 93, 282, 340, 90, 116, 1]
      French Words: ['il', 'pourrait', 'aller', 'aux', 'états-unis', 'en', 'mars', '.', '<EOS>']


## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus after you've submitted this project.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.


```python

```
