
import tensorflow as tf
import re
BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
numClasses = 2
lstmUnits = 100
learning_rate = 0.0001

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am',  'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in ', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    # punctuation without single quote
    rid_of_chars = re.compile("[^A-Za-z0-9 ]+")
    review = re.sub(rid_of_chars, " ", review.lower())
    #print([i for i in review.split(" ") if i != "" and i not in stop_words])
    return ([i for i in review.split(" ") if i != "" and i not in stop_words])
    #return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    #tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [BATCH_SIZE, numClasses],name = "labels" )
    input_data = tf.placeholder(tf.float32, [None,MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE],name = "input_data")
    dropout_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')
    #accuracy = tf.placeholder(tf.float32, name='accuracy')
    #data = tf.Variable(tf.zeros([BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE]), dtype=tf.float32)
    #data = tf.nn.embedding_lookup(data, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.6)
    value, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses],stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = tf.nn.relu((tf.matmul(last, weight) + bias))

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    Accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32),name = "accuracy" )

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    #Input placeholder: name = "input_data"
    #labels placeholder: name = "labels"
    #accuracy tensor: name = "accuracy"
    #loss tensor: name = "loss"
    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
