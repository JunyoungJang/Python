# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/11_Adversarial_Examples.ipynb

# -*- coding: utf-8 -*-


import tensorflow as tf, numpy as np, matplotlib.pyplot as plt, _pickle as pickle
import os, sys, re, tarfile, datetime
from six.moves import urllib
from scipy.misc import imread, imresize


cls_target = 300 # bookcase
noise_limit = 3
required_score = 0.99
max_iterations = 100
save_dir = "./temp/logfile"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'savefile.ckpt')


# load data which we will classify as bookcase #########################################################################
image_path = "/Users/Sungchul/Dropbox/Images/parrot.jpeg"
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

img_true = imread(image_path)
print(img_true.dtype, img_true.shape)
plt.imshow(img_true)
plt.show()
# load data which we will classify as bookcase #########################################################################


# Inception-v3 model download and extract if not exist #################################################################
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz' # Inception-v3 model URL
dest_directory = './tmp/imagenet' # directory where Inception-v3 model is downloaded
if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
filename = DATA_URL.split('/')[-1] # inception-2015-12-05.tgz
filepath = os.path.join(dest_directory, filename) # ./tmp/imagenet/inception-2015-12-05.tgz
if not os.path.exists(filepath): # download Inception-v3 model if not exist
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
tarfile.open(filepath, 'r:gz').extractall(dest_directory) # tarfile extract
# Inception-v3 model download and extract if not exist #################################################################


# Creates graph from saved graph_def.pb. ###############################################################################
# Open the graph-def file for binary reading.
path = os.path.join(dest_directory, 'classify_image_graph_def.pb')
with tf.gfile.FastGFile(path, 'rb') as f:
    # TensorFlow graphs are saved to disk as so-called Protocol Buffers
    # aka. proto-bufs which is a file-format that works on multiple platforms.
    # In this case it is saved as a binary file.
    graph_def = tf.GraphDef() # First we need to create an empty graph-def.
    graph_def.ParseFromString(f.read()) # Then we load the proto-buf file into the graph-def.
    tf.import_graph_def(graph_def, name='') # Finally we import the graph-def to the default TensorFlow graph.
# Creates graph from saved graph_def.pb. ###############################################################################


# 정수 형태의 node ID를 인간이 이해할 수 있는 레이블로 변환 ########################################################################
class NodeLookup(object):
    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                './tmp/imagenet', 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                './tmp/imagenet', 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """각각의 softmax node에 대해 인간이 읽을 수 있는 영어 단어를 로드 함.
        Args:
          label_lookup_path: 정수 node ID에 대한 문자 UID.
          uid_lookup_path: 인간이 읽을 수 있는 문자에 대한 문자 UID.
        Returns:
          정수 node ID로부터 인간이 읽을 수 있는 문자에 대한 dict.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # 문자 UID로부터 인간이 읽을 수 있는 문자로의 맵핑을 로드함.
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # 문자 UID로부터 정수 node ID에 대한 맵핑을 로드함.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # 마지막으로 정수 node ID로부터 인간이 읽을 수 있는 문자로의 맵핑을 로드함.
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]
# 정수 형태의 node ID를 인간이 이해할 수 있는 레이블로 변환 ########################################################################


# plot of original, original+noise, noise ##############################################################################
def normalize_image(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min) # Normalize so all values are between 0.0 and 1.0
    return x_norm

def plot_images(image, noise, noisy_image,
                name_source, name_target,
                score_source, score_source_org, score_target):
    """
    Plot the image, the noisy image and the noise.
    Also shows the class-names and scores.

    Note that the noise is amplified to use the full range of
    colours, otherwise if the noise is very low it would be
    hard to see.

    image: Original input image.
    noise: Noise that has been added to the image.
    noisy_image: Input image + noise.
    name_source: Name of the source-class.
    name_target: Name of the target-class.
    score_source: Score for the source-class.
    score_source_org: Original score for the source-class.
    score_target: Score for the target-class.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 10)) # Create figure with sub-plots.
    fig.subplots_adjust(hspace=0.1, wspace=0.1) # Adjust vertical spacing.

    smooth = True # Use interpolation to smooth pixels?
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # Plot the original image.
    # Note that the pixel-values are normalized to the [0.0, 1.0]
    # range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(image.reshape((299, 299, 3)) / 255.0, interpolation=interpolation)
    msg = "Original Image:\n{0} ({1:.2%})"
    xlabel = msg.format(name_source, score_source_org)
    ax.set_xlabel(xlabel)

    # Plot the noisy image.
    ax = axes.flat[1]
    ax.imshow(image.reshape((299, 299, 3)) / 255.0, interpolation=interpolation)
    msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
    xlabel = msg.format(name_source, score_source, name_target, score_target)
    ax.set_xlabel(xlabel)

    # Plot the noise.
    # The colours are amplified otherwise they would be hard to see.
    ax = axes.flat[2]
    ax.imshow(normalize_image(noise), interpolation=interpolation)
    xlabel = "Amplified Noise"
    ax.set_xlabel(xlabel)

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
# plot of original, original+noise, noise ##############################################################################


# top k prediction #####################################################################################################
with tf.Session() as session:
    tf.global_variables_initializer().run()

    # construction of adversarial network based on Inception-v3 model
    # 몇가지 유용한 텐서들:
    # 'softmax:0' : Scaled last layer tensor for 1000 labels
    # "softmax/logits:0" : Unscaled last layer tensor for 1000 labels
    # 'pool_3:0' : Next to last layer tensor of 2048 units
    # "ResizeBilinear:0" : Tensor for feeding resized decoded input images.
    # "DecodeJpeg:0" : Input layer tensor for feeding decoded input images.
    # 'DecodeJpeg/contents:0' : Input layer tensor for feeding jpeg input images.
    prob_tensor = session.graph.get_tensor_by_name('softmax:0')
    score_tensor = session.graph.get_tensor_by_name("softmax/logits:0")
    pool3_tensor = session.graph.get_tensor_by_name('pool_3:0')
    resized_image_tensor = session.graph.get_tensor_by_name("ResizeBilinear:0")
    decoded_image_tensor = session.graph.get_tensor_by_name("DecodeJpeg:0")
    jpeg_image_tensor = session.graph.get_tensor_by_name('DecodeJpeg/contents:0')

    # placeholder for target class-number
    pl_cls_target = tf.placeholder(dtype=tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score_tensor, labels=[pl_cls_target])

    # Get the gradient for the loss-function with regard to the resized input image.
    gradient = tf.gradients(loss, resized_image_tensor)

    # data preprocessing - put original image into Inception-v3 and save output of ResizeBilinear:0 node of Inception-v3
    # We will add a noise on this resized 299 x 299 x 3 image to classify as a bookcase
    feed_dict = {jpeg_image_tensor: image_data}
    pred, image = session.run([prob_tensor, resized_image_tensor], feed_dict=feed_dict)
    pred = np.squeeze(pred) # Convert to one-dimensional array.
    cls_source = np.argmax(pred) # Predicted class-number.
    score_source_org = pred.max() # Score for the predicted class (aka. probability or confidence).

    # check the class name of original image and target class name
    node_lookup = NodeLookup() # node ID --> 영어 단어 lookup을 생성한다.
    name_source = node_lookup.id_to_string(cls_source) # Names for the source and target classes.
    name_target = node_lookup.id_to_string(cls_target) # Names for the source and target classes.
    print(name_source, name_target)

    # training to construct an adversarial noise
    noise = 0
    for i in range(max_iterations):
        print("Iteration:", i)
        noisy_image = image + noise # The noisy image is just the sum of the input image and noise.
        noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=255.0)

        feed_dict = {resized_image_tensor: noisy_image, pl_cls_target: cls_target}
        pred, grad = session.run([prob_tensor, gradient], feed_dict=feed_dict)
        pred = np.squeeze(pred)

        score_source = pred[cls_source]
        score_target = pred[cls_target]

        grad = np.array(grad).squeeze()
        grad_absmax = np.abs(grad).max()
        if grad_absmax < 1e-10:
            grad_absmax = 1e-10
        step_size = 7 / grad_absmax

        msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_source, cls_source, name_source))
        msg = "Target score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_target, cls_target, name_target))
        msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
        print(msg.format(grad.min(), grad.max(), step_size))
        print()

        # If the score for the target-class is not high enough.
        if score_target < required_score:
            noise -= step_size * grad
            noise = np.clip(a=noise, a_min=-noise_limit, a_max=noise_limit)
        else:
            break

    # Plot the image and the noise.
    plot_images(image=image, noise=noise, noisy_image=noisy_image,
                name_source=name_source, name_target=name_target,
                score_source=score_source,
                score_source_org=score_source_org,
                score_target=score_target)

    # Print some statistics for the noise.
    msg = "Noise min: {0:.3f}, max: {1:.3f}, mean: {2:.3f}, std: {3:.3f}"
    print(msg.format(noise.min(), noise.max(), noise.mean(), noise.std()))