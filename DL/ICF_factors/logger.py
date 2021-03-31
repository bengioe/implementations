# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514.
#import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.contrib.tensorboard.plugins import projector
import os
import seaborn.apionly as sns
#import seaborn as sns
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
#norm = mpl.colors.Normalize(vmin=0, vmax=1)
cmap = plt.get_cmap('viridis')



class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self.log_dir = log_dir

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[0])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def heatmap_summary(self, tag, mat, step):
        # Receives N x K images tab
        # ipdb.set_trace()
        # mat = norm(mat)
        rgba_img = cmap(mat)
        rgb_img = np.delete(rgba_img, 3, 2)
        images = [rgb_img]

        """Log a heatmap from a 2D matrix."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[0])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def embedding_summary(self, tag, embedding, labels, step):
        self.write_metadata(labels)
        config = tf.ConfigProto(
                        device_count = {'GPU': 0}
                            )
        sess = tf.InteractiveSession(config=config)

        # Create randomly initialized embedding weights which will be trained.
        with tf.device("/cpu:0"):
            embedding_var = tf.Variable(embedding, name=tag)
            tf.global_variables_initializer().run()

            saver = tf.train.Saver()

            # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
            config = projector.ProjectorConfig()

            # You can add multiple embeddings. Here we add only one.
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = self.log_dir+'/metadata.tsv'
            # Link this tensor to its metadata file (e.g. labels).
            #embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

            # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(self.writer, config)
            saver.save(sess, self.log_dir+'/model.ckpt', step)

    def write_metadata(self, labels):
        metadata_path = self.log_dir+'/metadata.tsv'
        f = open(metadata_path, 'w+')
        for lab in labels:
            print(lab, file=f)


    def heat_plot(self, id_name, i_episode, mat):
        fig = plt.figure(1)
        #plt.title(r'$sel(\phi, h, a)$')
        sns.heatmap(mat, xticklabels=False, yticklabels=False, cmap='viridis')
        fig.savefig(f'{self.log_dir}/im/{i_episode}_{id_name}.png', bbox_inches='tight')   # save the figure to file
        plt.close(fig)
        im = plt.imread(f'{self.log_dir}/im/{i_episode}_{id_name}.png')
        np.save(f'{self.log_dir}/im/{i_episode}_{id_name}.npy', mat)
        self.image_summary(f'{id_name}', [im], i_episode)
        plt.close('close')

