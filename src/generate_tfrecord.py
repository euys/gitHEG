import os
import io
import pandas as pd
# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from PIL import Image
from collections import namedtuple

flags = tf.compat.v1.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'bikerider':
        return 1
    else:
        None


def int64_feature(value):
    return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=value))


def bytes_feature(value):
    return tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=value))


def float_list_feature(value):
    return tf.compat.v1.train.Feature(float_list=tf.compat.v1.train.FloatList(value=value))


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x
            in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.compat.v1.gfile.GFile(os.path.join(path, '{}'.format(group.filename)),
                        'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf-8')
    image_format = b'jpg'
    if ".png" in group.filename:
        image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf-8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.compat.v1.train.Example(features=tf.compat.v1.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.compat.v1.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join("C:\\Users\\admin\\Desktop\\bikerider-detector\\data\\training\\image")
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.compat.v1.app.run()
