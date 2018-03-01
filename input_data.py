import tensorflow as tf
import numpy as np
import os,cv2
import math

# you need to change this to your data directory
train_dir = 'C:/train/train/'


def get_files(file_dir, ratio):
    """
    Args:
        file_dir: file directory
        ratio:ratio of validation datasets
    Returns:
        list of images and labels  (images的图片地址和label)
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')  # 判断文件名是否==‘指定字符串’  cat.0.jpg

        # 相当于one-hot操作
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)  # cats label设为0
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)  # dogs label设为1
    print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)  # 所有样本数
    n_val = math.ceil(n_sample * ratio)  # 验证样本数 ceil() 函数返回数字的上入整数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """

    image = tf.cast(image, tf.string)  # image的数据格式转化成string
    label = tf.cast(label, tf.int32)

    # 构造输入队列
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])    # 读地址 得到图片文件内容
    image = tf.image.decode_jpeg(image_contents, channels=1)  # 解码为jpg格式
    # image = tf.image.rgb_to_grayscale(image)  # 转化为灰度图

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=50,
                                              capacity=capacity)  # capacity是队列中的容量（长度）
    # you can also use shuffle_batch
    #    image_batch, label_batch = tf.train.shuffle_batch([image,label],
    #                                                      batch_size=BATCH_SIZE,
    #                                                      num_threads=64,
    #                                                      capacity=CAPACITY,
    #                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


if __name__ == '__main__':
    train, train_label, val, val_label = get_files(train_dir, 0.2)
    train_batch, train_label_batch = get_batch(train, train_label, 208, 208, 50, 2000)
    print(train.shape, train_label_batch.shape)

