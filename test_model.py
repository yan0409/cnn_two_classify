# Test one image

import tensorflow as tf
import numpy as np
import model,os

test_dir = 'C:/catdog/test/'  # 测试数据路径
train_logs_dir = 'C:/catdog/logs/train/'  # 模型保存路径

def get_one_image(file_dir):
    """
    Randomly pick one image from test data
    Return: ndarray
    """
    test = []
    for file in os.listdir(file_dir):
        test.append(file_dir + file)
    print('There are %d test pictures\n' % (len(test)))

    n = len(test)
    ind = np.random.randint(0, n)  # 生成范围内的一个随机数
    img_test = test[ind]
    print(img_test)
    image_contents = tf.read_file(img_test)
    image = tf.image.decode_jpeg(image_contents, channels=1)  # 解码为jpg格式
    image = tf.image.resize_image_with_crop_or_pad(image, 208, 208)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, 208, 208, 1])  # 输入图片reshape成[1, 208, 208, 1]形式。第一个1 代表1张图片、第二个1代表输入通道（灰度图）
    return image


def test_one_image():
    """
    Test one image with the saved models and parameters
    """

    with tf.Graph().as_default():
        test_image = get_one_image(test_dir)

        BATCH_SIZE = 1
        N_CLASSES = 2
        logit = model.inference(test_image, BATCH_SIZE, N_CLASSES)  # model输出的结果
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[1, 208, 208, 1])
        saver = tf.train.Saver()
        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(train_logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            test_image = sess.run(test_image)
            prediction = sess.run(logit, feed_dict={x: test_image})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a cat with possibility %.6f' % prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' % prediction[:, 1])


if __name__ == '__main__':
    test_one_image()
