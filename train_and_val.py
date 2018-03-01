import os
import numpy as np
import tensorflow as tf
import input_data
import model

# you need to change the directories to yours.
train_dir = 'C:/train/train/'
test_dir = 'C:/catdog/test/'
train_logs_dir = 'C:/catdog/logs/train/'
val_logs_dir = 'C:/catdog/logs/val'

N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
RATIO = 0.2  # take 20% of dataset as validation data
BATCH_SIZE = 50
CAPACITY = 2000
MAX_STEP = 10000  # with current parameters, it is suggested to use MAX_STEP>10k
Train_num = 20000
learning_rate = 1e-4  # with current parameters, it is suggested to use learning rate<0.0001


def training():
    train, train_label, val, val_label = input_data.get_files(train_dir, RATIO)
    train_batch, train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    val_batch, val_label_batch = input_data.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)  # 神经网络最后一层的输出
    loss = model.losses(logits, train_label_batch)
    train_op = model.trainning(loss, learning_rate)
    acc = model.evaluation(logits, train_label_batch)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 1])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE])

    with tf.Session() as sess:
        saver = tf.train.Saver()  # tf.train.Saver类实现神经网络模型的保存和提取
        sess.run(tf.global_variables_initializer())

        # 开始输入队列的线程
        coord = tf.train.Coordinator()  # 创建多线程的实例

        '''
        运行任何训练步骤之前，需要调用tf.train.start_queue_runners函数，否则数据流图将一直挂起。
        tf.train.start_queue_runners 函数将会启动输入管道的线程，填充样本到队列中，以便出队操作可以从队列中拿到样本。
        '''
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_logs_dir, sess.graph)
        val_writer = tf.summary.FileWriter(val_logs_dir, sess.graph)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():  # 如果线程应该停止 则返回True
                    break
                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc], feed_dict={x: tra_images, y_: tra_labels})
                if step % 50 == 0:  # 每训练50个batch样本后输出train_loss\train_acc   50*64 = 3200 个样本
                    print('Step %d, train loss = %.3f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

                if step % 400 == 0 or (step + 1) == MAX_STEP:  # 每训练200个batch样本后输出val_loss\val_acc
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc],
                                                 feed_dict={x: val_images, y_: val_labels})
                    print('**  Step %d, val loss = %.3f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_logs_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)  # tf.train.Saver类实现神经网络模型的保存和提取

        except tf.errors.OutOfRangeError:  # 捕捉和处理由队列产生的异常，包括OutOfRangeError 异常，这个异常用于报告队列被关闭
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()  # 请求该线程停止
        coord.join(threads)  # 等待被指定的线程中止


# Test one image
'''

def get_one_image(file_dir):
    """
    Randomly pick one image from test data
    Return: ndarray
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    test = []
    for file in os.listdir(file_dir):
        test.append(file_dir + file)
    print('There are %d test pictures\n' % (len(test)))

    n = len(test)
    ind = np.random.randint(0, n)
    print(ind)
    img_test = test[ind]

    image = Image.open(img_test)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def test_one_image():
    """
    Test one image with the saved models and parameters
    """

    test_image = get_one_image(test_dir)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(test_image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 1])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

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

            prediction = sess.run(logit, feed_dict={x: test_image})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a cat with possibility %.6f' % prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' % prediction[:, 1])
'''

if __name__ == '__main__':
    training()
    # test_one_image()
