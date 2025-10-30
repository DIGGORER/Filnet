import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam
from IsoNet.models.unet.data_sequence import prepare_dataseq
from IsoNet.models.unet.model import Unet,other_Unet
import numpy as np
from tensorflow.keras.models import load_model


def train3D_continue(outFile,
                    model_file,
                    data_dir = 'data',
                    result_folder='results',
                    epochs=40,
                    lr=0.0004,
                    steps_per_epoch=128,
                    batch_size=64,
                    n_gpus=2):
    
    # logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
    # logging.debug('The tf message level {}'.format(os.environ['TF_CPP_MIN_LOG_LEVEL']))

    # metrics = ('mse', 'mae')
    # _metrics = [eval('loss_%s()' % m) for m in metrics]
    # optimizer = Adam(lr=lr)
    
    # model = load_model( model_file) # weight is a model

    #这段代码使用了 TensorFlow 中的 MirroredStrategy 分布式策略。这个策略通常用于在多个 GPU 上进行训练。在使用 MirroredStrategy 时，TensorFlow 会在每个 GPU 上创建一个模型副本，每个副本都包含完整的模型。然后，每个副本都会处理输入数据的一个子集，并计算梯度。最后，所有梯度都会汇总并平均，然后应用于每个模型副本的参数更新。
    #这种方式可以显著提高训练速度，特别是当你有多个 GPU 可用时。通过使用 MirroredStrategy，你可以方便地利用多个 GPU 的并行计算能力，加速模型的训练过程。
    strategy = tf.distribute.MirroredStrategy()
    # train_data = strategy.experimental_distribute_dataset(train_data)
    # test_data = strategy.experimental_distribute_dataset(test_data)
    if n_gpus > 1:
        with strategy.scope():
            model = load_model( model_file) #获取前面模型训练的参数
            optimizer = Adam(learning_rate = lr)
            model.compile(optimizer=optimizer, loss='mae', metrics=('mse','mae'))
    else:
        model = load_model( model_file) #获取前面模型训练的参数
        optimizer = Adam(learning_rate = lr)
        model.compile(optimizer=optimizer, loss='mae', metrics=('mse','mae'))
    # model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    logging.info("Loaded model from disk")

    # callback_list = []
    # check_point = ModelCheckpoint('{}/modellast.h5'.format(result_folder),
    #                             monitor='val_loss',
    #                             verbose=0,
    #                             save_best_only=False,
    #                             save_weights_only=False,
    #                             mode='auto',
    #                             period=1)
    # callback_list.append(check_point)
    # tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    # callback_list.append(tensor_board)
    logging.info("begin fitting")
    train_data, test_data = prepare_dataseq(data_dir, batch_size)#本质上获得数据一个迭代器 #可以用GPT来 写个pytorch版本
    train_data = tf.data.Dataset.from_generator(train_data,output_types=(tf.float32,tf.float32))
    test_data = tf.data.Dataset.from_generator(test_data,output_types=(tf.float32,tf.float32))
    if n_gpus > 1:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        test_data = test_data.with_options(options)
    history = model.fit(train_data, validation_data=test_data,
                                  epochs=epochs, steps_per_epoch=steps_per_epoch,validation_steps=np.ceil(0.1*steps_per_epoch),
                                  verbose=1)#这里开始训练
                                #   callbacks=callback_list)
    # if n_gpus>1:
    #     model_from_multimodel = model.get_layer('model_1')
    #     model_from_multimodel.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    #     model_from_multimodel.save(outFile)
    # else:
    model.save(outFile)
    return history

def other_train3D_continue(outFile,
                     model_file,
                     data_dir='data',
                     result_folder='results',
                     epochs=40,
                     lr=0.0004,
                     steps_per_epoch=128,
                     batch_size=64,
                     n_gpus=2):
    # logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
    # logging.debug('The tf message level {}'.format(os.environ['TF_CPP_MIN_LOG_LEVEL']))

    # metrics = ('mse', 'mae')
    # _metrics = [eval('loss_%s()' % m) for m in metrics]
    # optimizer = Adam(lr=lr)

    # model = load_model( model_file) # weight is a model

    # 这段代码使用了 TensorFlow 中的 MirroredStrategy 分布式策略。这个策略通常用于在多个 GPU 上进行训练。在使用 MirroredStrategy 时，TensorFlow 会在每个 GPU 上创建一个模型副本，每个副本都包含完整的模型。然后，每个副本都会处理输入数据的一个子集，并计算梯度。最后，所有梯度都会汇总并平均，然后应用于每个模型副本的参数更新。
    # 这种方式可以显著提高训练速度，特别是当你有多个 GPU 可用时。通过使用 MirroredStrategy，你可以方便地利用多个 GPU 的并行计算能力，加速模型的训练过程。

    # 定义自定义损失函数
    def custom_loss(y_true, y_pred):
        # 计算y_true与y_pred的均方误差
        mse_original = tf.reduce_mean(tf.square(y_true - y_pred))

        # 计算傅里叶变换
        y_true_fft = tf.signal.fft(tf.cast(tf.expand_dims(y_true, axis=-1), tf.complex64))
        y_pred_fft = tf.signal.fft(tf.cast(tf.expand_dims(y_pred, axis=-1), tf.complex64))

        # 计算傅里叶空间的均方误差
        mse_fft = tf.reduce_mean(tf.square(tf.abs(y_true_fft) - tf.abs(y_pred_fft)))

        # 返回总损失
        return mse_fft
    def custom_ssim_loss(y_true, y_pred):
        import tensorflow as tf
        def normalize(image):
            """归一化图像到 [0, 1] 范围"""
            min_val = tf.reduce_min(image)
            max_val = tf.reduce_max(image)
            return (image - min_val) / (max_val - min_val + tf.keras.backend.epsilon())  # 防止除以零
        y_true = normalize(y_true)
        y_pred = normalize(y_pred)
        # 计算 SSIM
        ssim_value = tf.image.ssim(y_true, y_pred, max_val=1.0)
        return -ssim_value
    
    strategy = tf.distribute.MirroredStrategy()
    # train_data = strategy.experimental_distribute_dataset(train_data)
    # test_data = strategy.experimental_distribute_dataset(test_data)
    if n_gpus > 1:
        with strategy.scope():
            model = load_model(model_file, custom_objects={'custom_ssim_loss': custom_ssim_loss})  # 获取前面模型训练的参数
            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss=custom_ssim_loss, metrics=('mse', 'mae'))
            # model.compile(optimizer=optimizer, loss='mae', metrics=('mse','mae'))
    else:
        model = load_model(model_file, custom_objects={'custom_ssim_loss': custom_ssim_loss})  # 获取前面模型训练的参数
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=custom_ssim_loss, metrics=('mse', 'mae'))
        # model.compile(optimizer=optimizer, loss='mae', metrics=('mse','mae'))
    # model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    logging.info("Loaded model from disk")

    # callback_list = []
    # check_point = ModelCheckpoint('{}/modellast.h5'.format(result_folder),
    #                             monitor='val_loss',
    #                             verbose=0,
    #                             save_best_only=False,
    #                             save_weights_only=False,
    #                             mode='auto',
    #                             period=1)
    # callback_list.append(check_point)
    # tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    # callback_list.append(tensor_board)
    logging.info("begin fitting")
    train_data, test_data = prepare_dataseq(data_dir, batch_size)  # 本质上获得数据一个迭代器 #可以用GPT来 写个pytorch版本
    train_data = tf.data.Dataset.from_generator(train_data, output_types=(tf.float32, tf.float32))
    test_data = tf.data.Dataset.from_generator(test_data, output_types=(tf.float32, tf.float32))
    if n_gpus > 1:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        test_data = test_data.with_options(options)
    history = model.fit(train_data, validation_data=test_data,
                        epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=np.ceil(0.1 * steps_per_epoch),
                        verbose=1)  # 这里开始训练
    #   callbacks=callback_list)
    # if n_gpus>1:
    #     model_from_multimodel = model.get_layer('model_1')
    #     model_from_multimodel.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    #     model_from_multimodel.save(outFile)
    # else:
    model.save(outFile)
    return history

def prepare_first_other_model(settings):
    model = other_Unet(filter_base=settings.filter_base,
            depth=settings.unet_depth,
            convs_per_depth=settings.convs_per_depth,
            kernel=settings.kernel,
            batch_norm=settings.batch_normalization,
            dropout=settings.drop_out,
            pool=settings.pool,
            residual = settings.residual,
            last_activation = 'linear',
            loss = 'mae',
            lr = settings.learning_rate)
    init_model_name = settings.other_result_dir+'/model_iter00.h5'
    model.save(init_model_name)
    return settings
def prepare_first_model(settings):
    model = Unet(filter_base=settings.filter_base, 
            depth=settings.unet_depth, 
            convs_per_depth=settings.convs_per_depth,
            kernel=settings.kernel,
            batch_norm=settings.batch_normalization, 
            dropout=settings.drop_out,
            pool=settings.pool,
            residual = settings.residual,
            last_activation = 'linear',
            loss = 'mae',
            lr = settings.learning_rate)
    init_model_name = settings.result_dir+'/model_iter00.h5'
    model.save(init_model_name)
    return settings

def train_data(settings):
    history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count),
                                        settings.unet_init_model,
                                        data_dir = settings.data_dir,
                                        result_folder = settings.result_dir,
                                        epochs=settings.epochs,
                                        steps_per_epoch=settings.steps_per_epoch,
                                        batch_size=settings.batch_size,
                                        lr = settings.learning_rate,
                                        n_gpus=settings.ngpus)

    # if settings.iter_count == 0 and settings.pretrained_model is None :
    #     history = train3D_seq('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1),
    #                                 data_dir = settings.data_dir,
    #                                 result_folder = settings.result_dir,
    #                                 epochs = settings.epochs,
    #                                 steps_per_epoch = settings.steps_per_epoch,
    #                                 batch_size = settings.batch_size,
    #                                 lr = settings.lr,
    #                                 dropout = settings.drop_out,
    #                                 filter_base = settings.filter_base,
    #                                 depth=settings.unet_depth,
    #                                 convs_per_depth = settings.convs_per_depth,
    #                                 batch_norm = settings.batch_normalization,
    #                                 kernel = settings.kernel,
    #                                 n_gpus = settings.ngpus)
    # elif settings.iter_count == 0 and settings.pretrained_model is not None:
    #     history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1),
    #                                     settings.pretrained_model,
    #                                     data_dir = settings.data_dir,
    #                                     result_folder = settings.result_dir,
    #                                     epochs=settings.epochs,
    #                                     steps_per_epoch=settings.steps_per_epoch,
    #                                     batch_size=settings.batch_size,
    #                                     lr = settings.lr,
    #                                     n_gpus=settings.ngpus)

    # else:
    #     history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1),
    #                                     '{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count),
    #                                     data_dir = settings.data_dir,
    #                                     result_folder = settings.result_dir,
    #                                     epochs=settings.epochs,
    #                                     steps_per_epoch=settings.steps_per_epoch,
    #                                     batch_size=settings.batch_size,
    #                                     n_gpus=settings.ngpus)

    return history

def train_other_data(settings):
    history = other_train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.other_result_dir,settings.iter_count),
                                        settings.other_unet_init_model,
                                        data_dir = settings.other_data_dir,
                                        result_folder = settings.other_result_dir,
                                        epochs=settings.epochs,
                                        steps_per_epoch=settings.steps_per_epoch,
                                        batch_size=settings.batch_size,
                                        lr = settings.learning_rate,
                                        n_gpus=settings.ngpus)

    # if settings.iter_count == 0 and settings.pretrained_model is None :
    #     history = train3D_seq('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1),
    #                                 data_dir = settings.data_dir,
    #                                 result_folder = settings.result_dir,
    #                                 epochs = settings.epochs,
    #                                 steps_per_epoch = settings.steps_per_epoch,
    #                                 batch_size = settings.batch_size,
    #                                 lr = settings.lr,
    #                                 dropout = settings.drop_out,
    #                                 filter_base = settings.filter_base,
    #                                 depth=settings.unet_depth,
    #                                 convs_per_depth = settings.convs_per_depth,
    #                                 batch_norm = settings.batch_normalization,
    #                                 kernel = settings.kernel,
    #                                 n_gpus = settings.ngpus)
    # elif settings.iter_count == 0 and settings.pretrained_model is not None:
    #     history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1),
    #                                     settings.pretrained_model,
    #                                     data_dir = settings.data_dir,
    #                                     result_folder = settings.result_dir,
    #                                     epochs=settings.epochs,
    #                                     steps_per_epoch=settings.steps_per_epoch,
    #                                     batch_size=settings.batch_size,
    #                                     lr = settings.lr,
    #                                     n_gpus=settings.ngpus)

    # else:
    #     history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1),
    #                                     '{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count),
    #                                     data_dir = settings.data_dir,
    #                                     result_folder = settings.result_dir,
    #                                     epochs=settings.epochs,
    #                                     steps_per_epoch=settings.steps_per_epoch,
    #                                     batch_size=settings.batch_size,
    #                                     n_gpus=settings.ngpus)

    return history



