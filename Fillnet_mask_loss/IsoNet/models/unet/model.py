from IsoNet.models.unet import builder,builder_fullconv,builder_fullconv_old,build_old_net
from tensorflow.keras.layers import Input,Add,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def other_Unet(filter_base=32,
         depth=3,
         convs_per_depth=3,
         kernel=(3, 3, 3),
         batch_norm=True,
         dropout=0.3,
         pool=None, residual=True,
         last_activation='linear',
         loss='mae',
         lr=0.0004,
         test_shape=None):
    # model = builder.build_unet(filter_base,depth,convs_per_depth,
    #            kernel,
    #            batch_norm,
    #            dropout,
    #            pool)
    model = builder_fullconv.other_build_unet(filter_base, depth, convs_per_depth,
                                        kernel,
                                        batch_norm,
                                        dropout,
                                        pool)

    # 定义自定义损失函数
    def compute_mse(y_true, y_pred, mw3d_0, mw3d_1, mw3d_2, mw3d_3, mw3d_4):
        import tensorflow as tf
        # 对 y_true 和 y_pred 做 3D FFT 变换
        y_true_fft = tf.signal.fft3d(tf.cast(tf.squeeze(y_true, axis=-1), tf.complex64))
        y_pred_fft = tf.signal.fft3d(tf.cast(tf.squeeze(y_pred, axis=-1), tf.complex64))

        # 计算各个 filter 的值
        sum_fft_mw0 = tf.reduce_sum(tf.abs(y_pred_fft * mw3d_0))
        sum_fft_mw1 = tf.reduce_sum(tf.abs(y_pred_fft * mw3d_1))
        sum_fft_mw2 = tf.reduce_sum(tf.abs(y_pred_fft * mw3d_2))
        sum_fft_mw3 = tf.reduce_sum(tf.abs(y_pred_fft * mw3d_3))
        sum_fft_mw4 = tf.reduce_sum(tf.abs(y_pred_fft * mw3d_4))
        tf.print("sum_fft_mw0", sum_fft_mw0)
        tf.print("sum_fft_mw1", sum_fft_mw1)
        tf.print("sum_fft_mw2", sum_fft_mw2)
        tf.print("sum_fft_mw3", sum_fft_mw3)
        tf.print("sum_fft_mw4", sum_fft_mw4)

        # 创建计算 MSE 的函数
        def compute_mse_mw0():
            reverse_mv = 1 - mw3d_0
            y_pred_fft_mw = y_pred_fft * reverse_mv
            y_true_fft_mw = y_true_fft * reverse_mv
            tf.print("Using mw3d_0")
            return tf.reduce_mean(tf.square(tf.abs(y_pred_fft_mw - y_true_fft_mw)))

        def compute_mse_mw1():
            reverse_mv = 1 - mw3d_1
            y_pred_fft_mw = y_pred_fft * reverse_mv
            y_true_fft_mw = y_true_fft * reverse_mv
            tf.print("Using mw3d_1")
            return tf.reduce_mean(tf.square(tf.abs(y_pred_fft_mw - y_true_fft_mw)))

        def compute_mse_mw2():
            reverse_mv = 1 - mw3d_2
            y_pred_fft_mw = y_pred_fft * reverse_mv
            y_true_fft_mw = y_true_fft * reverse_mv
            tf.print("Using mw3d_2")
            return tf.reduce_mean(tf.square(tf.abs(y_pred_fft_mw - y_true_fft_mw)))

        def compute_mse_mw3():
            reverse_mv = 1 - mw3d_3
            y_pred_fft_mw = y_pred_fft * reverse_mv
            y_true_fft_mw = y_true_fft * reverse_mv
            tf.print("Using mw3d_3")
            return tf.reduce_mean(tf.square(tf.abs(y_pred_fft_mw - y_true_fft_mw)))

        def compute_mse_mw4():
            reverse_mv = 1 - mw3d_4
            y_pred_fft_mw = y_pred_fft * reverse_mv
            y_true_fft_mw = y_true_fft * reverse_mv
            tf.print("Using mw3d_4")
            return tf.reduce_mean(tf.square(tf.abs(y_pred_fft_mw - y_true_fft_mw)))

        # 使用 tf.cond 进行条件判断
        condition_0 = tf.logical_and(tf.logical_and(sum_fft_mw0 < sum_fft_mw1, sum_fft_mw0 < sum_fft_mw2),
                                     tf.logical_and(sum_fft_mw0 < sum_fft_mw3, sum_fft_mw0 < sum_fft_mw4))

        condition_1 = tf.logical_and(tf.logical_and(sum_fft_mw1 < sum_fft_mw0, sum_fft_mw1 < sum_fft_mw2),
                                     tf.logical_and(sum_fft_mw1 < sum_fft_mw3, sum_fft_mw1 < sum_fft_mw4))

        condition_2 = tf.logical_and(tf.logical_and(sum_fft_mw2 < sum_fft_mw0, sum_fft_mw2 < sum_fft_mw1),
                                     tf.logical_and(sum_fft_mw2 < sum_fft_mw3, sum_fft_mw2 < sum_fft_mw4))

        condition_3 = tf.logical_and(tf.logical_and(sum_fft_mw3 < sum_fft_mw0, sum_fft_mw3 < sum_fft_mw1),
                                     tf.logical_and(sum_fft_mw3 < sum_fft_mw2, sum_fft_mw3 < sum_fft_mw4))

        condition_4 = tf.logical_and(tf.logical_and(sum_fft_mw4 < sum_fft_mw0, sum_fft_mw4 < sum_fft_mw1),
                                     tf.logical_and(sum_fft_mw4 < sum_fft_mw2, sum_fft_mw4 < sum_fft_mw3))

        # 使用 tf.cond 进行选择
        mse_mw_fft = tf.cond(condition_0, compute_mse_mw0,
                             lambda: tf.cond(condition_1, compute_mse_mw1,
                                             lambda: tf.cond(condition_2, compute_mse_mw2,
                                                             lambda: tf.cond(condition_3, compute_mse_mw3,
                                                                             lambda: tf.cond(condition_4,
                                                                                             compute_mse_mw4,
                                                                                             compute_mse_mw4)))))
        return mse_mw_fft

    def custom_suplus_loss(y_true, y_pred):
        import tensorflow as tf
        from IsoNet.preprocessing.simulate import mw3d_list
        tf.print(tf.shape(y_true))
        # 获取 mw3d_list
        mw3d_list = mw3d_list(64)
        mw3d_0 = tf.expand_dims(tf.signal.fftshift(tf.cast(mw3d_list[0], tf.complex64)), axis=0)
        mw3d_1 = tf.expand_dims(tf.signal.fftshift(tf.cast(mw3d_list[1], tf.complex64)), axis=0)
        mw3d_2 = tf.expand_dims(tf.signal.fftshift(tf.cast(mw3d_list[2], tf.complex64)), axis=0)
        mw3d_3 = tf.expand_dims(tf.signal.fftshift(tf.cast(mw3d_list[3], tf.complex64)), axis=0)
        mw3d_4 = tf.expand_dims(tf.signal.fftshift(tf.cast(mw3d_list[4], tf.complex64)), axis=0)
        # 对批量中的每个样本计算损失
        mse_losses = tf.map_fn(lambda x: compute_mse(x[0], x[1], mw3d_0, mw3d_1, mw3d_2, mw3d_3, mw3d_4),
                               (y_true, y_pred), dtype=tf.float32)
        # 返回所有样本的平均损失
        return tf.reduce_mean(mse_losses)


    # model = build_old_net.unet_block(filter_base,depth,convs_per_depth,
    #            kernel,
    #            batch_norm,
    #            dropout,
    #            (2,2,2))
    # import os
    # import sys
    # cwd = os.getcwd()
    # sys.path.insert(0,cwd)
    # import train_settings
    # model = builder_fullconv_old.build_unet(train_settings)

    # ***** Construct complete model from unet output
    if test_shape is None:
        inputs = Input((None, None, None, 1))  # 输入形状表示可以处理任意批量大小、任意序列长度、任意特征数量，且具有单个通道的数据。
    elif type(test_shape) is int:
        inputs = Input((test_shape, test_shape, test_shape, 1))
    unet_out = model(inputs)
    if residual:
        outputs = Add()([unet_out, inputs])
    else:
        outputs = unet_out
    # outputs = Activation(activation=last_activation)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=lr)

    # 对损失函数进行改进
    loss = custom_suplus_loss
    # if loss == 'mae' or loss == 'mse':
    #     metrics = ('mse', 'mae')

    # model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.compile(optimizer=optimizer, loss=loss, metrics=('mse', 'mae'))
    return model

def Unet(filter_base=32,
        depth=3,
        convs_per_depth=3,
        kernel=(3,3,3),
        batch_norm=True,
        dropout=0.3,
        pool=None,residual = True,
        last_activation = 'linear',
        loss = 'mae',
        lr = 0.0004,
        test_shape=None):

    # model = builder.build_unet(filter_base,depth,convs_per_depth,
    #            kernel,
    #            batch_norm,
    #            dropout,
    #            pool)
    model = builder_fullconv.build_unet(filter_base,depth,convs_per_depth,
            kernel,
            batch_norm,
            dropout,
            pool)

    # model = build_old_net.unet_block(filter_base,depth,convs_per_depth,
    #            kernel,
    #            batch_norm,
    #            dropout,
    #            (2,2,2))
    # import os
    # import sys
    # cwd = os.getcwd()
    # sys.path.insert(0,cwd)
    # import train_settings 
    # model = builder_fullconv_old.build_unet(train_settings)
    
    #***** Construct complete model from unet output
    if test_shape is None:
        inputs = Input((None,None,None,1)) # 输入形状表示可以处理任意批量大小、任意序列长度、任意特征数量，且具有单个通道的数据。
    elif type(test_shape) is int:
        inputs = Input((test_shape,test_shape,test_shape,1))
    unet_out = model(inputs) 
    if residual:
        outputs = Add()([unet_out, inputs])
    else:
        outputs = unet_out
    # outputs = Activation(activation=last_activation)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=lr)
    if loss == 'mae' or loss == 'mse':
        metrics = ('mse', 'mae')

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

if __name__ == "__main__":
    keras_model = Unet(filter_base=64,
        depth=3,
        convs_per_depth=3,
        kernel=(3,3,3),
        batch_norm=True,
        dropout=0.5,
        pool=(2,2,2),residual = True,
        last_activation = 'linear',
        loss = 'mae',
        lr = 0.0004)
    print(keras_model.summary())
