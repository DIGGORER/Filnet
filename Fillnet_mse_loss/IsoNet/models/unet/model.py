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
    def custom_loss(y_true, y_pred):
        # 计算y_true与y_pred的均方误差
        import tensorflow as tf
        mse_original = tf.reduce_mean(tf.square(y_true - y_pred))

        # 进行三维傅里叶变换
        y_true_fft = tf.signal.fft3d(tf.cast(y_true, tf.complex64))
        y_pred_fft = tf.signal.fft3d(tf.cast(y_pred, tf.complex64))

        # 计算傅里叶空间的均方误差
        mse_fft = tf.reduce_mean(tf.square(tf.abs(y_true_fft) - tf.abs(y_pred_fft)))

        # 返回总损失
        return mse_original


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
    loss = custom_loss
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
