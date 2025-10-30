from IsoNet.models.unet.blocks import conv_blocks, activation_my, decoder_block
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, MaxPooling3D, UpSampling3D, AveragePooling3D,Conv2D,Add,Conv2DTranspose,Conv3D,Conv3DTranspose,Dropout,BatchNormalization,Activation,LeakyReLU
from tensorflow.keras.layers import Concatenate

# define a decoder block

def build_unet(filter_base=32,depth=3,convs_per_depth=3,
               kernel=(3,3,3),
               batch_norm=True,
               dropout=0.0,
               pool=None):
    resnet = False
    # pool = (2,2,2)
    def _func(inputs):
        concatenate = []
        layer = inputs
        #begin contracting path
        for n in range(depth):
            current_depth_start = layer
            for i in range(convs_per_depth):
                layer = conv_blocks(filter_base*2**n,kernel,dropout=dropout,
                                    batch_norm=batch_norm,activation = "LeakyReLU",
                                    name="down_level_%s_no_%s" % (n, i))(layer)
            # if use res_block strategy
            if resnet:
                start_conv = Conv3D(filter_base*2**n,(1,1,1),
                            padding='same',kernel_initializer="he_uniform")(current_depth_start)
                layer = Add()([start_conv,layer])
                layer = activation_my("LeakyReLU")(layer)
            # save the last layer of current depth
            concatenate.append(layer)
            # dimension reduction with pooling or stride 2 convolution
            if pool is not None:
                layer = MaxPooling3D(pool)(layer)
            else:
                layer = conv_blocks(filter_base*2**n,kernel,strides=(2,2,2),activation="LeakyReLU")(layer)
        # begin bottleneck path
        b = layer
        bottle_start = layer
        for i in range(convs_per_depth-2):
            b = conv_blocks(filter_base*2**depth,kernel,dropout=None,
                                    batch_norm=None,activation="LeakyReLU",
                                    name="bottleneck_no_%s" % (i))(b)
        layer = conv_blocks(filter_base*2**(depth-1),kernel,dropout=None,
                                    batch_norm=None,activation=None,
                                    name="bottleneck_no_%s" % (convs_per_depth))(b)
        if resnet:
            layer = Add()([bottle_start,layer])
            layer = activation_my("LeakyReLU")(layer)


        for n in reversed(range(depth)):
            if pool is not None:
                layer = Concatenate(axis=-1)([UpSampling3D(pool)(layer),concatenate[n]])
            else:
                layer = decoder_block(layer, concatenate[n], filter_base*2**n, dropout=False,batchnorm=False,activation="LeakyReLU")
            current_depth_start = layer
            for i in range(convs_per_depth):
                layer = conv_blocks(filter_base * 2 ** n, kernel, dropout=dropout,
                                    batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, i),activation ="LeakyReLU")(layer)
            if resnet:
                start_conv = Conv3D(filter_base*2**n,(1,1,1),
                            padding='same',kernel_initializer="he_uniform")(current_depth_start)
                layer = Add()([start_conv,layer])
                layer = activation_my("LeakyReLU")(layer)
        final = conv_blocks(1, (1,1,1), dropout=None,activation='linear',
                                    batch_norm=None,name="fullconv_out")(layer)
        return final
    return _func

from tensorflow.keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, Dense, Reshape, Multiply, Add, Conv3D, Activation
import tensorflow as tf
#通道和空间注意力（CBAM 模块）
def cbam_block(input_tensor, reduction_ratio=8):

    """
    Convolutional Block Attention Module (CBAM)
    Combines Channel Attention and Spatial Attention.
    """
    channel_axis = -1
    input_channels = input_tensor.shape[channel_axis]

    # Channel Attention
    avg_pool = GlobalAveragePooling3D()(input_tensor)
    max_pool = GlobalMaxPooling3D()(input_tensor)

    dense_avg = Dense(input_channels // reduction_ratio, activation='relu')(avg_pool)
    dense_avg = Dense(input_channels, activation='sigmoid')(dense_avg)

    dense_max = Dense(input_channels // reduction_ratio, activation='relu')(max_pool)
    dense_max = Dense(input_channels, activation='sigmoid')(dense_max)

    channel_attention = Add()([dense_avg, dense_max])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = Reshape((1, 1, 1, input_channels))(channel_attention)
    channel_refined = Multiply()([input_tensor, channel_attention])

    # Spatial Attention
    avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)

    spatial_attention = Conv3D(1, (7, 7, 7), padding="same", activation='sigmoid')(concat)
    spatial_refined = Multiply()([channel_refined, spatial_attention])

    return spatial_refined

def other_build_unet(filter_base=32,depth=3,convs_per_depth=3,
               kernel=(3,3,3),
               batch_norm=True,
               dropout=0.0,
               pool=None):
    resnet = False
    # pool = (2,2,2)
    def _func(inputs):
        concatenate = []
        layer = inputs
        fft_concatenate = []
        fft_data = tf.signal.fft3d(tf.cast(tf.expand_dims(inputs, axis=-1), tf.complex64))
        fft_abs_data = tf.abs(fft_data)
        fft_sign_data = tf.sign(tf.math.real(fft_data))
        fft_layer = tf.squeeze(fft_abs_data * fft_sign_data, axis=-1)
        print(fft_layer.shape)

        #begin contracting path
        for n in range(depth):
            current_depth_start = layer
            fft_current_depth_start = fft_layer
            for i in range(convs_per_depth):
                layer = conv_blocks(filter_base*2**n,kernel,dropout=dropout,
                                    batch_norm=batch_norm,activation = "LeakyReLU",
                                    name="down_level_%s_no_%s" % (n, i))(layer)

                fft_layer = conv_blocks(filter_base*2**n,kernel,dropout=dropout,
                                    batch_norm=batch_norm,activation = "LeakyReLU",
                                    name="fft_down_level_%s_no_%s" % (n, i))(fft_layer)
            # if use res_block strategy
            if resnet:
                start_conv = Conv3D(filter_base*2**n,(1,1,1),
                            padding='same',kernel_initializer="he_uniform")(current_depth_start)
                layer = Add()([start_conv,layer])
                layer = activation_my("LeakyReLU")(layer)

                fft_start_conv = Conv3D(filter_base*2**n,(1,1,1),
                            padding='same',kernel_initializer="he_uniform")(fft_current_depth_start)
                fft_layer = Add()([fft_start_conv,fft_layer])
                fft_layer = activation_my("LeakyReLU")(fft_layer)

            # save the last layer of current depth
            concatenate.append(layer)
            fft_concatenate.append(fft_layer)

            # dimension reduction with pooling or stride 2 convolution
            if pool is not None:
                layer = MaxPooling3D(pool)(layer)

                fft_layer = MaxPooling3D(pool)(fft_layer)
            else:
                layer = conv_blocks(filter_base*2**n,kernel,strides=(2,2,2),activation="LeakyReLU")(layer)

                fft_layer = conv_blocks(filter_base*2**n,kernel,strides=(2,2,2),activation="LeakyReLU")(fft_layer)
        # begin bottleneck path
        b = layer
        bottle_start = layer

        fft_b = fft_layer
        fft_bottle_start = fft_layer

        for i in range(convs_per_depth-2):
            b = conv_blocks(filter_base*2**depth,kernel,dropout=None,
                                    batch_norm=None,activation="LeakyReLU",
                                    name="bottleneck_no_%s" % (i))(b)
            fft_b = conv_blocks(filter_base*2**depth,kernel,dropout=None,

                                    batch_norm=None,activation="LeakyReLU",
                                    name="fft_bottleneck_no_%s" % (i))(fft_b)

        layer = conv_blocks(filter_base*2**(depth-1),kernel,dropout=None,
                                    batch_norm=None,activation=None,
                                    name="bottleneck_no_%s" % (convs_per_depth))(b)
        fft_layer = conv_blocks(filter_base*2**(depth-1),kernel,dropout=None,
                                    batch_norm=None,activation=None,
                                    name="fft_bottleneck_no_%s" % (convs_per_depth))(fft_b)
        if resnet:
            layer = Add()([bottle_start,layer])
            layer = activation_my("LeakyReLU")(layer)

            fft_layer = Add()([fft_bottle_start,fft_layer])
            fft_layer = activation_my("LeakyReLU")(fft_layer)

        add_layer = Add()([layer,fft_layer])

        for n in reversed(range(depth)):
            add_concatenate = Add()([concatenate[n],fft_concatenate[n]])
            skip_connection = cbam_block(add_concatenate)

            if pool is not None:
                add_layer = Concatenate(axis=-1)([UpSampling3D(pool)(add_layer),skip_connection])
            else:
                add_layer = decoder_block(add_layer, skip_connection, filter_base*2**n, dropout=False,batchnorm=False,activation="LeakyReLU")
            add_current_depth_start = add_layer
            for i in range(convs_per_depth):
                add_layer = conv_blocks(filter_base * 2 ** n, kernel, dropout=dropout,
                                    batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, i),activation ="LeakyReLU")(add_layer)
            if resnet:
                add_start_conv = Conv3D(filter_base*2**n,(1,1,1),
                            padding='same',kernel_initializer="he_uniform")(add_current_depth_start)
                add_layer = Add()([add_start_conv,add_layer])
                add_layer = activation_my("LeakyReLU")(add_layer)
        final = conv_blocks(1, (1,1,1), dropout=None,activation='linear',
                                    batch_norm=None,name="fullconv_out")(add_layer)
        return final
    return _func