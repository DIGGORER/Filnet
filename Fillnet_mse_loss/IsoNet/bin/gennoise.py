import sys
import os,traceback
from IsoNet.util.dict2attr import Arg,idx2list
from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes
from IsoNet.util.metadata import MetaData
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os
import sys
import mrcfile
import numpy as np

from IsoNet.models.unet import builder,builder_fullconv,builder_fullconv_old,build_old_net
from tensorflow.keras.layers import Input,Add,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
def extract_tomos(settings):
    md = MetaData()
    md.read(settings.star_file)
    if len(md) == 0:
        sys.exit("No input exists. Please check it in input folder!")
    # 读取tomograms.star相关数据集信息 列如deconv和mask处理过后的文件位置
    for it in md:
        if settings.tomo_idx is None or str(it.rlnIndex) in settings.tomo_idx:  # setting.tomo_idx 是你要提取数据集编号 而it.rlnIndex是现在遍历的编号
            pixel_size = it.rlnPixelSize
            if "rlnDeconvTomoName" in md.getLabels() and os.path.isfile(it.rlnDeconvTomoName):
                # 经过deconv处理过的数据集
                with mrcfile.open(it.rlnDeconvTomoName, permissive=True) as mrcData:
                    dec_data = mrcData.data.astype(np.float32)
                print("Extract from origional tomogram {}".format(it.rlnMicrographName))
                # 没有经过deconv处理过的数据集 用原始的数据集
                with mrcfile.open(it.rlnMicrographName, permissive=True) as mrcData:
                    orig_data = mrcData.data.astype(np.float32)
            # 经过mask处理过的数据集
            if "rlnMaskName" in md.getLabels() and it.rlnMaskName not in [None, "None"]:
                with mrcfile.open(it.rlnMaskName, permissive=True) as m:
                    mask_data = m.data
            else:  # 没有进行mask处理的数据集
                mask_data = None
            # seed {tuple:3 0:ndarray 100 1:ndarray 100 2:ndarray 100}
            it.rlnNumberSubtomo = 2000
            seeds = create_cube_seeds(orig_data, it.rlnNumberSubtomo, settings.crop_size, mask=mask_data)  # 这里注意
            # subtomos {ndarray:{100,80,80,80}}
            oritomos = crop_cubes(orig_data, seeds, settings.crop_size)#获取原始数据集切片
            dectomos = crop_cubes(dec_data, seeds, settings.crop_size)#获取经过deconv后的数据集切片
            # save sampled subtomo to {results_dir}/subtomos instead of subtomo_dir (as previously does)
            base_name = os.path.splitext(os.path.basename(it.rlnMicrographName))[0]

            for j, s in enumerate(oritomos):#对原始的数据集切片保存到noisedata/origion
                im_name = '{}/{}_{:0>6d}.mrc'.format(settings.subtomo_folder[0], base_name, j)
                with mrcfile.new(im_name, overwrite=True) as output_mrc:
                    output_mrc.set_data(s.astype(np.float32))

            for j, s in enumerate(dectomos):#对经过deconv的数据集切片保存到noisedata/dencov
                im_name = '{}/{}_{:0>6d}.mrc'.format(settings.subtomo_folder[1], base_name, j)
                with mrcfile.new(im_name, overwrite=True) as output_mrc:
                    output_mrc.set_data(s.astype(np.float32))
def extract_origion_and_deconv(
        star_file: str = "tomograms.star",
        subtomo_folder = ["noisedata/origion","noisedata/dencov"],
        cube_size: int = 64,
        crop_size: int = None,
        log_level: str="info",
        tomo_idx = None
        ):
    d = locals()  # 将变量和对应的值做成字典赋值给d
    d_args = Arg(d)
    print("d_args", d_args)
    try:
        if os.path.isdir(subtomo_folder[0]):
            import shutil
            shutil.rmtree(subtomo_folder[0])  # 删除subtomo文件夹
        os.mkdir(subtomo_folder[0])

        if os.path.isdir(subtomo_folder[1]):
            import shutil
            shutil.rmtree(subtomo_folder[1])  # 删除subtomo文件夹
        os.mkdir(subtomo_folder[1])

        if crop_size is None:
            d_args.crop_size = cube_size + 16
        else:
            d_args.crop_size = crop_size
        d_args.subtomo_folder = subtomo_folder
        d_args.tomo_idx = idx2list(tomo_idx)
        extract_tomos(d_args)
    except Exception:
        error_text = traceback.format_exc()
        f = open('log.txt', 'a+')
        f.write(error_text)
        f.close()

def mkfloder(): #生成noise文件夹 mkfloder_address是个文件夹
    mkfloder_address = ['noisedata','dencov','origion','data','train_x','train_y','val_x','val_y','test_x','test_y']
    data_dir = './'
    if not os.path.exists(data_dir+mkfloder_address[0]): # 是否存在这个noisedata 这个目录其实是 noisedata这个目录
        os.makedirs(data_dir+mkfloder_address[0]) # 如果没有则创建这个目录
    data_dir = data_dir + mkfloder_address[0] + '/'
    if not os.path.exists(data_dir + mkfloder_address[1]):#'dencov'
        os.makedirs(data_dir + mkfloder_address[1])
    if not os.path.exists(data_dir + mkfloder_address[2]):#'origion'
        os.makedirs(data_dir + mkfloder_address[2])
    if not os.path.exists(data_dir + mkfloder_address[3]):#创建'data'
        os.makedirs(data_dir + mkfloder_address[3])
    data_dir = data_dir + mkfloder_address[3] + '/'

    for i in range(4,10):#创建'train_x','train_y','test_x','test_y'
        if not os.path.exists(data_dir + mkfloder_address[i]):
            os.makedirs(data_dir + mkfloder_address[i])
def divide_train_test():
    dir_floder = 'noisedata/data/'
    dirs_tomake = ['train_x', 'train_y', 'val_x', 'val_y', 'test_x', 'test_y']
    dirs_dec_ori = ['dencov','origion']
    for tomake in dirs_tomake: #这里检查train_x 等其他文件夹是否存在如果存在
        if os.path.isdir(dir_floder+tomake):
            import shutil
            shutil.rmtree(dir_floder+tomake)  # 删除subtomo文件夹
        os.mkdir(dir_floder+tomake)
    #获取当前origion目录下的所有文件做成一个列表 然后因为deconv 和origion中子文件都是相同的 所以只要一个就行
    dir_noisedata = 'noisedata'

    p = '{}/{}/'.format(dir_noisedata, dirs_dec_ori[0])
    path_all = sorted([f for f in os.listdir(p)])    #获取origion 和 deconv 中所有子文件
    print(path_all)
    len_path_all = len(path_all)
    use_len = int(len_path_all*0.1)  #随机取10%的样本当验证集
    val_path = np.random.choice(path_all,size=use_len,replace=False)

    temp_path = np.setdiff1d(path_all,val_path)
    test_path = np.random.choice(temp_path,size=use_len,replace=False)#10%的赝本当测试集
    train_path = np.setdiff1d(temp_path,test_path)#剩下80%当训练集
    #最后开始把样本从denov和origion分别复制到'train_x', 'train_y', 'test_x', 'test_y'
    for train in train_path:
        os.rename('{}/{}/{}'.format(dir_noisedata,dirs_dec_ori[0], train),
                  '{}/{}/{}'.format(dir_floder,dirs_tomake[0],train))
        os.rename('{}/{}/{}'.format(dir_noisedata,dirs_dec_ori[1], train),
                  '{}/{}/{}'.format(dir_floder,dirs_tomake[1],train))

    for val in val_path:
        os.rename('{}/{}/{}'.format(dir_noisedata, dirs_dec_ori[0], val),
                    '{}/{}/{}'.format(dir_floder, dirs_tomake[2], val))
        os.rename('{}/{}/{}'.format(dir_noisedata, dirs_dec_ori[1], val),
                    '{}/{}/{}'.format(dir_floder, dirs_tomake[3], val))

    for test in test_path:
        os.rename('{}/{}/{}'.format(dir_noisedata, dirs_dec_ori[0], test),
                    '{}/{}/{}'.format(dir_floder, dirs_tomake[4], test))
        os.rename('{}/{}/{}'.format(dir_noisedata, dirs_dec_ori[1], test),
                    '{}/{}/{}'.format(dir_floder, dirs_tomake[5], test))

#构建数据集函数
def prepare_dataseq(data_folder, batch_size):
    dirs_tomake = ['train_x','train_y', 'test_x', 'test_y', 'val_x', 'val_y']
    path_all = []
    for d in dirs_tomake:
        p = '{}/{}/'.format(data_folder, d)
        path_all.append(sorted([p+f for f in os.listdir(p)]))#其实也进行排序了
    train_data = get_gen(path_all[0], path_all[1], batch_size)
    test_data = get_gen(path_all[2], path_all[3], batch_size)
    val_data = get_gen(path_all[4], path_all[5], batch_size)
    return train_data, test_data ,val_data

#这段代码定义了一个名为 normalize 的函数，用于图像归一化。让我逐行解释其作用：
def normalize(x, percentile = True, pmin=4.0, pmax=96.0, axis=None, clip=False, eps=1e-20):
    """Percentile-based image normalization."""
    # 包括输入数据x、是否使用百分位数的归一化方法 percentile、百分位数的最小值 pmin 和最大值pmax、归一化的轴axis、是否剪裁归一化结果到[0, 1]范围内clip，以及一个很小的数值eps，用于防止除以零的情况。
    if percentile: #这一行检查是否使用基于百分位数的归一化方法。
        mi = np.percentile(x,pmin,axis=axis,keepdims=True)#如果axis=none 则把3维变1维 然后取4/100的数值 个人感觉把最小值去掉 然后取最小的值
        ma = np.percentile(x,pmax,axis=axis,keepdims=True)#如果axis=none 则把3维变1维 然后取96/100的数值 个人感觉把最高值去掉 然后取最高的值
        out = (x - mi) / ( ma - mi + eps )#计算归一化后的输出值，将输入数据 x 减去最小值 mi，再除以最大值 ma 减最小值 mi，加上一个很小的数值 eps 防止除以零。
        out = out.astype(np.float32)
        if clip:
            return np.clip(out,0,1)
        else:
            return out
    else:#否则进行普通的归一化 减去均值 除以方差
        out = (x-np.mean(x))/np.std(x)
        out = out.astype(np.float32)
        return out
def get_gen(x_set,y_set,batch_size,shuffle=True):
    def gen():
        while True:
            all_idx = np.arange(len(x_set))
            if shuffle:
                np.random.shuffle(all_idx)
            for i in range(len(x_set)//batch_size):
                idx = slice(i * batch_size,(i+1) * batch_size)
                idx = all_idx[idx]
                rx = np.array([mrcfile.open(x_set[j], permissive=True).data[:,:,:,np.newaxis]*-1 for j in idx])
                #图像归一化
                rx = normalize(rx,percentile=True)
                ry = np.array([mrcfile.open(y_set[j], permissive=True).data[:,:,:,np.newaxis]*-1 for j in idx])
                #图像归一化
                ry = normalize(ry,percentile=True)
                yield rx,ry
    return gen

def custom_loss(y_true, y_pred):
    # 计算y_true与y_pred的均方误差
    mse_original = tf.reduce_mean(tf.square(y_true - y_pred))

    # 计算傅里叶变换
    y_true_fft = tf.signal.fft(tf.cast(tf.expand_dims(y_true, axis=-1), tf.complex64))
    y_pred_fft = tf.signal.fft(tf.cast(tf.expand_dims(y_pred, axis=-1), tf.complex64))

    # 计算傅里叶空间的均方误差
    mse_fft = tf.reduce_mean(tf.square(tf.abs(y_true_fft) - tf.abs(y_pred_fft)))

    # 返回总损失
    return mse_original + mse_fft

def custom_fsc_loss(y_true, y_pred):
    # 计算y_true与y_pred的均方误差
    mse_original = tf.reduce_mean(tf.square(y_true - y_pred))

    # 计算傅里叶变换
    y_true_fft = tf.signal.fft(tf.cast(tf.expand_dims(y_true, axis=-1), tf.complex64))
    y_pred_fft = tf.signal.fft(tf.cast(tf.expand_dims(y_pred, axis=-1), tf.complex64))
    eps = 1e-10
    # 计算傅里叶空间的每个像素的fsc 范围 0——1
    fsc_fft = tf.reduce_mean(
        (tf.math.imag(y_true_fft) * tf.math.imag(y_pred_fft) +
         tf.math.real(y_true_fft) * tf.math.real(y_pred_fft)) /
        (tf.abs(y_true_fft) * tf.abs(y_pred_fft) + eps))
    # 返回总损失
    return 10*mse_original - fsc_fft

def custom_fmse_fsc_loss(y_true, y_pred):
    # 计算傅里叶变换
    y_true_fft = tf.signal.fft(tf.cast(tf.expand_dims(y_true, axis=-1), tf.complex64))
    y_pred_fft = tf.signal.fft(tf.cast(tf.expand_dims(y_pred, axis=-1), tf.complex64))

    # 计算傅里叶空间的均方误差
    mse_fft = tf.reduce_mean(tf.square(tf.abs(y_true_fft) - tf.abs(y_pred_fft)))
    eps = 1e-10
    # 计算傅里叶空间的每个像素的fsc 范围 0——1
    fsc_fft = tf.reduce_mean(
        (tf.math.imag(y_true_fft) * tf.math.imag(y_pred_fft) +
         tf.math.real(y_true_fft) * tf.math.real(y_pred_fft)) /
        (tf.abs(y_true_fft) * tf.abs(y_pred_fft) + eps))
    # 返回总损失
    return mse_fft - fsc_fft

def Unet(filter_base=32,
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
    model = builder_fullconv.build_unet(filter_base, depth, convs_per_depth,
                                        kernel,
                                        batch_norm,
                                        dropout,
                                        pool)

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
    loss = custom_fmse_fsc_loss
    # if loss == 'mae' or loss == 'mse':
    #     metrics = ('mse', 'mae')

    # model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.compile(optimizer=optimizer, loss=loss, metrics=('mse', 'mae'))
    return model
def train3D_continue(outFile='noisedata/last_model.h5' ,
                     data_dir='noisedata/data',
                     result_folder='results',
                     epochs=50,
                     lr=0.0004,
                     steps_per_epoch=128,
                     batch_size=1,
                     n_gpus=1):
    strategy = tf.distribute.MirroredStrategy()
    #看GPU是否初始化GPU
    if n_gpus > 1:
        with strategy.scope():
            model = Unet(filter_base=64, depth=3, convs_per_depth=3, kernel=(3, 3, 3), batch_norm=True, dropout=0.2,
                         pool=(2, 2, 2), residual=True, last_activation='linear', loss='mae', lr=lr)
    else:
        model = Unet(filter_base=64,depth=3,convs_per_depth=3,kernel=(3, 3, 3),batch_norm=True,dropout=0.2,
                     pool=(2, 2, 2), residual=True,last_activation='linear',loss='mae',lr=lr)

    train_data, test_data, val_data = prepare_dataseq(data_dir, batch_size)  # 本质上获得数据一个迭代器 #可以用GPT来 写个pytorch版本
    train_data = tf.data.Dataset.from_generator(train_data, output_types=(tf.float32, tf.float32))
    val_data = tf.data.Dataset.from_generator(val_data, output_types=(tf.float32, tf.float32))
    test_data = tf.data.Dataset.from_generator(test_data, output_types=(tf.float32, tf.float32))

    if n_gpus > 1:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)

    from tensorflow.keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    history = model.fit(train_data, validation_data=val_data,
                        epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=np.ceil(0.1 * steps_per_epoch),
                        callbacks=[checkpoint],verbose=1)  # 这里开始训练
    model.save(outFile)

    # 使用最佳模型进行测试
    best_model = tf.keras.models.load_model('best_model.h5',custom_objects={'custom_fmse_fsc_loss': custom_fmse_fsc_loss})
    test_loss = best_model.evaluate(test_data,  steps=np.ceil(0.1 * steps_per_epoch) ,verbose=1)
    print(f'Test loss: {test_loss}')

    return history

def predict(model_dir :str = "",ngpus :int = 1,data_dir :str=""):
    output_dir = "noisedata/zero.mrc"
    with mrcfile.open(data_dir, permissive=True) as mrcData:
        orig_data = mrcData.data.astype(np.float32)#(64,64,64)获取的尺寸
    orig_data = orig_data[np.newaxis,:,:,:]#(1, 64, 64, 64)
    print(orig_data.shape)
    if ngpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = tf.keras.models.load_model(model_dir,custom_objects={'custom_fsc_loss': custom_fsc_loss})
    else:
        model = tf.keras.models.load_model(model_dir,custom_objects={'custom_fsc_loss': custom_fsc_loss})

    output_data = model.predict(orig_data,verbose=0)#输出尺寸(1, 64, 64, 64, 1)
    print(output_data.shape)
    output_data = np.squeeze(output_data)#(64, 64, 64)#将输出尺寸的1维度全部取消掉
    print(output_data.shape)
    with mrcfile.new(output_dir, overwrite=True) as output_mrc:
        output_mrc.set_data(output_data)#输出经过模型处理的数据
    # pass
    # pass
def main():
    # mkfloder() #创建所要的文件夹
    # extract_origion_and_deconv(crop_size=64)#将提取origion 和 deconv 文件 放入noisedata 中
    # divide_train_test() # 将 origion 和deconv 文件分成训练集和测试集
    train3D_continue() # 开始模型训练
    # predict(model_dir = "E:\workspace\py_workspace\py_fold\IsoNet-last\IsoNet\\bin\\noisedata\last_model.h5", data_dir = "E:\workspace\py_workspace\py_fold\IsoNet-last\IsoNet\\bin\\noisedata\data\\test_x\TS01-wbp_000005.mrc") #预测模型效果
    pass
if __name__ == '__main__':
    main()