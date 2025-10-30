import os 
import sys
import logging
import sys
import mrcfile
from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes,DataCubes
from IsoNet.preprocessing.img_processing import normalize
from IsoNet.preprocessing.simulate import apply_wedge1 as  apply_wedge, mw2d,other_apply_wedge1
from IsoNet.preprocessing.simulate import apply_wedge_dcube,other_apply_wedge_dcube
from multiprocessing import Pool
import numpy as np
from functools import partial
from IsoNet.util.rotations import rotation_list
# from difflib import get_close_matches
from IsoNet.util.metadata import MetaData, Item, Label
import tensorflow as tf
from tensorflow.keras.models import load_model
#Make a new folder. If exist, nenew it
# Do not set basic config for logging here
# logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)

#将deconv和mask处理后的文件进行处理
def extract_subtomos(settings):
    '''
    extract subtomo from whole tomogram based on mask
    and feed to generate_first_iter_mrc to generate xx_iter00.xx
    '''
    md = MetaData()
    md.read(settings.star_file)
    if len(md)==0:
        sys.exit("No input exists. Please check it in input folder!")
    # 读取tomograms.star相关数据集信息 列如deconv和mask处理过后的文件位置
    subtomo_md = MetaData()
    subtomo_md.addLabels('rlnSubtomoIndex','rlnImageName','rlnCubeSize','rlnCropSize','rlnPixelSize')
    count=0
    for it in md:
        if settings.tomo_idx is None or str(it.rlnIndex) in settings.tomo_idx:#setting.tomo_idx 是你要提取数据集编号 而it.rlnIndex是现在遍历的编号
            pixel_size = it.rlnPixelSize
            if settings.use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels() and os.path.isfile(it.rlnDeconvTomoName):
                logging.info("Extract from deconvolved tomogram {}".format(it.rlnDeconvTomoName))
                # 经过deconv处理过的数据集
                with mrcfile.open(it.rlnDeconvTomoName, permissive=True) as mrcData:
                    orig_data = mrcData.data.astype(np.float32)
            else:
                print("Extract from origional tomogram {}".format(it.rlnMicrographName))
                # 没有经过deconv处理过的数据集 用原始的数据集
                with mrcfile.open(it.rlnMicrographName, permissive =True) as mrcData:
                    orig_data = mrcData.data.astype(np.float32)
            #经过mask处理过的数据集
            if "rlnMaskName" in md.getLabels() and it.rlnMaskName not in [None, "None"]:
                with mrcfile.open(it.rlnMaskName, permissive=True) as m:
                    mask_data = m.data
            else: # 没有进行mask处理的数据集
                mask_data = None
                logging.info(" mask not been used for tomogram {}!".format(it.rlnIndex))
            #seed {tuple:3 0:ndarray 100 1:ndarray 100 2:ndarray 100}
            seeds=create_cube_seeds(orig_data, it.rlnNumberSubtomo, settings.crop_size,mask=mask_data) #这里注意有我的创新点
            #subtomos {ndarray:{100,80,80,80}}
            subtomos=crop_cubes(orig_data,seeds,settings.crop_size)

            # save sampled subtomo to {results_dir}/subtomos instead of subtomo_dir (as previously does)
            base_name = os.path.splitext(os.path.basename(it.rlnMicrographName))[0]
            
            for j,s in enumerate(subtomos):
                im_name = '{}/{}_{:0>6d}.mrc'.format(settings.subtomo_dir, base_name, j)
                with mrcfile.new(im_name, overwrite=True) as output_mrc:
                    count+=1
                    subtomo_it = Item()
                    subtomo_md.addItem(subtomo_it)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnSubtomoIndex'), str(count))
                    subtomo_md._setItemValue(subtomo_it,Label('rlnImageName'), im_name)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnCubeSize'),settings.cube_size)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnCropSize'),settings.crop_size)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnPixelSize'),pixel_size)
                    output_mrc.set_data(s.astype(np.float32))
    subtomo_md.write(settings.subtomo_star)

def crop_to_size(array, crop_size, cube_size):
        start = crop_size//2 - cube_size//2
        end = crop_size//2 + cube_size//2
        return array[start:end,start:end,start:end]

def get_cubes_one(data_X, data_Y, settings, start = 0, mask = None, add_noise = 0):#给添加噪音并存储
    #这里data_X维度为(64,64,64)  data_Y维度为(64,64,64)
    '''
    crop out one subtomo and missing wedge simulated one from input data,
    and save them as train set
    '''
    #data_X = apply_wedge_dcube(data, mw)
    #data_Y = crop_to_size(data, settings.crop_size, settings.cube_size)
    #data_X = crop_to_size(apply_wedge_dcube(data, mw), settings.crop_size, settings.cube_size)

    if settings.noise_level_current > 0.0000001:#这个部分开始加噪音 #噪音水平是否大于一个很小的阈值，以确定是否需要添加噪音。如果噪音水平较低，则跳过添加噪音的步骤。
        if settings.noise_dir is not None: #如果noise_dir参数不为空，则说明噪音数据存储在指定的目录中。在这种情况下，代码会读取指定目录下的噪音数据文件。
            path_noise = sorted([settings.noise_dir+'/'+f for f in os.listdir(settings.noise_dir)])#获取指定目录下的所有文件列表，并按照文件名排序。
            path_index = np.random.randint(len(path_noise)) #随机选择一个索引，以确定使用哪个噪音文件。
            def read_vol(f):
                with mrcfile.open(f, permissive=True) as mf:
                    res = mf.data
                return res
            noise_volume = read_vol(path_noise[path_index])
        else:#则说明噪音数据是通过噪音生成器产生的。在这种情况下，代码会调用IsoNet.util.noise_generator模块中的make_noise_one函数生成噪音数据。
            from IsoNet.util.noise_generator import make_noise_one
            noise_volume = make_noise_one(cubesize = settings.cube_size,mode=settings.noise_mode)
        
        #Noise along y axis is indenpedent, so that the y axis can be permutated.
        noise_volume = np.transpose(noise_volume, axes=(1,0,2)) #这一行代码对噪音体积数据进行了转置操作，将y轴和x轴进行了交换。这通常用于处理数据维度不匹配的情况。
        noise_volume = np.random.permutation(noise_volume) #这里使用 np.random.permutation 函数对噪音数据进行了随机排列，以使噪音在y轴上独立分布。这样做的目的是为了使噪音在y轴上的分布更加随机，增加数据的多样性。
        noise_volume = np.transpose(noise_volume, axes=(1,0,2)) #这一行代码再次对噪音体积数据进行了转置操作，将y轴和x轴交换回原来的位置，使数据恢复到原始的维度顺序。

        #在这里进行模型加噪
        if False:
            data_X = data_X[np.newaxis,:,:,:]
            strategy = tf.distribute.MirroredStrategy()
            if settings.ngpus > 1:
                with strategy.scope():#
                    model = load_model("E:\workspace\py_workspace\py_fold\IsoNet-last\IsoNet\bin\noisedata\last_model.h5")
            else:
                model = load_model("E:\workspace\py_workspace\py_fold\IsoNet-last\IsoNet\bin\noisedata\last_model.h5")
            if settings.noise_level_current >= 0.2:
                data_X = np.squeeze(model.predict(data_X))

        data_X += settings.noise_level_current * noise_volume / np.std(noise_volume) #噪音数据乘以噪音水平系数 settings.noise_level_current，并除以噪音数据的标准差，然后加到原始数据 data_X 上。这样做的目的是将噪音添加到原始数据中，同时保持噪音水平的一致性和稳定性。
    # 最后将训练数据存储

    with mrcfile.new('{}/train_x/x_{}.mrc'.format(settings.data_dir, start), overwrite=True) as output_mrc:
        output_mrc.set_data(data_X.astype(np.float32))#存训练输入数据
    with mrcfile.new('{}/train_y/y_{}.mrc'.format(settings.data_dir, start), overwrite=True) as output_mrc:
        output_mrc.set_data(data_Y.astype(np.float32))#存训练目标数据
    return 0

from IsoNet.models.unet import builder,builder_fullconv
from tensorflow.keras.layers import Input,Add,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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

def get_other_unet_cubes_one(data_X, data_Y, settings, start=0, mask=None, add_noise=0):  # 给添加噪音并存储
    # 这里data_X维度为(64,64,64)  data_Y维度为(64,64,64)
    '''
    crop out one subtomo and missing wedge simulated one from input data,
    and save them as train set
    '''
    # data_X = apply_wedge_dcube(data, mw)
    # data_Y = crop_to_size(data, settings.crop_size, settings.cube_size)
    # data_X = crop_to_size(apply_wedge_dcube(data, mw), settings.crop_size, settings.cube_size)
    # if True:
    if settings.noise_level_current > 0.0000001:  # 这个部分开始加噪音 #噪音水平是否大于一个很小的阈值，以确定是否需要添加噪音。如果噪音水平较低，则跳过添加噪音的步骤。
        if settings.other_noise_dir is not None:  # 如果noise_dir参数不为空，则说明噪音数据存储在指定的目录中。在这种情况下，代码会读取指定目录下的噪音数据文件。
            path_noise = sorted(
                [settings.other_noise_dir + '/' + f for f in os.listdir(settings.other_noise_dir)])  # 获取指定目录下的所有文件列表，并按照文件名排序。
            path_index = np.random.randint(len(path_noise))  # 随机选择一个索引，以确定使用哪个噪音文件。

            def read_vol(f):
                with mrcfile.open(f, permissive=True) as mf:
                    res = mf.data
                return res

            noise_volume = read_vol(path_noise[path_index])
        else:  # 则说明噪音数据是通过噪音生成器产生的。在这种情况下，代码会调用IsoNet.util.noise_generator模块中的make_noise_one函数生成噪音数据。
            from IsoNet.util.noise_generator import make_noise_one
            noise_volume = make_noise_one(cubesize=settings.cube_size, mode=settings.noise_mode)

        # Noise along y axis is indenpedent, so that the y axis can be permutated.
        noise_volume = np.transpose(noise_volume, axes=(1, 0, 2))  # 这一行代码对噪音体积数据进行了转置操作，将y轴和x轴进行了交换。这通常用于处理数据维度不匹配的情况。
        noise_volume = np.random.permutation(
            noise_volume)  # 这里使用 np.random.permutation 函数对噪音数据进行了随机排列，以使噪音在y轴上独立分布。这样做的目的是为了使噪音在y轴上的分布更加随机，增加数据的多样性。
        noise_volume = np.transpose(noise_volume, axes=(1, 0, 2))  # 这一行代码再次对噪音体积数据进行了转置操作，将y轴和x轴交换回原来的位置，使数据恢复到原始的维度顺序。

        #在这里进行模型加噪
        if settings.iter_count >= 30 and settings.iter_count <= 10:
        # if True:
            data_X = data_X[np.newaxis, :, :, :]
            strategy = tf.distribute.MirroredStrategy()
            
            if settings.ngpus > 1:
                with strategy.scope():  #
                    model = tf.keras.models.load_model(
                        "/public2/home/yuyibei/IsoNet-unet-noise/IsoNet/bin/noisedata/last_model.h5")
            else:
                model = tf.keras.models.load_model("/public2/home/yuyibei/IsoNet-unet-noise/IsoNet/bin/noisedata/last_model.h5")

            data_X = np.squeeze(model.predict(data_X))

        data_X += settings.noise_level_current * noise_volume / np.std(
            noise_volume)  # 噪音数据乘以噪音水平系数 settings.noise_level_current，并除以噪音数据的标准差，然后加到原始数据 data_X 上。这样做的目的是将噪音添加到原始数据中，同时保持噪音水平的一致性和稳定性。
    # 最后将训练数据存储

    with mrcfile.new('{}/train_x/x_{}.mrc'.format(settings.other_data_dir, start), overwrite=True) as output_mrc:
        output_mrc.set_data(data_X.astype(np.float32))  # 存训练输入数据
    with mrcfile.new('{}/train_y/y_{}.mrc'.format(settings.other_data_dir, start), overwrite=True) as output_mrc:
        output_mrc.set_data(data_Y.astype(np.float32))  # 存训练目标数据
    return 0

def get_cubes(inp,settings):
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''
    """
    "当前迭代 MRC（在 'results' 中）+ 原始 subtomo 的信息
    归一化预测 + 归一化原始 -> 归一化
    根据旋转列表旋转并输入到 get_cubes_one"
    """
    mrc, start = inp
    root_name = mrc.split('/')[-1].split('.')[0]
    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(settings.result_dir,root_name,settings.iter_count-1)

    with mrcfile.open(mrc, permissive=True) as mrcData:
        iw_data = mrcData.data.astype(np.float32)*-1
    iw_data = normalize(iw_data, percentile = settings.normalize_percentile) #对 iw_data 进行归一化处理。normalize 函数通常用于将数据缩放到特定的范围或分布，以便更好地适应模型的训练。在这里，percentile = settings.normalize_percentile 参数指定了归一化时使用的百分位数，可能是根据设置中的参数进行的。

    with mrcfile.open(current_mrc, permissive=True) as mrcData:
        ow_data = mrcData.data.astype(np.float32)*-1
    ow_data = normalize(ow_data, percentile = settings.normalize_percentile) #进行百分比归一化
    #下面这句代码是给图片增加缺失的 等价于没挖空 本质上就是将经过模型后的样本与没有经过模型后的样本进行拼接 其实就是防止预测后错误叠加。
    orig_data = apply_wedge(ow_data, ld1=0, ld2=1) + apply_wedge(iw_data, ld1 = 1, ld2=0)
    #orig_data = ow_data
    orig_data = normalize(orig_data, percentile = settings.normalize_percentile)#进行百分比归一化

    rotated_data = np.zeros((len(rotation_list), *orig_data.shape))

    old_rotation = True
    if old_rotation:#这里的话就是将搞完缺失的样本进行左右反转，上下反转等操作 一共20个
        for i,r in enumerate(rotation_list):
            data = np.rot90(orig_data, k=r[0][1], axes=r[0][0])
            data = np.rot90(data, k=r[1][1], axes=r[1][0])
            rotated_data[i] = data
    else:
        from scipy.ndimage import affine_transform
        from scipy.stats import special_ortho_group
        for i in range(len(rotation_list)): # 这段代码其实跟上边一样但是不按rotation_list里的旋转而是随机旋转 一共随机20次
            rot = special_ortho_group.rvs(3) # 每次循环中，special_ortho_group.rvs(3)会生成一个随机的3x3特殊正交矩阵。这个矩阵被用作仿射变换的旋转矩阵。
            center = (np.array(orig_data.shape) -1 )/2. #这一行计算了原始数据的中心点
            offset = center-np.dot(rot,center) # 这一行计算了仿射变换的偏移量。它通过将旋转矩阵应用于中心点来计算。
            rotated_data[i] = affine_transform(orig_data,rot,offset=offset) #最后，affine_transform函数被调用，以原始数据orig_data、旋转矩阵rot和偏移量offset作为参数，来执行仿射变换。变换后的数据被存储在名为rotated_data的数组中的第i个位置。
    
    mw = mw2d(settings.crop_size) #产生挖空图像0 1二维矩阵
    datax = apply_wedge_dcube(rotated_data, mw) #对数据进行挖空

    for i in range(len(rotation_list)):
        data_X = crop_to_size(datax[i], settings.crop_size, settings.cube_size)
        data_Y = crop_to_size(rotated_data[i], settings.crop_size, settings.cube_size)
        get_cubes_one(data_X, data_Y, settings, start = start)
        start += 1#settings.ncube


def get_other_unet_cubes(inp, settings):
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''
    """
    "当前迭代 MRC（在 'results' 中）+ 原始 subtomo 的信息
    归一化预测 + 归一化原始 -> 归一化
    根据旋转列表旋转并输入到 get_cubes_one"
    """
    mrc, start = inp
    root_name = mrc.split('/')[-1].split('.')[0]
    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(settings.other_result_dir, root_name, settings.iter_count - 1)

    with mrcfile.open(mrc, permissive=True) as mrcData:
        iw_data = mrcData.data.astype(np.float32) * -1
    iw_data = normalize(iw_data,
                        percentile=settings.normalize_percentile)  # 对 iw_data 进行归一化处理。normalize 函数通常用于将数据缩放到特定的范围或分布，以便更好地适应模型的训练。在这里，percentile = settings.normalize_percentile 参数指定了归一化时使用的百分位数，可能是根据设置中的参数进行的。

    with mrcfile.open(current_mrc, permissive=True) as mrcData:
        ow_data = mrcData.data.astype(np.float32) * -1
    ow_data = normalize(ow_data, percentile=settings.normalize_percentile)  # 进行百分比归一化
    # 下面这句代码是给图片增加缺失的 等价于没挖空 本质上就是将经过模型后的样本与没有经过模型后的样本进行拼接 其实就是防止预测后错误叠加。
    orig_data = other_apply_wedge1(ow_data, ld1=0, ld2=1) + other_apply_wedge1(iw_data, ld1=1, ld2=0)
    # orig_data = ow_data
    orig_data = normalize(orig_data, percentile=settings.normalize_percentile)  # 进行百分比归一化

    rotated_data = np.zeros((len(rotation_list), *orig_data.shape))

    old_rotation = True
    if old_rotation:  # 这里的话就是将搞完缺失的样本进行左右反转，上下反转等操作 一共20个
        for i, r in enumerate(rotation_list):
            data = np.rot90(orig_data, k=r[0][1], axes=r[0][0])
            data = np.rot90(data, k=r[1][1], axes=r[1][0])
            rotated_data[i] = data
    else:
        from scipy.ndimage import affine_transform
        from scipy.stats import special_ortho_group
        for i in range(len(rotation_list)):  # 这段代码其实跟上边一样但是不按rotation_list里的旋转而是随机旋转 一共随机20次
            rot = special_ortho_group.rvs(3)  # 每次循环中，special_ortho_group.rvs(3)会生成一个随机的3x3特殊正交矩阵。这个矩阵被用作仿射变换的旋转矩阵。
            center = (np.array(orig_data.shape) - 1) / 2.  # 这一行计算了原始数据的中心点
            offset = center - np.dot(rot, center)  # 这一行计算了仿射变换的偏移量。它通过将旋转矩阵应用于中心点来计算。
            rotated_data[i] = affine_transform(orig_data, rot,
                                               offset=offset)  # 最后，affine_transform函数被调用，以原始数据orig_data、旋转矩阵rot和偏移量offset作为参数，来执行仿射变换。变换后的数据被存储在名为rotated_data的数组中的第i个位置。

    mw = mw2d(settings.crop_size)  # 产生挖空图像0 1二维矩阵
    datax = other_apply_wedge_dcube(rotated_data, mw)  # 对数据进行挖空

    for i in range(len(rotation_list)):
        data_X = crop_to_size(datax[i], settings.crop_size, settings.cube_size)
        data_Y = crop_to_size(rotated_data[i], settings.crop_size, settings.cube_size)
        get_other_unet_cubes_one(data_X, data_Y, settings, start=start)
        start += 1  # settings.ncube

def get_cubes_list(settings):
    '''
    generate new training dataset:
    map function 'get_cubes' to mrc_list from subtomo_dir
    seperate 10% generated cubes into test set.
    '''
    '''
    生成新的训练数据集：
    将函数'get_cubes'映射到从subtomo_dir中获取的mrc_list。
    将生成的10%立方体分离成测试集
    '''
    import os
    dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    #这里要改
    if settings.use_unet:
        if not os.path.exists(settings.data_dir): # 是否存在这个setting.data_dir 这个目录其实是results/data 这个目录
            os.makedirs(settings.data_dir) # 如果没有则创建这个目录
        for d in dirs_tomake:
            folder = '{}/{}'.format(settings.data_dir, d)#开始建立results/data/train_x or train_y or test_x or test_y
            if not os.path.exists(folder):
                os.makedirs(folder)

    if settings.use_other_unet:
        if not os.path.exists(settings.other_data_dir): # 是否存在这个setting.other_data_dir 这个目录其实是other_results/data 这个目录
            os.makedirs(settings.other_data_dir) # 如果没有则创建这个目录
        for d in dirs_tomake:
            folder = '{}/{}'.format(settings.other_data_dir, d)#开始建立other_results/data/train_x or train_y or test_x or test_y
            if not os.path.exists(folder):
                os.makedirs(folder)

    inp=[]#获取我要处理的数据 这里获取的是subtomo文件下的mrc
    for i,mrc in enumerate(settings.mrc_list):
        inp.append((mrc, i*len(rotation_list)))

    # inp: list 0f (mrc_dir, index * rotation times)
    # 这段代码使用了Python中的`multiprocessing.Pool`来实现并行处理任务。`multiprocessing.Pool`允许开发者创建多个子进程，每个子进程可以并行执行指定的函数。
    # 这对于需要处理大量数据或者计算密集型任务来说是非常有用的，可以充分利用多核CPU的性能。
    # 具体来说：
    # 1.`func = partial(get_cubes, settings=settings)`：这一行代码和前面解释的一样，通过`partial`函数将`get_cubes`函数的参数`settings`
    # 固定为`settings`，生成了一个新的函数`func`。
    # 2.`with Pool(settings.preprocessing_ncpus) as p:`：这里创建了一个`Pool`对象`p`，并指定了并行处理的进程数量为
    # `settings.preprocessing_ncpus`。`settings.preprocessing_ncpus`可能是一个整数，表示同时处理的子进程数量。
    # 3.`p.map(func, inp)`：利用`p.map`方法，将函数`func`应用到迭代器`inp`中的每个元素上。这样，`inp`中的每个元素会作为参数传递给`func`
    # 函数，并且可以并行地在多个子进程中执行。在这个例子中，`func`函数是`get_cubes`，而`inp`则是传递给`get_cubes`函数的输入参数列表。
    #在这个例子中，get_cubes 是一个函数，settings 是该函数的一个参数。通过 partial(get_cubes, settings=settings)，将 get_cubes 函数的参数 settings 固定为 settings，生成了一个新的函数 func。这样，当调用 func 时，实际上就是调用了 get_cubes，并且 settings 参数已经被预先设置好了，不需要每次调用时都传递相同的 settings 参数。
    # settings.preprocessing_ncpus = 1#测试方法
    #这里开始改swim_unet 和 unet
    if settings.use_unet:
        if settings.preprocessing_ncpus > 1: #这段代码根据设置中指定的处理器核心数来决定是串行还是并行处理数据集的生成。如果处理器核心数大于 1，则使用 multiprocessing.Pool 来并行处理数据生成任务；否则，通过循环逐个处理数据生成任务
            func = partial(get_cubes, settings=settings)
            with Pool(settings.preprocessing_ncpus) as p:
                p.map(func,inp)
        else:
            for i in inp:
                logging.info("{}".format(i))
                get_cubes(i, settings)#处理数据对小块进行挖空并反转产生数据集，并且在其中加入噪音

    if settings.use_other_unet:
        if settings.preprocessing_ncpus > 1: #这段代码根据设置中指定的处理器核心数来决定是串行还是并行处理数据集的生成。如果处理器核心数大于 1，则使用 multiprocessing.Pool 来并行处理数据生成任务；否则，通过循环逐个处理数据生成任务
            func = partial(get_other_unet_cubes, settings=settings)
            with Pool(settings.preprocessing_ncpus) as p:
                p.map(func,inp)
        else:
            for i in inp:
                logging.info("{}".format(i))
                get_other_unet_cubes(i, settings)#处理数据对小块进行挖空并反转产生数据集，并且在其中加入噪音

    if settings.use_unet:
        #接下来的代码是把处理过后的图片分成训练集和测试集9：1比例
        all_path_x = os.listdir(settings.data_dir+'/train_x')
        num_test = int(len(all_path_x) * 0.1)
        num_test = num_test - num_test % settings.ngpus + settings.ngpus
        all_path_y = ['y_'+i.split('_')[1] for i in all_path_x ]
        ind = np.random.permutation(len(all_path_x))[0:num_test] # 函数生成一个随机的排列，用于选择一定数量的索引。
        for i in ind:
            os.rename('{}/train_x/{}'.format(settings.data_dir, all_path_x[i]), '{}/test_x/{}'.format(settings.data_dir, all_path_x[i]) )
            os.rename('{}/train_y/{}'.format(settings.data_dir, all_path_y[i]), '{}/test_y/{}'.format(settings.data_dir, all_path_y[i]) )

    if settings.use_other_unet:
        #接下来的代码是把处理过后的图片分成训练集和测试集9：1比例
        all_path_x = os.listdir(settings.other_data_dir+'/train_x')
        num_test = int(len(all_path_x) * 0.1)
        num_test = num_test - num_test % settings.ngpus + settings.ngpus
        all_path_y = ['y_'+i.split('_')[1] for i in all_path_x ]
        ind = np.random.permutation(len(all_path_x))[0:num_test] # 函数生成一个随机的排列，用于选择一定数量的索引。
        for i in ind:
            os.rename('{}/train_x/{}'.format(settings.other_data_dir, all_path_x[i]), '{}/test_x/{}'.format(settings.other_data_dir, all_path_x[i]) )
            os.rename('{}/train_y/{}'.format(settings.other_data_dir, all_path_y[i]), '{}/test_y/{}'.format(settings.other_data_dir, all_path_y[i]) )
#这行代码是创建一个迭代次数长度的list并且加入相应的噪声
# [0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.05 0.05 0.05, 0.05 0.05 0.1  0.1  0.1  0.1  0.1  0.15 0.15 0.15 0.15 0.15 0.2  0.2, 0.2  0.2  0.2 ]
def get_noise_level(noise_level_tuple,noise_start_iter_tuple,iterations):
    assert len(noise_level_tuple) == len(noise_start_iter_tuple) and type(noise_level_tuple) in [tuple,list] #用于检查 noise_level_tuple 和 noise_start_iter_tuple 的长度是否相等，并且 noise_level_tuple 的类型是元组（tuple）或列表（list）。如果条件为假，将会引发 AssertionError。
    noise_level = np.zeros(iterations+1)
    for i in range(len(noise_start_iter_tuple)-1):
        #remove this assert because it may not be necessary, and cause problem when iterations <3
        #assert i < iterations and noise_start_iter_tuple[i] < noise_start_iter_tuple[i+1]
        noise_level[noise_start_iter_tuple[i]:noise_start_iter_tuple[i+1]] = noise_level_tuple[i]
    assert noise_level_tuple[-1] < iterations 
    noise_level[noise_start_iter_tuple[-1]:] = noise_level_tuple[-1]
    return noise_level

def generate_first_iter_mrc(mrc,settings):
    '''
    Apply mw to the mrc and save as xx_iter00.xx
    '''
    root_name = mrc.split('/')[-1].split('.')[0]
    extension = mrc.split('/')[-1].split('.')[1]
    with mrcfile.open(mrc, permissive=True) as mrcData:
        orig_data = normalize(mrcData.data.astype(np.float32)*-1, percentile = settings.normalize_percentile)
    orig_data = apply_wedge(orig_data, ld1=1, ld2=0)
    orig_data = normalize(orig_data, percentile = settings.normalize_percentile)

    if settings.use_unet:
        with mrcfile.new('{}/{}_iter00.{}'.format(settings.result_dir,root_name, extension), overwrite=True) as output_mrc:
            output_mrc.set_data(-orig_data)#这里其实可以看到iter是迭代次数，是经过模型后的
    if settings.use_other_unet:
        with mrcfile.new('{}/{}_iter00.{}'.format(settings.other_result_dir,root_name, extension), overwrite=True) as output_mrc:
            output_mrc.set_data(-orig_data)
    #preparation files for the first iteration
def prepare_first_iter(settings):
    # settings.preprocessing_ncpus = 1 #测试专用
    if settings.preprocessing_ncpus >1:
        with Pool(settings.preprocessing_ncpus) as p:
            func = partial(generate_first_iter_mrc, settings=settings)
            p.map(func, settings.mrc_list)
    else:
        for i in settings.mrc_list:
            generate_first_iter_mrc(i,settings)
    return settings

