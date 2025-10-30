import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import load_model
import mrcfile
from IsoNet.preprocessing.img_processing import normalize
import numpy as np
import tensorflow.keras.backend as K
import os
from IsoNet.util.toTile import reform3D
from tqdm import tqdm
import sys
from IsoNet.models.SwinUnet.train import Swim_Unet
import torch
from IsoNet.preprocessing.img_processing import change_size,combine_cude,get_turn_notation_list,recover_turn_notation_list
import torch.nn as nn
def swimunet_predict(settings):
    #这里代码得改
    # model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1))
    if settings.use_swim_unet:
        model = Swim_Unet()
        # print('{}/model_iter{:0>2d}.pth'.format(settings.result_dir,settings.iter_count-1))
        if settings.ngpus > 1:
            model = nn.DataParallel(model).cuda()  # 必须先将模型包装为DataParallel
            state_dict = torch.load('{}/model_iter{:0>2d}.pth'.format(settings.other_result_dir,settings.iter_count-1))
            model.state_dict = state_dict
        else:
            model = model.cuda()
            state_dict = torch.load('{}/model_iter{:0>2d}.pth'.format(settings.other_result_dir,settings.iter_count-1))
            model.load_state_dict(state_dict)  # 加载模型

    N = settings.predict_batch_size
    num_batches = len(settings.mrc_list)
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = []
    for i,mrc in enumerate(list(settings.mrc_list) + list(settings.mrc_list[:append_number])):
        root_name = mrc.split('/')[-1].split('.')[0]
        with mrcfile.open(mrc, permissive = True) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
        real_data=normalize(real_data, percentile = settings.normalize_percentile)

        cube_size = real_data.shape[0]
        pad_size1 = (settings.predict_cropsize - cube_size)//2
        pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 != 0 else pad_size1
        padi = (pad_size1,pad_size2)
        real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')

        if (i+1)%N != 0:
            data.append(real_data)
        else:
            data.append(real_data)
            data = np.array(data)

            if settings.use_swim_unet:#swim_unet模型 #这里需要把(80,80,80)分成4个(64,64,64)然后拼接成(80,80,80)
                predicted_last_list = []
                for data_seg in data: #data的里面数量
                    # combined_data = np.zeros((80, 80, 80))#每次都要清零
                    # data_seg_1 = data_seg[:64,:64,:64]
                    # data_seg_2 = data_seg[16:80,:64,:64]
                    # data_seg_3 = data_seg[:64,16:80,:64]
                    # data_seg_4 = data_seg[16:80,16:80,:64]
                    # data_seg_5 = data_seg[:64,:64,16:80]
                    # data_seg_6 = data_seg[16:80,:64,16:80]
                    # data_seg_7 = data_seg[:64,16:80,16:80]
                    # data_seg_8 = data_seg[16:80,16:80,16:80]
                    # data_seglist = [data_seg_1[np.newaxis,:,:,:],data_seg_2[np.newaxis,:,:,:],data_seg_3[np.newaxis,:,:,:],data_seg_4[np.newaxis,:,:,:],
                    #                 data_seg_5[np.newaxis,:,:,:],data_seg_6[np.newaxis,:,:,:],data_seg_7[np.newaxis,:,:,:],data_seg_8[np.newaxis,:,:,:]]
                    data_seglist = change_size(cude_size=80,small_cude_size=64,data_seg=data_seg)
                    predicted_list =[]
                    for data_seglist_value in data_seglist:
                        model.eval()  # 设置模型为评估模式，这会关闭 dropout 和 batch normalization
                        with torch.no_grad():  # 禁用梯度计算，加快推断速度
                            predicted = model(torch.tensor(data_seglist_value).cuda()) #将截断的数组通过模型进行复原
                            predicted_list.append(np.squeeze(predicted).cpu()) #将维度从[1,64,64,64]变成[64,64,64]
                    # combined_data[:64,:64,:64] += np.array(predicted_list[0])
                    # combined_data[16:80,:64,:64] += np.array(predicted_list[1])
                    # combined_data[:64,16:80,:64] += np.array(predicted_list[2])
                    # combined_data[16:80,16:80,:64] += np.array(predicted_list[3])
                    # combined_data[:64,:64,16:80] += np.array(predicted_list[4])
                    # combined_data[16:80,:64,16:80] += np.array(predicted_list[5])
                    # combined_data[:64,16:80,16:80] += np.array(predicted_list[6])
                    # combined_data[16:80,16:80,16:80] += np.array(predicted_list[7])
                    #
                    # #下个阶段将这个[80,80,80] 分别除以相应的平均值
                    # combined_data[16:64,16:64,16:64] /= 8
                    #
                    # combined_data[16:64,16:64,:16] /=4
                    # combined_data[16:64,16:64,64:80] /=4
                    # combined_data[16:64,:16,16:64] /=4
                    # combined_data[16:64,64:80,16:64] /=4
                    # combined_data[:16,16:64,16:64] /=4
                    # combined_data[64:80,16:64,16:64] /=4
                    #
                    # combined_data[16:64,:16,:16] /= 2
                    # combined_data[16:64,64:80,:16] /= 2
                    # combined_data[16:64,:16,64:80] /= 2
                    # combined_data[16:64,64:80,64:80] /= 2
                    #
                    # combined_data[:16,16:64,:16] /= 2
                    # combined_data[:16,16:64,64:80] /= 2
                    # combined_data[64:80,16:64,:16] /= 2
                    # combined_data[64:80,16:64,64:80] /= 2
                    #
                    # combined_data[:16,:16,16:64] /= 2
                    # combined_data[:16,64:80,16:64] /= 2
                    # combined_data[64:80,:16,16:64] /= 2
                    # combined_data[64:80,64:80,16:64] /= 2
                    combined_data = combine_cude(80,64,predicted_list)
                    predicted_last_list.append(combined_data)

            if settings.use_swim_unet: #swimunt 模型
                predicted = predicted_last_list

            for j,outData in enumerate(predicted):
                count = i + j - N + 1
                if count < len(settings.mrc_list):
                    m_name = settings.mrc_list[count]

                    root_name = m_name.split('/')[-1].split('.')[0]
                    end_size = pad_size1+cube_size
                    outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
                    outData1 = normalize(outData1, percentile = settings.normalize_percentile)
                    with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(settings.other_result_dir,root_name,settings.iter_count-1), overwrite=True) as output_mrc:
                        output_mrc.set_data(-outData1)
            data = []
    K.clear_session()

def other_unet_predict(settings):
    #加载自定义损失函数
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
        #SSIM_loss
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
        
    #这里代码得改
    # model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1))
    if settings.use_other_unet:#unet模型
        strategy = tf.distribute.MirroredStrategy()
        if settings.ngpus >1:
            with strategy.scope():
                model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.other_result_dir,settings.iter_count-1),custom_objects={'custom_ssim_loss': custom_ssim_loss})
        else:
            model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.other_result_dir,settings.iter_count-1),custom_objects={'custom_ssim_loss': custom_ssim_loss})

    N = settings.predict_batch_size
    num_batches = len(settings.mrc_list)
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = []
    for i,mrc in enumerate(list(settings.mrc_list) + list(settings.mrc_list[:append_number])):
        root_name = mrc.split('/')[-1].split('.')[0]
        with mrcfile.open(mrc, permissive = True) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
        real_data=normalize(real_data, percentile = settings.normalize_percentile)

        cube_size = real_data.shape[0]
        pad_size1 = (settings.predict_cropsize - cube_size)//2
        pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 != 0 else pad_size1
        padi = (pad_size1,pad_size2)
        real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')

        if (i+1)%N != 0:
            data.append(real_data)
        else:
            data.append(real_data)
            data = np.array(data)

            if settings.use_other_unet:#unet模型直接输入 不需要进行切片 因为这个模型可以输入任何像这样的维度如[1,None,None,None]维度的参数
                predicted=model.predict(data[:,:,:,:,np.newaxis], batch_size= settings.predict_batch_size,verbose=0)

            if settings.use_other_unet:#unet模型
                predicted = predicted.reshape(predicted.shape[0:-1])

            for j,outData in enumerate(predicted):
                count = i + j - N + 1
                if count < len(settings.mrc_list):
                    m_name = settings.mrc_list[count]

                    root_name = m_name.split('/')[-1].split('.')[0]
                    end_size = pad_size1+cube_size
                    outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
                    outData1 = normalize(outData1, percentile = settings.normalize_percentile)
                    with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(settings.other_result_dir,root_name,settings.iter_count-1), overwrite=True) as output_mrc:
                        output_mrc.set_data(-outData1)
            data = []
    K.clear_session()
def unet_predict(settings):
    #这里代码得改
    # model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1))
    if settings.use_unet:#unet模型
        strategy = tf.distribute.MirroredStrategy()
        if settings.ngpus >1:
            with strategy.scope():
                model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))
        else:
            model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))

    N = settings.predict_batch_size
    num_batches = len(settings.mrc_list)
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = []
    for i,mrc in enumerate(list(settings.mrc_list) + list(settings.mrc_list[:append_number])):
        root_name = mrc.split('/')[-1].split('.')[0]
        with mrcfile.open(mrc, permissive = True) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
        real_data=normalize(real_data, percentile = settings.normalize_percentile)

        cube_size = real_data.shape[0]
        pad_size1 = (settings.predict_cropsize - cube_size)//2
        pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 != 0 else pad_size1
        padi = (pad_size1,pad_size2)
        real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')

        if (i+1)%N != 0:
            data.append(real_data)
        else:
            data.append(real_data)
            data = np.array(data)

            if settings.use_unet:#unet模型直接输入 不需要进行切片 因为这个模型可以输入任何像这样的维度如[1,None,None,None]维度的参数
                predicted=model.predict(data[:,:,:,:,np.newaxis], batch_size= settings.predict_batch_size,verbose=0)

            if settings.use_unet:#unet模型
                predicted = predicted.reshape(predicted.shape[0:-1])

            for j,outData in enumerate(predicted):
                count = i + j - N + 1
                if count < len(settings.mrc_list):
                    m_name = settings.mrc_list[count]

                    root_name = m_name.split('/')[-1].split('.')[0]
                    end_size = pad_size1+cube_size
                    outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
                    outData1 = normalize(outData1, percentile = settings.normalize_percentile)
                    with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(settings.result_dir,root_name,settings.iter_count-1), overwrite=True) as output_mrc:
                        output_mrc.set_data(-outData1)
            data = []
    K.clear_session()

def predict(settings):
    if settings.use_unet:
        unet_predict(settings)#使用unet模型
    if settings.use_other_unet:
        other_unet_predict(settings)#使用swim_unet模型

#对数据进行补全预测
def predict_one(args,one_tomo,output_file=None):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc
    import logging
    from IsoNet.models.SwinUnet.train import Swim_Unet
    from IsoNet.preprocessing.img_processing import change_size,combine_cude
    import torch
    if True:#unet模型
        if args.ngpus > 1:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = tf.keras.models.load_model(args.model)
        else:
            model = tf.keras.models.load_model(args.model)
    else: #swimunet模型
        model = Swim_Unet()
        # print('{}/model_iter{:0>2d}.pth'.format(settings.result_dir,settings.iter_count-1))
        if args.ngpus > 1:
            model = nn.DataParallel(model).cuda()  # 必须先将模型包装为DataParallel
            state_dict = torch.load(args.model)
            model.state_dict = state_dict
        else:
            model = model.cuda()
            state_dict = torch.load(args.model)
            model.load_state_dict(state_dict)  # 加载模型

    logging.info("Loaded model from disk")
    root_name = one_tomo.split('/')[-1].split('.')[0]

    if output_file is None:
        if os.path.isdir(args.output_file):
            output_file = args.output_file+'/'+root_name+'_corrected.mrc'
        else:
            output_file = root_name+'_corrected.mrc'

    logging.info('predicting:{}'.format(root_name))

    with mrcfile.open(one_tomo,permissive=True) as mrcData:
        real_data = mrcData.data.astype(np.float32)*-1
        voxelsize = mrcData.voxel_size
    real_data = normalize(real_data,percentile=args.normalize_percentile) #对图片进行百分比归一化
    data=np.expand_dims(real_data,axis=-1) #给数据增加维度
    reform_ins = reform3D(data)
    data = reform_ins.pad_and_crop_new(args.cube_size,args.crop_size)
    #to_predict_data_shape:(n,cropsize,cropsize,cropsize,1)
    #imposing wedge to every cubes
    #data=wedge_imposing(data)

    N = args.batch_size #* args.ngpus * 4 # 8*4*8
    num_patches = data.shape[0]
    if num_patches%N == 0: #如果这里要复原的样本能被batch_size整除则不做任何事 否则将样本前面的数据再次加入样本中扩大样本使其能被batch_size整除
        append_number = 0
    else:
        append_number = N - num_patches%N
    data = np.append(data, data[0:append_number], axis = 0)#从自己原来的前面部分进行添加到自己后面
    num_big_batch = data.shape[0]//N#数据/批量大小
    outData = np.zeros(data.shape)#经过模型输出输出的数据
    logging.info("total batches: {}".format(num_big_batch))
    for i in tqdm(range(num_big_batch), file=sys.stdout):
        in_data = data[i*N:(i+1)*N]#进行取数据
        # in_data_gen = get_gen_single(in_data,args.batch_size,shuffle=False)
        # in_data_tf = tf.data.Dataset.from_generator(in_data_gen,output_types=tf.float32)
        # 这个部分我需要自己写一个函数来改变
        if True:#unet模型
            outData[i*N:(i+1)*N] = model.predict(in_data,verbose=0) #将预测数据添加到输出数据上

        elif False:#swimunet模型 没有翻转
            predicted_last_list = []
            for temp_data in in_data:
                temp_data = temp_data.reshape(temp_data.shape[0],temp_data.shape[1],temp_data.shape[2])
                data_seglist = change_size(96,64,temp_data)
                predicted_list = []
                for data_seglist_value in data_seglist:
                    model.eval()  # 设置模型为评估模式，这会关闭 dropout 和 batch normalization
                    with torch.no_grad():  # 禁用梯度计算，加快推断速度
                        predicted = model(torch.tensor(data_seglist_value).cuda())  # 将截断的数组通过模型进行复原
                        predicted_list.append(np.squeeze(predicted).cpu())  # 将维度从[1,96,96,96]变成[96,96,96]
                combined_data = combine_cude(96, 64, predicted_list)[:,:,:,np.newaxis]
                predicted_last_list.append(combined_data)
            outData[i * N:(i + 1) * N] = predicted_last_list

        else:#swimunet模型 并且有24块翻转
            predicted_last_list = []
            for temp_data in in_data:
                temp_data = temp_data.reshape(temp_data.shape[0],temp_data.shape[1],temp_data.shape[2])
                data_seglist = change_size(96,64,temp_data)
                predicted_list = []
                for data_seglist_value in data_seglist:
                    turn_notation_list = get_turn_notation_list(data_seglist_value) #把每个小块翻转一共24种
                    turn_notation_list = np.squeeze(turn_notation_list)
                    model.eval()  # 设置模型为评估模式，这会关闭 dropout 和 batch normalization
                    with torch.no_grad():  # 禁用梯度计算，加快推断速度
                        predicted = model(torch.tensor(turn_notation_list).cuda())  # 将截断的数组通过模型进行复原
                        recover_list = recover_turn_notation_list(predicted.cpu()) #将每个小块的24种 翻转到同一面
                        cube_zero = np.zeros([64,64,64])
                        for pre_cube in recover_list:
                            cube_zero += pre_cube
                        predicted = cube_zero/24 #将24种全部相加再除以24
                        predicted_list.append(np.squeeze(predicted))  # 将维度从[1,96,96,96]变成[96,96,96]
                combined_data = combine_cude(96, 64, predicted_list)[:,:,:,np.newaxis] #合成
                predicted_last_list.append(combined_data)
            outData[i * N:(i + 1) * N] = predicted_last_list


    outData = outData[0:num_patches]#这里只取这个数据刚开始的data.shape[0] 因为有补全data = np.append(data, data[0:append_number], axis = 0)在这行代码上但补全只是为了适应模型并不需要

    outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), args.cube_size, args.crop_size) #把小切片3维图像复原成完整的大的3维图像

    outData = normalize(outData,percentile=args.normalize_percentile)#最后再次百分比归一化图像
    with mrcfile.new(output_file, overwrite=True) as output_mrc:#存储经过模型的图像
        output_mrc.set_data(-outData)
        output_mrc.voxel_size = voxelsize
    K.clear_session()
    logging.info('Done predicting')
    # predict(args.model,args.weight,args.mrc_file,args.output_file, cubesize=args.cubesize, cropsize=args.cropsize, batch_size=args.batch_size, gpuID=args.gpuID, if_percentile=if_percentile)
