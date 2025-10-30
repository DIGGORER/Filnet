import logging
from IsoNet.preprocessing.prepare import get_cubes_list,get_noise_level, prepare_first_iter
from IsoNet.util.dict2attr import save_args_json,load_args_from_json
import numpy as np
import os
import sys
import shutil
from IsoNet.util.metadata import MetaData
from IsoNet.util.utils import mkfolder
from IsoNet.models.SwinUnet.train import *

def run(args):
    if args.log_level == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        #logging.basicConfig(format='%(asctime)s.%(msecs)03d, %(levelname)-8s %(message)s',
        #datefmt="%Y-%m-%d,%H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
    try:

        logging.info('\n######Isonet starts refining######\n')

        if args.continue_from is not None:
            logging.info('\n######Isonet Continues Refining######\n')
            args_continue = load_args_from_json(args.continue_from)
            for item in args_continue.__dict__:
                if args_continue.__dict__[item] is not None and (args.__dict__ is None or not hasattr(args, item)):
                    args.__dict__[item] = args_continue.__dict__[item]
        args = run_whole(args)#设置模型参数

        #environment
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        if args.log_level == 'debug':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        ### Seperate network with other modules in case we may use pytorch in the future ###
        if True:
            #查看gpu是否可用
            # check_gpu(args)
            from IsoNet.models.unet.predict import predict
            from IsoNet.models.unet.train import prepare_first_model,prepare_first_other_model ,train_data,train_other_data

        ###  find current iterations ###
        # 这一行首先检查args对象是否有一个名为iter_count的属性。就是开始时候是第几轮
        # 如果存在iter_count属性，那么将iter_count的值赋给current_iter，表示当前迭代次数。
        # 如果不存在iter_count属性，则将current_iter设为1，表示当前迭代次数为1 应为刚开始默认为1
        current_iter = args.iter_count if hasattr(args, "iter_count") else 1
        if args.continue_from is not None:
            current_iter += 1

        ###  Main Loop ###
        ###  1. find network model file ###
        ###  2. prediction if network found ###
        ###  3. prepare training data ###
        ###  4. training and save model file ###
        for num_iter in range(current_iter,args.iterations + 1): #开始从迭代1到args.iterations 默认30 1--30 current_iter是自己设置的
            logging.info("Start Iteration{}!".format(num_iter))

            ### Select a subset of subtomos, useful when the number of subtomo is too large ###
            if args.select_subtomo_number is not None:
                #这段代码是用于从给定的 args.all_mrc_list 中随机选择一定数量的元素，数量由 args.select_subtomo_number 指定。np.random.choice 是 NumPy 库中的一个函数，用于从给定的数组中进行随机选择。参数 size 指定了选择的数量，replace = False 表示选择的元素不可重复（即不放回抽样）。选出的元素存储在 args.mrc_list 中，供后续代码使用
                args.mrc_list = np.random.choice(args.all_mrc_list, size = int(args.select_subtomo_number), replace = False)
            else:
                args.mrc_list = args.all_mrc_list

            ### Update the iteration count ### #更新迭代次数
            args.iter_count = num_iter

            if args.pretrained_model is not None:#使用预训练模型
                ### use pretrained model ###
                if args.use_unet: #如果unet被使用
                    mkfolder(args.result_dir)
                if args.use_other_unet: #如果swim_unet被使用
                    mkfolder(args.other_result_dir)

                shutil.copyfile(args.pretrained_model,'{}/model_iter{:0>2d}.h5'.format(args.result_dir,num_iter-1))
                logging.info('Use Pretrained model as the output model of iteration {} and predict subtomograms'.format(num_iter-1))
                args.pretrained_model = None
                logging.info("Start predicting subtomograms!")
                predict(args) #要改
                logging.info("Done predicting subtomograms!")
            elif args.continue_from is not None:
                ### Continue from a json file ###
                logging.info('Continue from previous model: {}/model_iter{:0>2d}.h5 of iteration {} and predict subtomograms \
                '.format(args.result_dir,num_iter -1,num_iter-1))
                args.continue_from = None
                logging.info("Start predicting subtomograms!")
                predict(args) #要改
                logging.info("Done predicting subtomograms!")
            elif num_iter == 1: #初始化模型以及相关参数
                ### First iteration ### #result文件
                if args.use_unet:
                    mkfolder(args.result_dir) # 生成result文件
                if args.use_other_unet:
                    mkfolder(args.other_result_dir)

                if args.use_unet:
                    prepare_first_model(args) # unet第一次模型初始化 1.#这里的函数需要改
                if args.use_other_unet:
                    prepare_first_other_model(args) # swim_unet模型初始化
                    

                prepare_first_iter(args) # 第一次生成数据集 #这里函数不用任何操作 #这里其实也需要修改
            else:
                ### Subsequent iterations for all conditions ###
                logging.info("Start predicting subtomograms!")
                predict(args) #要改
                logging.info("Done predicting subtomograms!")
            #这句话也是非常重要每一次迭代需要上次训练状态！！！！
            if args.use_unet:#unet模型 注意init_model路径 这里得非unet 和swim_unet
                args.unet_init_model = "{}/model_iter{:0>2d}.h5".format(args.result_dir, num_iter - 1)
            if args.use_other_unet:#swimunet模型
                args.other_unet_init_model = "{}/model_iter{:0>2d}.h5".format(args.other_result_dir, num_iter - 1)

            ### Noise settings ###
            num_noise_volume = 1000
            # if True:
            if num_iter>=args.noise_start_iter[0] and (not os.path.isdir(args.noise_dir) or len(os.listdir(args.noise_dir))< num_noise_volume ):#os.path.isdir表示判断是否存在这个路径
                from IsoNet.util.noise_generator import make_noise_folder

                if args.use_unet:
                    make_noise_folder(args.noise_dir, args.noise_mode, args.cube_size, num_noise_volume,
                                      ncpus=args.preprocessing_ncpus)

            if num_iter >= args.noise_start_iter[0] and (not os.path.isdir(args.other_noise_dir) or len(os.listdir(args.other_noise_dir)) < num_noise_volume):  # os.path.isdir表示判断是否存在这个路径
                from IsoNet.util.noise_generator import make_noise_folder

                if args.use_other_unet:
                    make_noise_folder(args.other_noise_dir, args.noise_mode, args.cube_size, num_noise_volume,
                                      ncpus=args.preprocessing_ncpus)

            noise_level_series = get_noise_level(args.noise_level,args.noise_start_iter,args.iterations)#[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.05 0.05 0.05, 0.05 0.05 0.1 0.1 0.1 0.1 0.1 0.15 0.15 0.15 0.15 0.15 0.2 0.2,0.2 0.2 0.2]
            args.noise_level_current = noise_level_series[num_iter]#获取相应的迭代次数的噪音值
            logging.info("Noise Level:{}".format(args.noise_level_current))

            ### remove data_dir and generate training data in data_dir### 删除data_dir并在data_dir中生成训练数据
            try:
                if args.use_unet:
                    shutil.rmtree(args.data_dir) #删除unet目录树创建的train test
                if args.use_other_unet:
                    shutil.rmtree(args.other_data_dir) #删除其他模型目录数train test
            except OSError as e:
                print(f"Failed to remove directory: {e}")
                pass
            get_cubes_list(args) #旋转提取的数据集并做成数据集 在这个方法的子方法里面有加入噪音元素
            logging.info("Done preparing subtomograms!")

            ### remove all the mrc files in results_dir ###
            if args.remove_intermediate is True:
                logging.info("Remove intermediate files in iteration {}".format(args.iter_count-1))
                for mrc in args.mrc_list:
                    root_name = mrc.split('/')[-1].split('.')[0]
                    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(args.result_dir,root_name,args.iter_count-1)
                    os.remove(current_mrc)

            ### start training and save model and json ###
            logging.info("Start training!")
            # 以下是我要改的代码
            if args.use_unet: # unet模型 开始训练
                history = train_data(args) # train based on init model and save new one as model_iter{num_iter}.h5
                args.losses = history.history['loss']
                save_args_json(args,args.result_dir+'/refine_iter{:0>2d}.json'.format(num_iter)) # 保存训练一轮后的各个参数
            if args.use_other_unet: # swimunet模型 开始训练
                history = train_other_data(args)  # train based on init model and save new one as model_iter{num_iter}.h5
                args.losses = history.history['loss']
                save_args_json(args, args.other_result_dir + '/refine_iter{:0>2d}.json'.format(num_iter))  # 保存训练一轮后的各个参数
            # 下面不用改
            logging.info("Done training!")

            ### for last iteration predict subtomograms ###
            if num_iter == args.iterations and args.remove_intermediate == False:
                logging.info("Predicting subtomograms for last iterations")
                args.iter_count += 1
                predict(args) #要改
                args.iter_count -= 1

            logging.info("Done Iteration{}!".format(num_iter))

    except Exception:
        import traceback #traceback.format_exc() 是 Python 标准库中的一个函数，用于获取当前异常的回溯信息并返回一个字符串表示该回溯信息。在异常发生时，通常会使用这个函数来获取详细的错误信息，以便进行调试或者记录错误日志。
        error_text = traceback.format_exc()  # 捕获异常，并获取回溯信息
        f =open('log.txt','a+')
        f.write(error_text)
        f.close()
        logging.error(error_text)
        #logging.error(exc_value)


def run_whole(args):#设置各种模型参数
    '''
    Consume all the argument parameters
    '''
    #读取subtomo.star文档中参数
    md = MetaData()
    md.read(args.subtomo_star)
    #*******set fixed parameters*******
    args.crop_size = md._data[0].rlnCropSize
    args.cube_size = md._data[0].rlnCubeSize
    args.predict_cropsize = args.crop_size
    args.residual = True
    #*******calculate parameters********
    if args.gpuID is None: #如果没有设置gpuID则设置默认gpuid号可以用nvidia-smi
        args.gpuID = '0,1,2,3'
    else:
        args.gpuID = str(args.gpuID)#将其转换成字符串形式

    if args.iterations is None:#如果没设置迭代数则默认30次迭代次数
        args.iterations = 30
    args.ngpus = len(list(set(args.gpuID.split(','))))

    if args.result_dir is None:#设置输出文件夹
        args.result_dir = 'results'
    if args.other_result_dir is None:#设置输出文件夹
        args.other_result_dir = 'other_results'

    if args.data_dir is None:
        args.data_dir = args.result_dir + '/data'
    if args.other_data_dir is None:
        args.other_data_dir = args.other_result_dir + '/data'

    if args.batch_size is None:#这里可以看到作者设置了批量大小 因为如果多gpu 则每个gpu得到的数据_批量大小是 数据_批量大小/gpu_数量
        args.batch_size = max(4, 2 * args.ngpus)
    args.predict_batch_size = args.batch_size
    if args.filter_base is None:
        args.filter_base = 64
        # if md._data[0].rlnPixelSize >15:
        #     args.filter_base = 32
        # else:
        #     args.filter_base = 64
    if args.steps_per_epoch is None:
        if args.select_subtomo_number is None:#设置每一次迭代要的epoch数量
            args.steps_per_epoch = min(int(len(md) * 6/args.batch_size) , 200)
        else:
            args.steps_per_epoch = min(int(int(args.select_subtomo_number) * 6/args.batch_size) , 200)
    if args.learning_rate is None:#设置学习率
        args.learning_rate = 0.0004
    #if args.noise_level is None:
    #    args.noise_level = (0.05,0.10,0.15,0.20)
    #if args.noise_start_iter is None:
    #    args.noise_start_iter = (11,16,21,26)
    if args.noise_mode is None:
        args.noise_mode = 'noFilter'

    if args.noise_dir is None:
        args.noise_dir = args.result_dir +'/training_noise'

    if args.other_noise_dir is None:
        args.other_noise_dir = args.other_result_dir +'/training_noise'

    if args.log_level is None:
        args.log_level = "info"

    if len(md) <=0:#判断是否有数据集
        logging.error("Subtomo list is empty!")
        sys.exit(0)
    args.all_mrc_list = []
    for i,it in enumerate(md):
        if "rlnImageName" in md.getLabels():
            args.all_mrc_list.append(it.rlnImageName)
    return args

#判断时候有GPU
def check_gpu(args):
    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_str = out_str[0].decode('utf-8')
    if 'CUDA Version' not in out_str:
        raise RuntimeError('No GPU detected, Please check your CUDA version and installation')

    #import tensorflow related modules after setting environment
    import tensorflow as tf

    gpu_info =  tf.config.list_physical_devices('GPU')
    logging.debug(gpu_info)
    if len(gpu_info)!=args.ngpus:
        if len(gpu_info) == 0:
            logging.error('No GPU detected, Please check your CUDA version and installation')
            raise RuntimeError('No GPU detected, Please check your CUDA version and installation')
        else:
            logging.error('Available number of GPUs don\'t match requested GPUs \n\n Detected GPUs: {} \n\n Requested GPUs: {}'.format(gpu_info,args.gpuID))
            raise RuntimeError('Re-enter correct gpuID')
