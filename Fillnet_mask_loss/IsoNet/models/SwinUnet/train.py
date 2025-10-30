import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from IsoNet.models.SwinUnet.networks.vision_transformer import SwinUnet as ViT_seg
from IsoNet.models.SwinUnet.trainer import trainer_result
from IsoNet.models.SwinUnet.config import get_config
from torchsummary import summary
import torch.nn as nn
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='E:\workspace\py_workspace\py_fold\IsoNet-swimunet\IsoNet\\bin\other_results\data', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Result', help='experiment_name') #default = 'Synapse'
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, #输入时候通道数量
                    default=64, help='output channel of network')
parser.add_argument('--output_dir',default='./output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10, help='maximum epoch number to train')#默认为150
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu') #设置batch_ssize 大小
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu') #Gpu数量
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, #学习率
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, #输入时候图片大小64x64
                    default=64, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
# 这行的require = True去掉可以不用输入参数
parser.add_argument('--cfg', type=str, default='/public/home/yuyibei2023/IsoNet-unet-noise/IsoNet/models/SwinUnet/configs/swin_tiny_patch4_window7_224_lite.yaml' , metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
args = parser.parse_args()
# if args.dataset == "Synapse":
#     args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)
def Swim_Unet():
    net = ViT_seg(None, img_size=args.img_size, num_classes=args.num_classes) # 构建模型
    return net

def prepare_first_SwimUnet_model(settings):
    model = Swim_Unet()
    if args.n_gpu >1:
        model = nn.DataParallel(model).cuda()  # 必须先将模型包装为DataParallel
    else:
        model = model.cuda()
    init_model_name = settings.other_result_dir+'/model_iter00.pth'
    # save_mode_path = os.path.join('E:\workspace\py_workspace\py_fold\IsoNet-master\IsoNet\models\SwinUnet\output_dir', 'epoch_' + str(0) + '.pth')
    torch.save(model.state_dict(), init_model_name)

def tran3D_SwimUnet(settings):
    outfile = '{}/model_iter{:0>2d}.pth'.format(settings.other_result_dir,settings.iter_count)
    net = Swim_Unet()
    if args.n_gpu >1:
        net = nn.DataParallel(net).cuda()  # 必须先将模型包装为DataParallel
        state_dict = torch.load(settings.swim_unet_init_model)
        net.state_dict = state_dict
        net = trainer_result(args, net, args.output_dir) # 像字典一样调用函数设置参数 #开始训练 将结果保存
        torch.save(net.state_dict, outfile) # 训练完成进行保存
    else:
        net = net.cuda()
        state_dict = torch.load(settings.swim_unet_init_model)
        net.load_state_dict(state_dict)#加载模型
        net = trainer_result(args, net, args.output_dir) # 像字典一样调用函数设置参数 #开始训练 将结果保存
        torch.save(net.state_dict(), outfile) # 训练完成进行保存

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    # dataset_config = {
    #     'Synapse': {
    #         'root_path': args.root_path,
    #         'list_dir': './lists/lists_Synapse',
    #         'num_classes': 64, #这里要改输出的维度第三位维
    #     },
    # }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    # args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.root_path = dataset_config[dataset_name]['root_path']
    # args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(None, img_size=None, num_classes=None).cuda() # 构建模型
    net.load_from(config) # 是否有预训练模型 加载预训练模型
    # input_data = torch.randn(1, 64, 64, 64).cuda() # 1个样本，64个通道，64x64尺寸 # 测试用的
    # print("输入样本的维度:",input_data.shape)
    # print("模型输出数据样式:",net(input_data).shape) #测试用的
    # summary(net, (64, args.img_size, args.img_size)) #查看模型中各个参数
    trainer_result(args, net, args.output_dir) #像字典一样调用函数设置参数