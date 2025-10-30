import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from IsoNet.models.SwinUnet.utils import DiceLoss
from torchvision import transforms
import torch.nn.functional as F
import sys
sys.path.append('E:\workspace\py_workspace\py_fold\IsoNet-master\IsoNet\models\SwinUnet\datasets')
# from utils import test_single_volume
def trainer_result(args, model, snapshot_path):

    #from dataset_synapse import Synapse_dataset, RandomGenerator
    from IsoNet.models.SwinUnet.datasets.dataset_result import tomograms_result
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))#开始记录里面的log放在日志里
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    #从这里开始是对数据集进行操作
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    db_train = tomograms_result(base_dir=args.root_path, split="train",transform=None)
    db_test = tomograms_result(base_dir=args.root_path, split="test",transform=None)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    # 配合下面的损失函数
    # ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log') #进行log记录
    iter_num = 0 #开始迭代次数为0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    print('len(trainloader)',len(trainloader))
    print("max_epoch:",max_epoch)
    print("max_iterations:",max_iterations)
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            if args.n_gpu > 1:
                outputs = model.module(image_batch)
            else:
                outputs = model(image_batch)
            #损失函数 这里需要改 换成mae 和 mse 函数
            loss = F.mse_loss(outputs, label_batch.float())
            # loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss = 0.4 * loss_mse + 0.6 * loss_dice
            #做梯度下降操作不需要改代码
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 #学习率变化随迭代次数变化

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            #记录到日志不需要管
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            #可能用不到这段代码
            # if iter_num % 20 == 0:
            #     image = image_batch[1, 0:1, :, :]
            #     image = (image - image.min()) / (image.max() - image.min()) #图像进行归一化处理，将像素值缩放到0和1之间。通过减去图像中的最小值，然后除以最大值与最小值之间的差，可以将像素值映射到0到1的范围内。
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

        #可能也用不到这段代码
        # save_interval = 50  # int(max_epoch/6) 在epoch_num在50的时候保存模型
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        #最后一轮的时候保存模型 可能也不需要这段代码 在isonet中保存了
        # if epoch_num >= max_epoch - 1:#在epoch_num最后一轮的时候 保存模型
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break
        total_loss_mae = 0.0
        total_loss_mse = 0.0
        total_samples = 0
        model.eval()  # 切换为评估模式
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(testloader):
                # 在测试数据上进行推理
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                if args.n_gpu > 1:
                    outputs = model.module(image_batch)
                else:
                    outputs = model(image_batch)
                #损失函数 这里需要改 换成mae 和 mse 函数
                # 使用平均绝对误差（MAE）损失函数
                # print(outputs.shape,label_batch.shape)
                # print(type(outputs),type(label_batch))
                loss_mae = F.l1_loss(outputs, label_batch.float())
                # 使用均方误差（MSE）损失函数
                loss_mse = F.mse_loss(outputs, label_batch.float())
                total_loss_mae += loss_mae.item()
                total_loss_mse += loss_mse.item()
                total_samples += image_batch.size(0)
                # print('test_loss_mae:', loss_mae)
                # print('test_loss_mse:', loss_mse)
        # plot_result(dice_, hd95_, snapshot_path, args)
        # 计算整个测试集上的平均损失
        avg_loss_mae = total_loss_mae / total_samples
        avg_loss_mse = total_loss_mse / total_samples

        print('Average test_loss_mae:', avg_loss_mae)
        print('Average test_loss_mse:', avg_loss_mse)
    writer.close()
    # return "Training Finished!"
    #返回训练好的模型
    return model