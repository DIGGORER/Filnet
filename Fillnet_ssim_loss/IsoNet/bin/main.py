import sys
sys.path.append('/public/home/yuyibei2023/IsoNet-unet-noise-modify_loss9_absplus_hiv')
from IsoNet.bin.isonet import ISONET
import mrcfile
import numpy as np
import scipy
import os

#
# def modify_photo(matrix):
#     matrix =  matrix - matrix.min()
#     print(matrix.min())
#     matrix = (matrix / matrix.max()) * 255
#     matrix = np.uint8(matrix)
#     return matrix
# def showphoto(vol_file):
#     with mrcfile.open(vol_file, permissive=True) as f:
#         header_in = f.header
#         vol = f.data
#         voxelsize = f.voxel_size
#     # 生成一个二维矩阵（示例）
#     import numpy as np
#     matrix = vol[81]#.astype(np.uint8)
#     matrix = modify_photo(matrix)
#     print(matrix)
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl
#     import cv2
#     # 显示灰度图像
#     cv2.imshow('Gray Image', matrix)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def show_photo():
#     import cv2
#     img = cv2.imread("E:\workspace\py_workspace\py_fold\IsoNet-master\IsoNet\\bin\lhl.jpg")
#     img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(img1)
#     img = deconv_image(img1,judge_3D=False)
#     img = modify_photo(img)
#     print('图片2',img)
#     cv2.imshow("image", img1)  # 显示图片，后面会讲解
#     cv2.waitKey(0)  # 等待按键
#     cv2.imshow("image", img)  # 显示图片，后面会讲解
#     cv2.waitKey(0)  # 等待按键

# def show_maskphoto():
#     import cv2
#     img1 = cv2.imread("E:\workspace\py_workspace\py_fold\IsoNet-master\IsoNet\\bin\\1.jpeg")
#     # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     print(img1)
#     img = mask_image(img1)
#     print(img)
#     # img = modify_photo(img)
#     print('图片2',img)
#     cv2.imshow("image", img1)  # 显示图片，后面会讲解
#     cv2.waitKey(0)  # 等待按键
#     cv2.imshow("image", img)  # 显示图片，后面会讲解
#     cv2.waitKey(0)  # 等待按键
# def tom_ctf1d(pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048):
#
#     ny = 1 / pixelsize
#
#
#     lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
#     lambda2 = lambda1 * 2
#
#
#     points = np.arange(0,length)
#     points = points.astype(float)
#     points = points/(2 * length)*ny
#
#     k2 = points**2
#     term1 = lambda1**3 * cs * k2**2
#
#     w = np.pi / 2 * (term1 + lambda2 * defocus * k2) - phaseshift
#
#     acurve = np.cos(w) * amplitude
#     pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
#     bfactor = np.exp(-bfactor * k2 * 0.25)
#
#     return (pcurve + acurve)*bfactor

# def deconv_image(image,highpassnyquist=0.02,snrfalloff=1.0,angpix=10.0,deconvstrength=1.0,voltage=300,cs=2.7,defocus=0.0,phaseshift=0,phaseflipped=False,ncpu=8,judge_3D=True):
#     # 生成频域数据
#     data = np.arange(0,1+1/2047.,1/2047.) #1维 [2048]
#     highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi;#1维 [2048]
#     highpass = 1-np.cos(highpass);#1维 [2048]
#     eps = 1e-6 #常数
#     snr = np.exp(-data * snrfalloff * 100 / angpix) * (10**deconvstrength) * highpass + eps #常数
#     # snr[0] = -1
#     # 计算CTF（对比度传递函数）
#     ctf = tom_ctf1d(angpix*1e-10, voltage * 1e3, cs * 1e-3, -defocus*1e-6, 0.07, phaseshift / 180 * np.pi, 0); #1维
#     if phaseflipped:
#         ctf = abs(ctf)
#
#     # 计算CTF（对比度传递函数）
#     wiener = ctf/(ctf*ctf+1/snr);#1维 2048
#
#     denom = ctf*ctf+1/snr #1维 2048
#     #np.savetxt('den.txt',denom)
#     #np.savetxt('snr.txt',snr)
#     #np.savetxt('hipass.txt',highpass)
#     #np.savetxt('ctf.txt',ctf)
#     #np.savetxt('wiener.txt',wiener, fmt='%f')
#
#     # 准备体积的坐标
#     s1 = - int(np.shape(image)[1] / 2) #常数
#     f1 = s1 + np.shape(image)[1] - 1 #常数
#     m1 = np.arange(s1,f1+1) #1维
#
#     s2 = - int(np.shape(image)[0] / 2) #常数
#     f2 = s2 + np.shape(image)[0] - 1 #常数
#     m2 = np.arange(s2,f2+1) #1维
#     s3=None
#     f3=None
#     m3=None
#     if judge_3D:
#         s3 = - int(np.shape(image)[2] / 2)
#         f3 = s3 + np.shape(image)[2] - 1
#         m3 = np.arange(s3,f3+1) #1维
#
#     #s3 = -floor(size(vol,3)/2);
#     #f3 = s3 + size(vol,3) - 1;
#     # 生成坐标的网格
#     if judge_3D:
#         x, y, z = np.meshgrid(m1,m2,m3)
#         x = x.astype(np.float32) / np.abs(s1)
#         y = y.astype(np.float32) / np.abs(s2)
#         z = z.astype(np.float32) / np.maximum(1, np.abs(s3))
#         #z = z.astype(float) / np.abs(s3);
#         r = np.sqrt(x**2+y**2+z**2)
#         r = np.minimum(1, r)
#         r = np.fft.ifftshift(r)#傅里叶变换
#     else:
#         x, y = np.meshgrid(m1,m2)
#         x = x.astype(np.float32) / np.abs(s1)
#         y = y.astype(np.float32) / np.abs(s2)
#         #z = z.astype(float) / np.abs(s3);
#         r = np.sqrt(x**2+y**2)
#         r = np.minimum(1, r)
#         r = np.fft.ifftshift(r)#傅里叶变换
#
#     #x = 0:1/2047:1;
#     ramp = np.interp(r, data,wiener).astype(np.float32) #一维线性插值
#     # 插值维纳滤波器值
#     #ramp = np.interp(data,wiener,r);
#     # 执行去卷积
#     deconv = np.real(scipy.fft.ifftn(scipy.fft.fftn(image, overwrite_x=True, workers=ncpu) * ramp, overwrite_x=True, workers=ncpu))
#     deconv = deconv.astype(np.float32)
#     # 归一化去卷积后的体积
#     std_deconv = np.std(deconv)
#     std_vol = np.std(image)
#     ave_vol = np.average(image)
#     # deconv = deconv/std_deconv* std_vol + ave_vol
#     deconv /= std_deconv
#     deconv *= std_vol
#     deconv += ave_vol
#     # 保存去卷积后的体积
#     return deconv
# def mask_image(image,side=9,density_percentage=90., std_percentage=40.,mask_boundary = None,surface=None):
#     from IsoNet.util.filter import maxmask,stdmask
#     from scipy.ndimage import gaussian_filter
#     from skimage.transform import resize
#     sp=np.array(image.shape) #1维 [200 464 480]
#     sp2 = sp//2 #1维 [100 232 240]
#     image = resize(image, sp2, anti_aliasing=True)
#     gauss = gaussian_filter(image, side / 2)
#     if density_percentage <= 99.8:
#         mask1 = maxmask(gauss,side=side, percentile=density_percentage)
#     else:
#         mask1 = np.ones(sp2)
#
#     if std_percentage <= 99.8:
#         mask2 = stdmask(gauss,side=side, threshold=std_percentage)
#     else:
#         mask2 = np.ones(sp2)
#     print(mask1.shape,mask2.shape)
#     out_mask_bin = np.multiply(mask1, mask2)
#
#     if mask_boundary is not None:
#         from IsoNet.util.filter import boundary_mask
#         mask3 = boundary_mask(image, mask_boundary)
#         out_mask_bin = np.multiply(out_mask_bin, mask3)
#
#     if (surface is not None) and surface < 1:
#         for i in range(int(surface * sp2[0])):
#             out_mask_bin[i] = 0
#         for i in range(int((1 - surface) * sp2[0]), sp2[0]):
#             out_mask_bin[i] = 0
#     print(out_mask_bin.shape)
#     # 这种操作通常用于数据的插值或者重采样，通过填充缺失值或者对数据进行平滑处理，以便进一步的处理或分析。
#     out_mask = np.zeros(sp)
#     out_mask[0:-1:2, 0:-1:2, 0:-1:2] = out_mask_bin
#     out_mask[0:-1:2, 0:-1:2, 1::2] = out_mask_bin
#     out_mask[0:-1:2, 1::2, 0:-1:2] = out_mask_bin
#     out_mask[0:-1:2, 1::2, 1::2] = out_mask_bin
#     out_mask[1::2, 0:-1:2, 0:-1:2] = out_mask_bin
#     out_mask[1::2, 0:-1:2, 1::2] = out_mask_bin
#     out_mask[1::2, 1::2, 0:-1:2] = out_mask_bin
#     out_mask[1::2, 1::2, 1::2] = out_mask_bin
#     out_mask = (out_mask > 0.5).astype(np.uint8)
#     return out_mask
def main():
    # show_maskphoto()
    # show_photo()
    # vol_file = "E:\\workspace\\py_workspace\\py_fold\\IsoNet-master\\data_isonet\\HIV\\demo_data\\tomograms\\TS45-wbp.rec"
    # showphoto(vol_file)
    # ISONET.prepare_star(None,'E:\\workspace\\py_workspace\\py_fold\\IsoNet-master\\data_isonet\\neuron')
    # ISONET.prepare_star(None,'E:\\workspace\\py_workspace\\py_fold\\IsoNet-master\\data_isonet\\neuron')
    # ISONET.prepare_star(None,'/public/home/yuyibei2023/IsoNet-unet-noise-modify_loss9_absplus_hiv/data_isonet/HIV/demo_data/tomograms') #生成tomograms.star
    # ISONET.deconv(None,star_file='/public/home/yuyibei2023/IsoNet-unet-noise-modify_loss9_absplus_hiv/IsoNet/bin/tomograms.star',snrfalloff=0.7,deconvstrength=1) # 生成deconv文件夹
    # ISONET.make_mask(None,star_file='/public/home/yuyibei2023/IsoNet-unet-noise-modify_loss9_absplus_hiv/IsoNet/bin/tomograms.star') # 生成 mask文件夹
    # ISONET.extract(None,star_file='/public/home/yuyibei2023/IsoNet-unet-noise-modify_loss9_absplus_hiv/IsoNet/bin/tomograms.star',crop_size=96) # 生成sybtomo文件夹
    # from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes,DataCubes
    # from IsoNet.preprocessing.img_processing import normalize
    # from IsoNet.preprocessing.simulate import apply_wedge1 as  apply_wedge, mw2d
    # from IsoNet.preprocessing.simulate import apply_wedge_dcube
    # from multiprocessing import Pool
    # import numpy as np
    # from functools import partial
    # from IsoNet.util.rotations import rotation_list
    # # from difflib import get_close_matches
    # from IsoNet.util.metadata import MetaData, Item, Label
    # import tensorflow as tf
    # from tensorflow.keras.models import load_model
    # strategy = tf.distribute.MirroredStrategy()
    # if 0 > 1:
    #     with strategy.scope():  #
    #         model = tf.keras.models.load_model(
    #                 "/public2/home/yuyibei/IsoNet-unet-noise/IsoNet/bin/noisedata/last_model.h5")
    # else:
    #     model = tf.keras.models.load_model("/public2/home/yuyibei/IsoNet-unet-noise/IsoNet/bin/noisedata/last_model.h5")
    
    ISONET.refine(None,subtomo_star='./subtomo.star',gpuID='0', use_unet = True) # 生成result文件夹
    # ISONET.predict(None, star_file='/public2/home/yuyibei/IsoNet-unet-noise/tomograms.star',tomo_idx=1,gpuID='0',model='/public2/home/yuyibei/IsoNet-unet-noise/results/model_iter30.h5')
    pass
    #unet 输入的时候为（批量大小,none,none,none,1）
if __name__ == '__main__':
    # from IsoNet.models.unet.model import *
    # model = Unet()

    # print(Input((None, None, None, 1)))
    # print(model.summary())
    # new_data = np.random.rand(1, 40, 40, 40)
    # print(model.predict(new_data[:,:,:,:]))
    # import torch
    # print(torch.cuda.is_available())
    main()

