import numpy as np
import mrcfile
import os
import logging
import gc
import scipy.fft

def tom_ctf1d(pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048):

    ny = 1 / pixelsize


    lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    lambda2 = lambda1 * 2


    points = np.arange(0,length)
    points = points.astype(float)
    points = points/(2 * length)*ny

    k2 = points**2
    term1 = lambda1**3 * cs * k2**2

    w = np.pi / 2 * (term1 + lambda2 * defocus * k2) - phaseshift

    acurve = np.cos(w) * amplitude
    pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
    bfactor = np.exp(-bfactor * k2 * 0.25)


    return (pcurve + acurve)*bfactor

def wiener1d(angpix, voltage, cs, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift):
    data = np.arange(0,1+1/2047.,1/2047.)
    highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi
    highpass = 1-np.cos(highpass)

    snr = np.exp(-data * snrfalloff * 100 / angpix) * (10 ** deconvstrength) * highpass
    #snr[0] = -1
    ctf = tom_ctf1d(angpix*1e-10, voltage * 1e3, cs * 1e-3, -defocus*1e-6, 0.07, phaseshift / 180 * np.pi, 0)
    if phaseflipped:
        ctf = abs(ctf)

    wiener = ctf/(ctf*ctf+1/snr)
    return ctf, wiener

# 将原数据集进行处理 如果先前用过mask也不会对mask过后数据进行处理
def tom_deconv_tomo(vol_file, out_file,angpix, voltage, cs, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift, ncpu=8):
    # 读取体积文件 图片大小464 x 480 在z轴上有125张图片。
    with mrcfile.open(vol_file, permissive=True) as f:
        header_in = f.header
        vol = f.data
        voxelsize = f.voxel_size

    # 生成频域数据
    data = np.arange(0,1+1/2047.,1/2047.) #1维 [2048]
    highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi;#1维 [2048]
    highpass = 1-np.cos(highpass);
    eps = 1e-6 #常数
    snr = np.exp(-data * snrfalloff * 100 / angpix) * (10**deconvstrength) * highpass + eps #1维 2048
    # snr[0] = -1
    # 计算CTF（对比度传递函数）
    ctf = tom_ctf1d(angpix*1e-10, voltage * 1e3, cs * 1e-3, -defocus*1e-6, 0.07, phaseshift / 180 * np.pi, 0);#1维 2048
    if phaseflipped:
        ctf = abs(ctf)

    # 计算CTF（对比度传递函数）
    wiener = ctf/(ctf*ctf+1/snr);#1维 2048

    denom = ctf*ctf+1/snr #1维 2048
    #np.savetxt('den.txt',denom)
    #np.savetxt('snr.txt',snr)
    #np.savetxt('hipass.txt',highpass)
    #np.savetxt('ctf.txt',ctf)
    #np.savetxt('wiener.txt',wiener, fmt='%f')

    # 准备体积的坐标
    s1 = - int(np.shape(vol)[1] / 2) #常数 -232
    f1 = s1 + np.shape(vol)[1] - 1 #常数231
    m1 = np.arange(s1,f1+1) #1维 464

    s2 = - int(np.shape(vol)[0] / 2) #常数 -100
    f2 = s2 + np.shape(vol)[0] - 1 #常数 99
    m2 = np.arange(s2,f2+1) #一维 200

    s3 = - int(np.shape(vol)[2] / 2) #常数 -240
    f3 = s3 + np.shape(vol)[2] - 1 #常数 239
    m3 = np.arange(s3,f3+1) #1维 480

    #s3 = -floor(size(vol,3)/2);
    #f3 = s3 + size(vol,3) - 1;
    # 生成坐标的网格
    x, y, z = np.meshgrid(m1,m2,m3)
    x = x.astype(np.float32) / np.abs(s1) #3维 200，464，480
    y = y.astype(np.float32) / np.abs(s2) #3维 200，464，480
    z = z.astype(np.float32) / np.maximum(1, np.abs(s3)) #3维 200，464，480
    #z = z.astype(float) / np.abs(s3);
    r = np.sqrt(x**2+y**2+z**2) #3维 200，464，480
    del x,y,z
    gc.collect()
    r = np.minimum(1, r)
    r = np.fft.ifftshift(r)#傅里叶变换 3维 200，464，480

    #x = 0:1/2047:1;
    ramp = np.interp(r, data,wiener).astype(np.float32) #一维线性插值 #3维 200，464，480
    del r
    gc.collect()
    # 插值维纳滤波器值
    #ramp = np.interp(data,wiener,r);
    # 执行去卷积
    deconv = np.real(scipy.fft.ifftn(scipy.fft.fftn(vol, overwrite_x=True, workers=ncpu) * ramp, overwrite_x=True, workers=ncpu))
    deconv = deconv.astype(np.float32) #3维 200，464，480
    # 归一化去卷积后的体积
    std_deconv = np.std(deconv) #常数
    std_vol = np.std(vol) #常数
    ave_vol = np.average(vol) #常数
    del vol,ramp
    gc.collect()
    # deconv = deconv/std_deconv* std_vol + ave_vol
    deconv /= std_deconv
    deconv *= std_vol
    deconv += ave_vol # 3维 维度200，464，480
    gc.collect()
    # 保存去卷积后的体积
    if out_file is not None:
        out_name = out_file
    else:
        out_name = os.path.splitext(vol_file)[0]+'_deconv.mrc'

    with mrcfile.new(out_name,overwrite=True) as n:
        n.set_data(deconv) #.astype(type(vol[0,0,0]))
        n.voxel_size = voxelsize
        n.header.origin = header_in.origin
        n.header.nversion = header_in.nversion
    #return real(ifftn(fftn(single(vol)).*ramp));
    return os.path.splitext(vol_file)[0]+'_deconv.mrc'

class Chunks:
    def __init__(self,chunk_size=200,overlap=0.25):
        self.overlap = overlap
        #num can be either int or tuple
        self.chunk_size = chunk_size

    def get_chunks(self,tomo_name):
        #side*(1-overlap)*(num-1)+side = sp + side*overlap -> side *(1-overlap) * num = side
        root_name = os.path.splitext(os.path.basename(tomo_name))[0]
        with mrcfile.open(tomo_name, permissive=True) as f:
            vol = f.data#.astype(np.float32)
        cropsize = int(self.chunk_size*(1+self.overlap))
        cubesize = self.chunk_size
        sp = np.array(vol.shape)
        self._sp = sp
        self._N = sp//cubesize+1
        padi = int((cropsize - cubesize)/2)
        padsize = (self._N*cubesize + padi - sp).astype(int)
        data = np.pad(vol,((padi,padsize[0]),(padi,padsize[1]),(padi,padsize[2])),'symmetric')
        chunks_file_list = []
        for i in range(self._N[0]):
            for j in range(self._N[1]):
                for k in range(self._N[2]):
                    cube = data[i*cubesize:i*cubesize+cropsize,
                            j*cubesize:j*cubesize+cropsize,
                            k*cubesize:k*cubesize+cropsize]
                    file_name = './deconv_temp/'+root_name+'_{}_{}_{}.mrc'.format(i,j,k)
                    with mrcfile.new(file_name,overwrite=True) as n:
                        n.set_data(cube)
                    chunks_file_list.append(file_name)
        return chunks_file_list


    def restore(self,new_file_list):
        cropsize = int(self.chunk_size*(1+self.overlap))
        cubesize = self.chunk_size
        new = np.zeros((self._N[0]*cubesize,self._N[1]*cubesize,self._N[2]*cubesize),dtype = np.float32)
        start=int((cropsize-cubesize)/2)
        end=int((cropsize+cubesize)/2)
        for i in range(self._N[0]):
            for j in range(self._N[1]):
                for k in range(self._N[2]):
                    one_chunk_file = new_file_list[i*self._N[1]*self._N[2]+j*self._N[2]+k]
                    with mrcfile.open(one_chunk_file, permissive=True) as f:
                        one_chunk_data = f.data
                    new[i*cubesize:(i+1)*cubesize,j*cubesize:(j+1)*cubesize,k*cubesize:(k+1)*cubesize] \
                            = one_chunk_data[start:end,start:end,start:end]
                    
        return new[0:self._sp[0],0:self._sp[1],0:self._sp[2]]


def deconv_one(tomo, out_tomo, voltage=300.0, cs=2.7, defocus=1.0, pixel_size=1.0,snrfalloff=1.0, deconvstrength=1.0,highpassnyquist=0.02,chunk_size=200,overlap_rate = 0.25,ncpu=4):
    """
    对输入的电子断层扫描三维重建体像进行去卷积处理。

    参数:
    tomo (str): 输入三维重建体像文件路径
    out_tomo (str): 输出去卷积后的体像文件路径
    voltage (float): 电子束加速电压(kV)
    cs (float): 球面像差系数(mm)
    defocus (float): 像差值(um)
    pixel_size (float): 像素尺寸(A)
    snrfalloff (float): SNR衰减系数
    deconvstrength (float): 去卷积强度
    highpassnyquist (float): 高通滤波器的截止频率
    chunk_size (int): 将体像分块处理时每块的大小
    overlap_rate (float): 分块处理时块之间的重叠率
    ncpu (int): 使用的CPU核心数

    返回:
    None
    """
    import mrcfile
    from multiprocessing import Pool
    from functools import partial
    from IsoNet.util.deconvolution import tom_deconv_tomo,Chunks
    import shutil
    import time
    # 记录开始时间
    t1 = time.time()
    # 清理以前的文件夹并创建临时文件夹
    if os.path.isdir('./deconv_temp'):
        shutil.rmtree('./deconv_temp')
    os.mkdir('./deconv_temp')

    # 获取输入体像文件的根名称
    root_name = os.path.splitext(os.path.basename(tomo))[0]

    print(root_name)

    # 记录去卷积参数
    logging.info('deconv: {}| pixel: {}| defocus: {}| snrfalloff:{}| deconvstrength:{}'.format(tomo, pixel_size, defocus ,snrfalloff, deconvstrength))
    # 如果不分块处理，直接调用去卷积函数
    if chunk_size is None:
        tom_deconv_tomo(tomo,out_tomo,pixel_size, voltage, cs, defocus,snrfalloff,deconvstrength,highpassnyquist,phaseflipped=False, phaseshift=0,ncpu=ncpu)
    else:
        # 分块处理
        c = Chunks(chunk_size=chunk_size,overlap=overlap_rate)
        chunks_list = c.get_chunks(tomo) # list of name of subtomograms  # 获取分块列表
        # chunks_gpu_num_list = [[array,j%num_gpu] for j,array in enumerate(chunks_list)]
        chunks_deconv_list = []
        with Pool(ncpu) as p:
            # 创建部分函数用于多进程调用
            partial_func = partial(tom_deconv_tomo,out_file=None,angpix=pixel_size,voltage=voltage, cs=cs, defocus=defocus, snrfalloff=snrfalloff,
                    deconvstrength=deconvstrength, highpassnyquist=highpassnyquist, phaseflipped=False, phaseshift=0,ncpu=1) 
            # 多进程去卷积处理每一块
            chunks_deconv_list = list(p.map(partial_func,chunks_list))
        # 合并去卷积后的块
        vol_restored = c.restore(chunks_deconv_list)
        # 读取输入体像的头信息
        with mrcfile.open(tomo, permissive=True) as n:
            header_input = n.header
            pixel_size = n.voxel_size

        # 将去卷积后的体像写入输出文件
        with mrcfile.new(out_tomo, overwrite=True) as mrc:
            mrc.set_data(vol_restored)
            mrc.voxel_size = pixel_size
            #print(header_input)
            #print(mrc.header)
            mrc.header.origin = header_input.origin
            mrc.header.nversion=header_input.nversion
    # 删除临时文件夹
    shutil.rmtree('./deconv_temp')

    # 记录结束时间并计算耗时
    t2 = time.time()
    logging.info('time consumed: {:10.4f} s'.format(t2-t1))

if __name__=='__main__':
    import sys
    import time
    import argparse
    parser = argparse.ArgumentParser(
    description="Deconvolve with cpu", add_help=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("mrcFile",type=str,default=None)
    parser.add_argument("outFile",type=str,default=None)
    parser.add_argument("defocus",type=float,default=None)
    parser.add_argument("pixsize",type=float,default=None)
    parser.add_argument("snrfalloff",type=float,default=1.0)
    parser.add_argument("deconvstrength",type=float,default=1.0)
    parser.add_argument("--tile",type=tuple,default=(1,4,4))
    parser.add_argument("--ncpu",type=int,default=8)
    args = parser.parse_args()
    start = time.time()

    # deconv_one(args.mrcFile, args.outFile,defocus=args.defocus/10000.0, pixel_size=args.pixsize,snrfalloff=args.snrfalloff, deconvstrength=args.deconvstrength,tile=args.tile,ncpu=args.ncpu)
    tom_deconv_tomo(args.mrcFile, voltage=args.voltage, cs=args.cs, defocus=args.defocus/10000.0, angpix=args.pixsize,snrfalloff=args.snrfalloff, deconvstrength=args.deconvstrength,
                    highpassnyquist=0.1, phaseflipped=False, phaseshift=0)
