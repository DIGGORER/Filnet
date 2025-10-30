
#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def mw2d(dim,missingAngle=[30,30]):#dim 表示生成的二维数组的维度大小 missingAngle 是一个包含两个元素的列表，用于指定缺失角度，默认为 [30, 30]
    mw=np.zeros((dim,dim),dtype=np.double)
    missingAngle = np.array(missingAngle)
    missing=np.pi/180*(90-missingAngle)
    for i in range(dim):
        for j in range(dim):
            y=(i-dim/2)
            x=(j-dim/2)
            if x==0:# and y!=0:
                theta=np.pi/2
            #elif x==0 and y==0:
            #    theta=0
            #elif x!=0 and y==0:
            #    theta=np.pi/2
            else:
                theta=abs(np.arctan(y/x))

            if x**2+y**2<=min(dim/2,dim/2)**2:
                if x > 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x > 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)

            if int(y) == 0:
                mw[i,j]=1
    #from mwr.util.image import norm_save
    #norm_save('mw.tif',self._mw)
    return mw

def mw3d_list(dim,missingAngle=[30,30]):#dim 表示生成的三维数组的维度大小 missingAngle 是一个包含两个元素的列表，用于指定缺失角度，默认为 [30, 30]
    mw=np.zeros((dim,dim),dtype=np.double)
    missingAngle = np.array(missingAngle)
    missing=np.pi/180*(90-missingAngle)
    for i in range(dim):
        for j in range(dim):
            y=(i-dim/2)
            x=(j-dim/2)
            if x==0:# and y!=0:
                theta=np.pi/2
            #elif x==0 and y==0:
            #    theta=0
            #elif x!=0 and y==0:
            #    theta=np.pi/2
            else:
                theta=abs(np.arctan(y/x))

            if x**2+y**2<=min(dim/2,dim/2)**2:
                if x > 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x > 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)

            if int(y) == 0:
                mw[i,j]=1
    #from mwr.util.image import norm_save
    #norm_save('mw.tif',self._mw)
    mw = mw * 0 + (1 - mw) * 1 #取反
    origin_mw3d = []
    mw3d_list = []
    for i in range(dim):
        origin_mw3d.append(mw)
    origin_mw3d = np.array(origin_mw3d)
    mw3d_1 = origin_mw3d
    mw3d_2 = origin_mw3d
    # mw3d_3 = origin_mw3d
    mw3d_4 = origin_mw3d
    mw3d_5 = origin_mw3d
    mw3d_6 = origin_mw3d

    mw3d_list.append(mw3d_1)

    mw3d_2 = np.rot90(mw3d_2, k=1, axes=(1,2))
    mw3d_list.append(mw3d_2)

    # mw3d_3 = np.rot90(mw3d_3, k=1, axes=(1,0))#这个不需要
    # mw3d_list.append(mw3d_3)

    mw3d_4 = np.rot90(mw3d_4, k=1, axes=(0,2))
    mw3d_list.append(mw3d_4)

    mw3d_5 = np.rot90(np.rot90(mw3d_5, k=1, axes=(1,0)),k=1, axes=(0,2))
    mw3d_list.append(mw3d_5)

    mw3d_6 = np.rot90(np.rot90(mw3d_6, k=1, axes=(0,2)),k=1, axes=(1,0))
    mw3d_list.append(mw3d_6)

    return mw3d_list


#import tensorflow as tf
def apply_wedge_dcube(ori_data, mw2d, mw3d=None):#ori_data是原始数据，mw2d是一个二维的Wedge-shaped mask，mw3d是一个可选的三维Wedge-shaped mask。
    if mw3d is None: #这里检查是否提供了三维的Wedge-shaped mask
        #if len(ori_data.shape) > 3:
        #    ori_data = np.squeeze(ori_data, axis=-1)
        data = np.rot90(ori_data, k=1, axes=(1,2)) #clock wise of counter clockwise??
        data = np.fft.ifft2(np.fft.fftshift(mw2d)[np.newaxis, np.newaxis, :, :] * np.fft.fft2(data))
        #data = tf.signal.ifft2d(np.fft.fftshift(mw2d)[np.newaxis, np.newaxis, :, :] * tf.signal.fft2d(data))
        data = np.real(data)
        data=np.rot90(data, k=3, axes=(1,2))

    else:
        import mrcfile
        with mrcfile.open(mw3d, permissive=True) as mrc:
            mw = mrc.data
        mwshift = np.fft.fftshift(mw)
        data = np.zeros_like(ori_data)
        for i,d in enumerate(ori_data):
            f_data = np.fft.fftn(d)
            outData = mwshift*f_data
            inv = np.fft.ifftn(outData)
            data[i] = np.real(inv).astype(np.float32)
            #data[i] = normalize(data[i],percentile=True)

    return data

def apply_wedge(ori_data, ld1 = 1, ld2 =0):
    #apply -60~+60 wedge to single cube
    data = np.rot90(ori_data, k=1, axes=(0,1)) #clock wise of counter clockwise??
    mw = TwoDPsf(data.shape[1], data.shape[2]).getMW()

    #if inverse:
    #    mw = 1-mw
    mw = mw * ld1 + (1-mw) * ld2

    mw3d = np.zeros(data.shape,dtype=np.complex)
    f_data = np.fft.fftn(data)
    for i, item in enumerate(f_data):
        mw3d[i] = mw
    mwshift = np.fft.fftshift(mw)
    outData = mwshift*f_data
    inv = np.fft.ifftn(outData)
    real = np.real(inv).astype(np.float32)
    out = np.rot90(real, k=3, axes=(0,1))
    return out

def apply_wedge1(ori_data, ld1 = 1, ld2 =0, mw3d = None): #定义了一个函数 apply_wedge1，接受四个参数：ori_data 表示原始数据，ld1 和 ld2 是两个楔形效应参数的权重默认值分别为 1 和 0，mw3d 是一个可选参数，表示三维楔形效应的文件路径。

    if mw3d is None: #检查是否提供了三维楔形效应文件路径，如果未提供，则执行以下代码块。
        data = np.rot90(ori_data, k=1, axes=(0,1)) #clock wise of counter clockwise?? #将输入数据逆时针旋转 90 度，将 ori_data 中的数据在 0 和 1 轴上进行旋转。
        mw = mw2d(data.shape[1]) #调用 mw2d 函数生成二维中值滤波器，用于生成二维的楔形效应。 在轴指定的平面中将数组旋转 90 度。旋转方向是从第一轴朝向第二轴。  k=1支旋转一次 一次90度
        #mv实际上是生成一个掩膜矩阵
        #if inverse:
        #    mw = 1-mw
        mw = mw * ld1 + (1-mw) * ld2 #将两个楔形效应参数进行加权混合，生成最终的楔形效应滤波器。

        outData = np.zeros(data.shape,dtype=np.float32) #创建一个与输入数据相同形状的全零数组，用于存储处理后的数据。
        mw_shifted = np.fft.fftshift(mw) #对楔形效应滤波器进行傅立叶变换移位，使其在频率域中心。
        for i, item in enumerate(data): #遍历数据的每个切片。
            outData_i=np.fft.ifft2(mw_shifted * np.fft.fft2(item)) #对每个数据切片执行傅立叶变换，然后与楔形效应滤波器相乘，再执行逆傅立叶变换，得到处理后的数据切片。 #实际上是原有小切片图片序列每张图片进行挖空
            outData[i] = np.real(outData_i) #将处理后的数据切片保存到 outData 中。

        outData.astype(np.float32) #将数据类型转换为 np.float32。
        outData=np.rot90(outData, k=3, axes=(0,1)) #等价于将处理后的数据逆时针旋转 90 度，将其旋转回原始方向。其实是顺时针旋转270回归原位
        return outData #返回处理后的数据
    else:
        import mrcfile
        with mrcfile.open(mw3d, permissive=True) as mrc:
            mw = mrc.data
        mw = np.fft.fftshift(mw) #对楔形效应数据进行傅立叶变换的移位操作，将其置于频率域的中心。
        mw = mw * ld1 + (1-mw) * ld2 #将楔形效应数据与权重参数 ld1 和 ld2 进行加权混合，生成最终的楔形效应滤波器。

        f_data = np.fft.fftn(ori_data) #对原始数据进行三维傅立叶变换，将其转换到频率域。
        outData = mw*f_data #将楔形效应滤波器应用到原始数据的频率域上，得到处理后的频率域数据。
        inv = np.fft.ifftn(outData) #对处理后的频率域数据执行三维逆傅立叶变换，将其转换回时域。
        outData = np.real(inv).astype(np.float32) #取逆傅立叶变换后的实部，将其转换为 np.float32 类型，得到最终的处理结果。
    return outData

#import tensorflow as tf
def other_apply_wedge_dcube(ori_data, mw2d, mw3d=None):#ori_data是原始数据，mw2d是一个二维的Wedge-shaped mask，mw3d是一个可选的三维Wedge-shaped mask。
    if mw3d is None: #这里检查是否提供了三维的Wedge-shaped mask
        #if len(ori_data.shape) > 3:
        #    ori_data = np.squeeze(ori_data, axis=-1)
        data = np.rot90(ori_data, k=1, axes=(1,2)) #clock wise of counter clockwise??
        data = np.fft.ifft2(np.fft.fftshift(mw2d)[np.newaxis, np.newaxis, :, :] * np.fft.fft2(data))
        #data = tf.signal.ifft2d(np.fft.fftshift(mw2d)[np.newaxis, np.newaxis, :, :] * tf.signal.fft2d(data))
        abs_data = np.abs(data)
        real_sign = np.sign(np.real(data))
        data = abs_data * real_sign
        
        data=np.rot90(data, k=3, axes=(1,2))

    else:
        import mrcfile
        with mrcfile.open(mw3d, permissive=True) as mrc:
            mw = mrc.data
        mwshift = np.fft.fftshift(mw)
        data = np.zeros_like(ori_data)
        for i,d in enumerate(ori_data):
            f_data = np.fft.fftn(d)
            outData = mwshift*f_data
            inv = np.fft.ifftn(outData)
            
            abs_inv = np.abs(inv).astype(np.float32)
            real_sign = np.sign(np.real(inv))
            data[i] = (abs_inv * real_sign).astype(np.float32)
            #data[i] = normalize(data[i],percentile=True)

    return data

def other_apply_wedge1(ori_data, ld1 = 1, ld2 =0, mw3d = None): #定义了一个函数 apply_wedge1，接受四个参数：ori_data 表示原始数据，ld1 和 ld2 是两个楔形效应参数的权重默认值分别为 1 和 0，mw3d 是一个可选参数，表示三维楔形效应的文件路径。

    if mw3d is None: #检查是否提供了三维楔形效应文件路径，如果未提供，则执行以下代码块。
        data = np.rot90(ori_data, k=1, axes=(0,1)) #clock wise of counter clockwise?? #将输入数据逆时针旋转 90 度，将 ori_data 中的数据在 0 和 1 轴上进行旋转。
        mw = mw2d(data.shape[1]) #调用 mw2d 函数生成二维中值滤波器，用于生成二维的楔形效应。 在轴指定的平面中将数组旋转 90 度。旋转方向是从第一轴朝向第二轴。  k=1支旋转一次 一次90度
        #mv实际上是生成一个掩膜矩阵
        #if inverse:
        #    mw = 1-mw
        mw = mw * ld1 + (1-mw) * ld2 #将两个楔形效应参数进行加权混合，生成最终的楔形效应滤波器。

        outData = np.zeros(data.shape,dtype=np.float32) #创建一个与输入数据相同形状的全零数组，用于存储处理后的数据。
        mw_shifted = np.fft.fftshift(mw) #对楔形效应滤波器进行傅立叶变换移位，使其在频率域中心。
        for i, item in enumerate(data): #遍历数据的每个切片。
            outData_i=np.fft.ifft2(mw_shifted * np.fft.fft2(item)) #对每个数据切片执行傅立叶变换，然后与楔形效应滤波器相乘，再执行逆傅立叶变换，得到处理后的数据切片。 #实际上是原有小切片图片序列每张图片进行挖空
            abs_outData_i = np.abs(outData_i)
            real_sign = np.sign(np.real(outData_i))
            outData[i] = abs_outData_i * real_sign #将处理后的数据切片保存到 outData 中。

        outData.astype(np.float32) #将数据类型转换为 np.float32。
        outData=np.rot90(outData, k=3, axes=(0,1)) #等价于将处理后的数据逆时针旋转 90 度，将其旋转回原始方向。其实是顺时针旋转270回归原位
        return outData #返回处理后的数据
    else:
        import mrcfile
        with mrcfile.open(mw3d, permissive=True) as mrc:
            mw = mrc.data
        mw = np.fft.fftshift(mw) #对楔形效应数据进行傅立叶变换的移位操作，将其置于频率域的中心。
        mw = mw * ld1 + (1-mw) * ld2 #将楔形效应数据与权重参数 ld1 和 ld2 进行加权混合，生成最终的楔形效应滤波器。

        f_data = np.fft.fftn(ori_data) #对原始数据进行三维傅立叶变换，将其转换到频率域。
        outData = mw*f_data #将楔形效应滤波器应用到原始数据的频率域上，得到处理后的频率域数据。
        inv = np.fft.ifftn(outData) #对处理后的频率域数据执行三维逆傅立叶变换，将其转换回时域。
        abs_inv=np.abs(inv)
        real_sign = np.sign(np.real(inv))
        outData = (abs_inv*real_sign).astype(np.float32) #取逆傅立叶变换后的实部，将其转换为 np.float32 类型，得到最终的处理结果。
    return outData
