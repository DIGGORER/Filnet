"""
Generate mask by comparing local variance and global variance
"""
import numpy as np
import logging
#这个函数的作用是基于输入的三维体积数据，生成一个二值掩模（mask），其中包含高于给定百分比阈值的最大值的位置。
def maxmask(tomo, side=5,percentile=60):#tomo输入的三维体积数据，通常是一个 Numpy数组 side最大滤波器的大小，用于确定局部最大值的窗口大小，默认为 5。 percentile=60 百分位数阈值，用于确定要保留的最大值的百分比，默认为 60
    from scipy.ndimage.filters import maximum_filter #导入 scipy 库中的 maximum_filter 函数，用于执行最大滤波操作。
    # print('maximum_filter')
    #对输入的三维体积数据应用最大滤波器，以便检测局部最大值。
    filtered = maximum_filter(-tomo, 2*side+1, mode='reflect') #最大滤波操作 -tomo可以将原始数据中的最大值转换为最小值，这样最大滤波器可以检测到原始数据中的最大值  2*side+1是滤波器的大小，表示在每个方向上滤波器的窗口大小。mode='reflect'表示在边界处采用反射模式进行处理，确保边界处的滤波效果与内部一致。
    out =  filtered > np.percentile(filtered,100-percentile) # 使用 np.percentile() 函数计算滤波后数据的第 (100-percentile) 百分位数，得到阈值。 将滤波后的数据与阈值进行比较，生成一个布尔数组，其中 True 表示滤波后的数据高于阈值，False 表示低于阈值。
    out = out.astype(np.uint8)#将滤波后的数据与阈值进行比较，生成一个布尔数组，其中 True 表示滤波后的数据高于阈值，False 表示低于阈值。
    return out #返回生成的二值掩模。掩模中的高亮区域表示输入数据中高于给定百分比阈值的最大值的位置。

#基于输入的三维体积数据，生成一个二值掩模（mask），其中包含高于给定百分比阈值的标准差的位置
def stdmask(tomo,side=10,threshold=60):#tomo: 输入的三维体积数据，通常是一个 Numpy 数组。 side: 卷积核的大小，用于计算局部标准差，默认为10,threshold: 百分位数阈值，用于确定要保留的标准差的百分比，默认为 60。
    from scipy.signal import convolve #导入 scipy 库中的 convolve 函数，用于执行卷积操作。
    # print('std_filter')
    tomosq = tomo**2 #计算输入数据的平方，以便后续计算
    ones = np.ones(tomo.shape) #创建一个与输入数据形状相同的全 1 数组。
    eps = 0.001 #定义一个很小的常数，用于避免除以零的情况。
    kernel = np.ones((2*side+1, 2*side+1, 2*side+1)) #创建一个三维的全 1 卷积核，用于计算局部统计量。
    s = convolve(tomo, kernel, mode="same") #对输入数据应用卷积核，计算局部的数据总和。
    s2 = convolve(tomosq, kernel, mode="same") #对输入数据的平方应用卷积核，计算局部的数据平方和。
    ns = convolve(ones, kernel, mode="same") + eps #全 1 数组应用卷积核，计算每个局部的元素个数，并加上一个很小的常数以避免除以零。

    out = np.sqrt((s2 - s**2 / ns) / ns + eps) #根据局部的数据总和、数据平方和和元素个数，计算局部的标准差。#这里使用了无偏估计，避免了方差的偏低。
    # out = out>np.std(tomo)*threshold
    out  = out>np.percentile(out, 100-threshold) #使用 np.percentile() 函数计算标准差数据的第 (100-threshold) 百分位数，得到阈值。#将标准差数据与阈值进行比较，生成一个布尔数组，其中 True 表示标准差高于阈值，False 表示低于阈值。

    return out.astype(np.uint8) #回生成的二值掩模。掩模中的高亮区域表示输入数据中标准差高于给定百分比阈值的位置。

def boundary_mask(tomo, mask_boundary, binning = 2):
    out = np.zeros(tomo.shape, dtype = np.float32)
    import os
    import sys
    if mask_boundary[-4:] == '.mod':
        os.system('model2point {} {}.point >> /dev/null'.format(mask_boundary, mask_boundary[:-4]))
    else:
        logging.error("mask boundary file should end with .mod but got {} !\n".format(mask_boundary))
        sys.exit()
    
    points = np.loadtxt(mask_boundary[:-4]+'.point', dtype = np.float32)/binning
    
    def get_polygon(points):
        if len(points) == 0:
            logging.info("No polygonal mask")
            return None
        elif len(points) <= 2:
            logging.error("In {}, {} points cannot defines a polygon of mask".format(mask_boundary, len(points)))
            sys.exit()
        else:
            logging.info("In {}, {} points defines a polygon of mask".format(mask_boundary, len(points)))
            return points[:,[1,0]]
    
    if points.ndim < 2: 
        logging.error("In {}, too few points to define a boundary".format(mask_boundary))
        sys.exit()

    z1=points[-2][-1]
    z0=points[-1][-1]

    if abs(z0 - z1) < 5:
        zmin = 0
        zmax = tomo.shape[0]
        polygon = get_polygon(points)
        logging.info("In {}, all points defines a polygon with full range in z".format(mask_boundary))

    else:
        zmin = max(min(z0,z1),0) 
        zmax = min(max(z0,z1),tomo.shape[0])
        polygon = get_polygon(points[:-2])
        logging.info("In {}, the last two points defines the z range of mask".format(mask_boundary))

    '''
    if points.ndim != 2 or points.shape[0] == 1:
        logging.error("In {}, too few points to define a boundary".format(mask_boundary))
        sys.exit()
    elif points.shape[0] == 2 and np.abs(points[-1][-1] - points[-2][-1]) > 5:
        logging.info("In {},the two points defines the z range of mask".format(mask_boundary))
        zmin = max(min(points[-1,-1],points[-2][-1]),0)
        zmax = min(max(points[-1,-1],points[-2][-1]),tomo.shape[0])
        polygon = None
    elif points.shape[0] <5 and np.abs(points[-1][-1] - points[-2][-1]) < 6 :
        logging.error("In {}, {} points can not make a polygon".format(mask_boundary,points.shape[0]-2))
        sys.exit()
    elif points.shape[0] > 4 and np.abs(points[-1][-1] - points[-2][-1]) > 5:
        logging.info("In {}, the last two points defines the z range of mask".format(mask_boundary))
        zmin = max(min(points[-1,-1],points[-2][-1]),0)
        zmax = min(max(points[-1,-1],points[-2][-1]),tomo.shape[0])
        polygon = points[:-2,[1,0]]
    else:
        zmin = 0
        zmax = tomo.shape[0]
        polygon = points[:,[1,0]]
    '''
    zmin = int(zmin)
    zmax = int(zmax)
    if polygon is None:
        out[zmin:zmax,:,:] = 1
    else:
        from matplotlib.path import Path
        poly_path = Path(polygon)
        y, x = np.mgrid[:tomo.shape[1],:tomo.shape[2]]
        coors = np.hstack((y.reshape(-1, 1), x.reshape(-1,1)))
        mask = poly_path.contains_points(coors)
        mask = mask.reshape(tomo.shape[1],tomo.shape[2])
        mask = mask.astype(np.float32)
        out[zmin:zmax,:,:] = mask[np.newaxis,:,:]

    return out

if __name__ == "__main__":
    import sys
    import mrcfile
    args = sys.argv
    with mrcfile.open(args[1], permissive=True) as n:
        tomo = n.data
    mask = stdmask_mpi(tomo,cubelen=20,cubesize=80,ncpu=20,if_rescale=True)
    with mrcfile.new(args[2],overwrite=True) as n:
        n.set_data(mask)