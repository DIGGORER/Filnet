import numpy as np
# from tifffile import imsave,imread
# from mwr.util.mrcfile import *
# from mwr.util.filter import no_background_patches
#import mrcfile
#自己写的一个函数用于不可以改形状的模型进行改变形状图片
def get_turn_notation_list(notation): #一共获得24种不同方向图像
    turn_notation_list = []
    for i in range(4):
        temp_notation = np.rot90(notation,k = i,axes=(1, 2)) #向下翻转
        for a in range(4):
            last_notation = np.rot90(temp_notation, k = a,axes=(2, 3)) #逆时针翻转
            turn_notation_list.append(last_notation)

    temp_notation = np.rot90(notation, k=1, axes=(1, 3))#向右翻转
    for a in range(4):
        last_notation = np.rot90(temp_notation, k=a, axes=(2, 3))  # 逆时针翻转
        turn_notation_list.append(last_notation)

    temp_notation = np.rot90(notation, k=1, axes=(3, 1))  # 向左翻转
    for a in range(4):
        last_notation = np.rot90(temp_notation, k=a, axes=(2, 3))  # 逆时针翻转
        turn_notation_list.append(last_notation)
    return turn_notation_list
def recover_turn_notation_list(turn_notation_list): # 将 这24种不同方向的图像复原
    recover_notation_list = []
    for i in range(4):#先将向下翻转的矩阵翻转
        for a in range(4):
            temp_notation = np.rot90(turn_notation_list[i * 4 + a], k = a,axes=(2, 1))
            temp_notation = np.rot90(temp_notation, k = i,axes=(1, 0))
            recover_notation_list.append(temp_notation)

    for i in range(4):#把向右翻转的矩阵翻转
        temp_notation = np.rot90(turn_notation_list[16+i], k = i, axes=(2, 1))
        temp_notation = np.rot90(temp_notation, k=1, axes=(2, 0))
        recover_notation_list.append(temp_notation)

    for i in range(4):#把向左翻转的矩阵翻转
        temp_notation = np.rot90(turn_notation_list[20+i], k = i, axes=(2, 1))
        temp_notation = np.rot90(temp_notation, k=1, axes=(0, 2))
        recover_notation_list.append(temp_notation)
    return recover_notation_list

def change_size(cude_size=80,small_cude_size=64,data_seg=None):
    cude_dis_size = cude_size - small_cude_size #16
    data_seg_1 = data_seg[:small_cude_size, :small_cude_size, :small_cude_size]
    data_seg_2 = data_seg[cude_dis_size:cude_size, :small_cude_size, :small_cude_size]
    data_seg_3 = data_seg[:small_cude_size, cude_dis_size:cude_size, :small_cude_size]
    data_seg_4 = data_seg[cude_dis_size:cude_size, cude_dis_size:cude_size, :small_cude_size]
    data_seg_5 = data_seg[:small_cude_size, :small_cude_size, cude_dis_size:cude_size]
    data_seg_6 = data_seg[cude_dis_size:cude_size, :small_cude_size, cude_dis_size:cude_size]
    data_seg_7 = data_seg[:small_cude_size, cude_dis_size:cude_size, cude_dis_size:cude_size]
    data_seg_8 = data_seg[cude_dis_size:cude_size, cude_dis_size:cude_size, cude_dis_size:cude_size]
    data_seglist = [data_seg_1[np.newaxis, :, :, :],data_seg_2[np.newaxis, :, :, :],
                    data_seg_3[np.newaxis, :, :, :],data_seg_4[np.newaxis, :, :, :],
                    data_seg_5[np.newaxis, :, :, :],data_seg_6[np.newaxis, :, :, :],
                    data_seg_7[np.newaxis, :, :, :],data_seg_8[np.newaxis, :, :, :]]
    return data_seglist
def combine_cude(cude_size=80,small_cude_size=64,predicted_list=None):#将预测后的值相加并且重叠部分取平均
    cude_dis_size = cude_size - small_cude_size #16
    combined_data = np.zeros((cude_size, cude_size, cude_size))
    #把预测好的图像进行合成
    combined_data[:small_cude_size, :small_cude_size, :small_cude_size] += np.array(predicted_list[0])
    combined_data[cude_dis_size:cude_size, :small_cude_size, :small_cude_size] += np.array(predicted_list[1])
    combined_data[:small_cude_size, cude_dis_size:cude_size, :small_cude_size] += np.array(predicted_list[2])
    combined_data[cude_dis_size:cude_size, cude_dis_size:cude_size, :small_cude_size] += np.array(predicted_list[3])
    combined_data[:small_cude_size, :small_cude_size, cude_dis_size:cude_size] += np.array(predicted_list[4])
    combined_data[cude_dis_size:cude_size, :small_cude_size, cude_dis_size:cude_size] += np.array(predicted_list[5])
    combined_data[:small_cude_size, cude_dis_size:cude_size, cude_dis_size:cude_size] += np.array(predicted_list[6])
    combined_data[cude_dis_size:cude_size, cude_dis_size:cude_size, cude_dis_size:cude_size] += np.array(predicted_list[7])

    combined_data[cude_dis_size:small_cude_size, cude_dis_size:small_cude_size, cude_dis_size:small_cude_size] /= 8

    combined_data[cude_dis_size:small_cude_size, cude_dis_size:small_cude_size, :cude_dis_size] /= 4
    combined_data[cude_dis_size:small_cude_size, cude_dis_size:small_cude_size, small_cude_size:cude_size] /= 4
    combined_data[cude_dis_size:small_cude_size, :cude_dis_size, cude_dis_size:small_cude_size] /= 4
    combined_data[cude_dis_size:small_cude_size, small_cude_size:cude_size, cude_dis_size:small_cude_size] /= 4
    combined_data[:cude_dis_size, cude_dis_size:small_cude_size, cude_dis_size:small_cude_size] /= 4
    combined_data[small_cude_size:cude_size, cude_dis_size:small_cude_size, cude_dis_size:small_cude_size] /= 4

    combined_data[cude_dis_size:small_cude_size, :cude_dis_size, :cude_dis_size] /= 2
    combined_data[cude_dis_size:small_cude_size, small_cude_size:cude_size, :cude_dis_size] /= 2
    combined_data[cude_dis_size:small_cude_size, :cude_dis_size, small_cude_size:cude_size] /= 2
    combined_data[cude_dis_size:small_cude_size, small_cude_size:cude_size, small_cude_size:cude_size] /= 2

    combined_data[:cude_dis_size, cude_dis_size:small_cude_size, :cude_dis_size] /= 2
    combined_data[:cude_dis_size, cude_dis_size:small_cude_size, small_cude_size:cude_size] /= 2
    combined_data[small_cude_size:cude_size, cude_dis_size:small_cude_size, :cude_dis_size] /= 2
    combined_data[small_cude_size:cude_size, cude_dis_size:small_cude_size, small_cude_size:cude_size] /= 2

    combined_data[:cude_dis_size, :cude_dis_size, cude_dis_size:small_cude_size] /= 2
    combined_data[:cude_dis_size, small_cude_size:cude_size, cude_dis_size:small_cude_size] /= 2
    combined_data[small_cude_size:cude_size, :cude_dis_size, cude_dis_size:small_cude_size] /= 2
    combined_data[small_cude_size:cude_size, small_cude_size:cude_size, cude_dis_size:small_cude_size] /= 2

    return combined_data
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

def toUint8(data):
    data=np.real(data)
    data=data.astype(np.double)
    ma=np.max(data)
    mi=np.min(data)
    data=(data-mi)/(ma-mi)*255
    data=np.clip(data,0,255)
    data=data.astype(np.uint8)
    return data

def toUint16(data):
    data=np.real(data)
    data=data.astype(np.double)
    ma=np.max(data)
    mi=np.min(data)
    data=(data-mi)/(ma-mi)*65535
    data=data.astype(np.uint16)
    return data

def crop_center(img,cropx,cropy,cropz):
    z,y,x = img.shape[0],img.shape[1],img.shape[2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)
    return img[startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]

def create_seed_2D(img2D,nPatchesPerSlice,patchSideLen):
    y,x = img2D.shape[0],img2D.shape[1]

    seedx = np.random.rand(nPatchesPerSlice)*(x-patchSideLen)+patchSideLen//2
    seedy = np.random.rand(nPatchesPerSlice)*(y-patchSideLen)+patchSideLen//2
    seedx = seedx.astype(int)
    seedy = seedy.astype(int)

    return seedx,seedy
def print_filter_mask(img3D,nPatchesPerSlice,patchSideLen,threshold=0.4,percentile=99.9):
    sp=img3D.shape
    mask=np.zeros(sp).astype(np.uint8)
    myfilter = no_background_patches(threshold=threshold,percentile=percentile)
    for i in range(sp[0]):
        mask[i]=myfilter(img3D[i].reshape(1,sp[1],sp[2]),(patchSideLen,patchSideLen))

    return mask

def create_filter_seed_2D(img2D,nPatchesPerSlice,patchSideLen, patch_mask):
    
    sp=img2D.shape

    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip((patchSideLen,patchSideLen), sp)])
    valid_inds = np.where(patch_mask[border_slices])
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    sample_inds = np.random.choice(len(valid_inds[0]),nPatchesPerSlice , replace=len(valid_inds[0])< nPatchesPerSlice)
    rand_inds = [v[sample_inds] for v in valid_inds]
    return rand_inds[1],rand_inds[0]


def create_cube_seeds(img3D,nCubesPerImg,cubeSideLen,mask=None):
    sp=img3D.shape
    if mask is None:
        cubeMask=np.ones(sp)
    else:
        cubeMask=mask
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip((cubeSideLen,cubeSideLen,cubeSideLen), sp)])
    valid_inds = np.where(cubeMask[border_slices])
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    sample_inds = np.random.choice(len(valid_inds[0]), nCubesPerImg, replace=len(valid_inds[0]) < nCubesPerImg)
    rand_inds = [v[sample_inds] for v in valid_inds]
    return (rand_inds[0],rand_inds[1], rand_inds[2])
    #return (seedz,seedy,seedx)

def crop_seed2D(img2D,seedx,seedy,cropx,cropy):
    y,x = img2D.shape[0],img2D.shape[1]
    
    patchshape = img2D[seedy-(cropy//2):seedy+(cropy)//2,seedx-(cropx//2):seedx+(cropx//2)].shape
    return img2D[seedy-(cropy//2):seedy+(cropy)//2,seedx-(cropx//2):seedx+(cropx//2)]#.astype(int)

def create_patch_image_2D(image2D,seedx,seedy,patchSideLen):
    y,x = image2D.shape
    patches = np.zeros([seedx.size,patchSideLen,patchSideLen])
    for i in range(seedx.size):
        patches[i] = crop_seed2D(image2D,seedx[i],seedy[i],patchSideLen,patchSideLen)
    return patches
    
def crop_cubes(img3D,seeds,cubeSideLen):
    size=len(seeds[0])
    cube_size=(cubeSideLen,cubeSideLen,cubeSideLen)
    cubes=[img3D[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,cube_size))] for r in zip(*seeds)]
    cubes=np.array(cubes)
    return cubes

def rotate(data,angle,axes=0):
    sp=data.shape
    theta=angle/180*np.pi
    cos_theta=np.cos(theta)
    sin_theta=np.sin(theta)
    sideLen=np.min([sp[1],sp[2]])//np.sqrt(2)
    sideLen=sideLen.astype(np.uint16)
    rotated=np.zeros([sp[0],sideLen,sideLen],dtype=np.uint8)
    for _z in range(sp[0]):
        print(_z)
        for _y in range(sideLen):
            for _x in range(sideLen):
                y_prime=int((_y-sideLen//2)*cos_theta-(_x-sideLen//2)*sin_theta+sp[1]//2)
                x_prime= int((_x-sideLen//2)*cos_theta+(_y-sideLen//2)*sin_theta+sp[2]//2)
                rotated[_z,_y,_x]=data[_z,y_prime,x_prime]
    return rotated