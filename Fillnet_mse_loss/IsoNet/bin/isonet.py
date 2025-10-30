# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import sys
import os

import fire
import logging
import os, sys, traceback
from IsoNet.util.dict2attr import Arg,check_parse,idx2list #这个库有我所要函数变量参数
from fire import core
from IsoNet.util.metadata import MetaData,Label,Item

class ISONET:
    """
    ISONET: Train on tomograms and restore missing-wedge\n
    for detail description, run one of the following commands:

    isonet.py prepare_star -h
    isonet.py prepare_subtomo_star -h
    isonet.py deconv -h
    isonet.py make_mask -h
    isonet.py extract -h
    isonet.py refine -h
    isonet.py predict -h
    isonet.py resize -h
    isonet.py gui -h
    """
    """
    ISONET: 这是一个用于在层析重建图像(tomograms)上进行训练并恢复缺失楔形(missing-wedge)的工具。
    该工具提供了以下命令用于不同的功能:
    isonet.py prepare_star -h
        准备用于训练的星文件(STAR file)数据。星文件是电子显微镜领域常用的一种数据格式。
    isonet.py prepare_subtomo_star -h
        准备用于训练的子层析重建图像(subtomograms)的星文件数据。
    isonet.py deconv -h
        对层析重建图像进行去卷积(deconvolution)处理。
    isonet.py make_mask -h
        生成用于训练的掩码(mask)数据。
    isonet.py extract -h
        从层析重建图像中提取子层析重建图像。
    isonet.py refine -h
        对训练模型进行refinement(精炼)操作。
    isonet.py predict -h
        使用训练好的模型对新的层析重建图像进行预测(prediction)。
    isonet.py resize -h
        调整层析重建图像的大小。
    isonet.py gui -h
        启动图形用户界面(GUI)版本的ISONET工具。
    如需了解每个命令的详细说明,可运行相应的命令并加上"-h"参数,例如"isonet.py prepare_star -h"。
    """

    #log_file = "log.txt"
    #创建tomograms.star文件记录相关数据集信息, folder_name放数据集文件夹路径不是具体数据集的路径
    def prepare_star(self,folder_name, output_star='tomograms.star',pixel_size = 10.0, defocus = 0.0, number_subtomos = 100):
        """
        \nThis command generates a tomograms.star file from a folder containing only tomogram files (.mrc or .rec).\n
        isonet.py prepare_star folder_name [--output_star] [--pixel_size] [--defocus] [--number_subtomos]
        :param folder_name: (None) directory containing tomogram(s). Usually 1-5 tomograms are sufficient.
        :param output_star: (tomograms.star) star file similar to that from "relion". You can modify this file manually or with gui.
        :param pixel_size: (10) pixel size in angstroms. Usually you want to bin your tomograms to about 10A pixel size.
        Too large or too small pixel sizes are not recommended, since the target resolution on Z-axis of corrected tomograms should be about 30A.
        :param defocus: (0.0) defocus in Angstrom. Only need for ctf deconvolution. For phase plate data, you can leave defocus 0.
        If you have multiple tomograms with different defocus, please modify them in star file or with gui.
        :param number_subtomos: (100) Number of subtomograms to be extracted in later processes.
        If you want to extract different number of subtomograms in different tomograms, you can modify them in the star file generated with this command or with gui.

        这个命令从一个只包含层析重建图像文件(.mrc或.rec)的文件夹中生成一个tomograms.star文件。

        isonet.py prepare_star folder_name [--output_star] [--pixel_size] [--defocus] [--number_subtomos]

        :param folder_name: (None) 包含层析重建图像的目录。通常1-5个层析重建图像就足够了。
        :param output_star: (tomograms.star) 生成类似于Relion输出的star文件。您可以手动或通过gui修改这个文件。
        :param pixel_size: (10) 以埃(Angstrom)为单位的像素大小。通常您会想要将层析重建图像bin到大约10埃的像素大小。
        不推荐使用太大或太小的像素大小,因为经过校正的层析重建图像在Z轴上的目标分辨率应该约为30埃。
        :param defocus: (0.0) 以埃为单位 的缺陷。只有在进行ctf去卷积时才需要。对于相位板数据,可以将缺陷值设为0。
        如果您有多个具有不同缺陷值的层析重建图像,请在生成的star文件中或通过gui进行修改。
        :param number_subtomos: (100) 在后续过程中要提取的子层析重建图像数量。
        如果您希望在不同的层析重建图像中提取不同数量的子层析重建图像,您可以在使用此命令生成的star文件中或通过gui进行修改。
        """
        print(folder_name)
        md = MetaData()
        md.addLabels('rlnIndex','rlnMicrographName','rlnPixelSize','rlnDefocus','rlnNumberSubtomo','rlnMaskBoundary')
        tomo_list = sorted(os.listdir(folder_name))
        i = 0
        # 遍历 tomogram 文件
        for tomo in tomo_list:
            if tomo[-4:] == '.rec' or tomo[-4:] == '.mrc':
                i+=1
                it = Item()
                md.addItem(it)
                md._setItemValue(it,Label('rlnIndex'),str(i)) # 编号
                md._setItemValue(it,Label('rlnMicrographName'),os.path.join(folder_name,tomo)) #数据集地址
                md._setItemValue(it,Label('rlnPixelSize'),pixel_size) #
                md._setItemValue(it,Label('rlnDefocus'),defocus) #
                md._setItemValue(it,Label('rlnNumberSubtomo'),number_subtomos) #
                md._setItemValue(it,Label('rlnMaskBoundary'),None) #
        # 将元数据写入输出 STAR 文件
        md.write(output_star)
    #不知道有什么作用
    def prepare_subtomo_star(self, folder_name, output_star='subtomo.star', pixel_size: float=10.0, cube_size = None):
        """
        \nThis command generates a subtomo star file from a folder containing only subtomogram files (.mrc).
        This command is usually not necessary in the traditional workflow, because "isonet.py extract" will generate this subtomo.star for you.\n
        isonet.py prepare_subtomo_star folder_name [--output_star] [--cube_size]
        :param folder_name: (None) directory containing subtomogram(s).
        :param output_star: (subtomo.star) output star file for subtomograms, will be used as input in refinement.
        :param pixel_size: (10) The pixel size in angstrom of your subtomograms.
        :param cube_size: (None) This is the size of the cubic volumes used for training. This values should be smaller than the size of subtomogram.
        And the cube_size should be divisible by 8. If this value isn't set, cube_size is automatically determined as int(subtomo_size / 1.5 + 1)//16 * 16

        这个命令从一个只包含子层析重建图像文件(.mrc)的文件夹中生成一个subtomo.star文件。
        在传统工作流程中,通常不需要这个命令,因为"isonet.py extract"会为您生成这个subtomo.star文件。

        isonet.py prepare_subtomo_star folder_name [--output_star] [--cube_size]

        :param folder_name: (None) 包含子层析重建图像的目录。
        :param output_star: (subtomo.star) 子层析重建图像的输出星文件,将被用作refinement的输入。
        :param pixel_size: (10) 您的子层析重建图像的像素大小(以埃为单位)。
        :param cube_size: (None) 这是用于训练的立方体体素大小。这个值应该小于子层析重建图像的大小。
        并且cube_size应该能被8整除。如果没有设置这个值,cube_size会自动确定为int(subtomo_size / 1.5 + 1)//16 * 16。
        """
        #TODO check folder valid, logging
        # 检查文件夹是否存在
        if not os.path.isdir(folder_name):
            print("the folder does not exist")
        import mrcfile
        md = MetaData()
        md.addLabels('rlnSubtomoIndex','rlnImageName','rlnCubeSize','rlnCropSize','rlnPixelSize')
        subtomo_list = sorted(os.listdir(folder_name))
        # 遍历子层析重建图像文件
        for i,subtomo in enumerate(subtomo_list):
            subtomo_name = os.path.join(folder_name,subtomo)
            try:
                with mrcfile.open(subtomo_name, mode='r', permissive=True) as s:
                    crop_size = s.header.nx
            except:
                print("Warning: Can not process the subtomogram: {}!".format(subtomo_name))
                continue
            # 处理 cube_size
            if cube_size is not None:
                cube_size = int(cube_size)
                if cube_size >= crop_size:
                    cube_size = int(crop_size / 1.5 + 1)//16 * 16
                    print("Warning: Cube size should be smaller than the size of subtomogram volume! Using cube size {}!".format(cube_size))
                    #警告：立方体大小应小于次体积的大小！使用立方体大小{}
            else:
                cube_size = int(crop_size / 1.5 + 1)//16 * 16
            # 创建 Item 对象并设置各个属性值
            it = Item()
            md.addItem(it)
            md._setItemValue(it,Label('rlnSubtomoIndex'),str(i+1))
            md._setItemValue(it,Label('rlnImageName'),subtomo_name)
            md._setItemValue(it,Label('rlnCubeSize'),cube_size)
            md._setItemValue(it,Label('rlnCropSize'),crop_size)
            md._setItemValue(it,Label('rlnPixelSize'),pixel_size)

            # f.write(str(i+1)+' ' + os.path.join(folder_name,tomo) + '\n')
        # 将元数据写入输出 STAR 文件
        md.write(output_star)
    #进行把图片进行卷积处理,使图片变清晰
    def deconv(self, star_file: str,
        deconv_folder:str="./deconv",
        voltage: float=300.0,
        cs: float=2.7,
        snrfalloff: float=None,
        deconvstrength: float=None,
        highpassnyquist: float=0.02,
        chunk_size: int=None,
        overlap_rate: float= 0.25,
        ncpu:int=4,
        tomo_idx: str=None):
        """
        \nCTF deconvolution for the tomograms.\n
        isonet.py deconv star_file [--deconv_folder] [--snrfalloff] [--deconvstrength] [--highpassnyquist] [--overlap_rate] [--ncpu] [--tomo_idx]
        This step is recommended because it enhances low resolution information for a better contrast. No need to do deconvolution for phase plate data.
        :param deconv_folder: (./deconv) Folder created to save deconvoluted tomograms.
        :param star_file: (None) Star file for tomograms.
        :param voltage: (300.0) Acceleration voltage in kV.
        :param cs: (2.7) Spherical aberration in mm.
        :param snrfalloff: (1.0) SNR fall rate with the frequency. High values means losing more high frequency.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 1.0 will be used.
        :param deconvstrength: (1.0) Strength of the deconvolution.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 1.0 will be used.
        :param highpassnyquist: (0.02) Highpass filter for at very low frequency. We suggest to keep this default value.
        :param chunk_size: (None) When your computer has enough memory, please keep the chunk_size as the default value: None . Otherwise, you can let the program crop the tomogram into multiple chunks for multiprocessing and assembly them into one. The chunk_size defines the size of individual chunk. This option may induce artifacts along edges of chunks. When that happen, you may use larger overlap_rate.
        :param overlap_rate: (None) The overlapping rate for adjecent chunks.
        :param ncpu: (4) Number of cpus to use.
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16

        这个命令用于对层析重建图像进行CTF去卷积。

        isonet.py deconv star_file [--deconv_folder] [--snrfalloff] [--deconvstrength] [--highpassnyquist] [--overlap_rate] [--ncpu] [--tomo_idx]

        执行这个步骤是推荐的,因为它可以增强低分辨率信息,从而获得更好的对比度。对于相位板数据,无需执行去卷积。

        :param deconv_folder: (./deconv) 用于保存去卷积后的层析重建图像的文件夹。
        :param star_file: (None) 层析重建图像的星文件。
        :param voltage: (300.0) 加速电压,单位kV。
        :param cs: (2.7) 球面像差,单位mm。
        :param snrfalloff: (1.0) 信噪比随频率下降的速率。值越高,意味着丢失更多高频信息。
        如果没有设置这个值,程序将尝试从星文件中查找这个参数。
        如果没有设置并且星文件中也没有找到,将使用默认值1.0。
        :param deconvstrength: (1.0) 去卷积的强度。
        如果没有设置这个值,程序将尝试从星文件中查找这个参数。
        如果没有设置并且星文件中也没有找到,将使用默认值1.0。
        :param highpassnyquist: (0.02) 针对很低频率的高通滤波器。我们建议保留这个默认值。
        :param chunk_size: (None) 当您的计算机有足够的内存时,请保留chunk_size为默认值None。否则,您可以让程序将层析重建图像分割为多个块进行多进程处理,然后将它们组装成一个。chunk_size定义了每个块的大小。这个选项可能会在块的边缘产生伪影。如果发生这种情况,您可以使用更大的overlap_rate。
        :param overlap_rate: (None) 相邻块之间的重叠率。
        :param ncpu: (4) 使用的CPU数量。
        :param tomo_idx: (None) 如果设置了这个值,将只处理列在这个索引中的层析重建图像。例如1,2,4或5-10,15,16。
        该命令用于对层析重建图像进行CTF去卷积,以增强低分辨率信息,提高图像对比度。它适用于非相位板数据,相位板数据无需执行去卷积。

        主要参数解释如下:

        deconv_folder: 存储去卷积后层析重建图像的文件夹路径。
        star_file: 层析重建图像的星文件路径。
        一些CTF参数如voltage(加速电压)、cs(球面像差)、snrfalloff(信噪比下降率)、deconvstrength(去卷积强度)等,可从星文件读取或手动设置。
        highpassnyquist: 很低频率下的高通滤波器参数,建议使用默认值。
        chunk_size和overlap_rate: 用于内存不足时分块处理层析重建图像的参数。
        ncpu: 使用的CPU数量。
        tomo_idx: 指定只处理某些层析重建图像的索引。
        总的来说,这个命令对层析重建图像执行了去卷积处理,以提高图像质量,为后续的训练做准备。
        """
        from IsoNet.util.deconvolution import deconv_one
        # 设置日志格式和级别
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        logging.info('\n######Isonet starts ctf deconvolve######\n')
        try:  # 读取元数据
            md = MetaData()
            md.read(star_file)
            if not 'rlnSnrFalloff' in md.getLabels(): # 添加缺失的标签
                md.addLabels('rlnSnrFalloff','rlnDeconvStrength','rlnDeconvTomoName')
                for it in md:
                    md._setItemValue(it,Label('rlnSnrFalloff'),1.0)
                    md._setItemValue(it,Label('rlnDeconvStrength'),1.0)
                    md._setItemValue(it,Label('rlnDeconvTomoName'),None)
            # 创建输出文件夹./deconv
            if not os.path.isdir(deconv_folder)                                                                                                                                         :
                os.mkdir(deconv_folder)

            tomo_idx = idx2list(tomo_idx)
            for it in md: # 处理每个项目
                #这里可以看到TSO1-wbp.rec,TS43-wbp.rec,TS45-wbp.rec 分别1 2 3 tomo_idx编号
                if tomo_idx is None or str(it.rlnIndex) in tomo_idx:
                    if snrfalloff is not None:
                        md._setItemValue(it,Label('rlnSnrFalloff'), snrfalloff)
                    if deconvstrength is not None:
                        md._setItemValue(it,Label('rlnDeconvStrength'),deconvstrength)

                    tomo_file = it.rlnMicrographName
                    base_name = os.path.basename(tomo_file)
                    deconv_tomo_name = '{}/{}'.format(deconv_folder,base_name)
                    # 执行去卷积
                    deconv_one(it.rlnMicrographName,deconv_tomo_name,voltage=voltage,cs=cs,defocus=it.rlnDefocus/10000.0, pixel_size=it.rlnPixelSize,snrfalloff=it.rlnSnrFalloff, deconvstrength=it.rlnDeconvStrength,highpassnyquist=highpassnyquist,chunk_size=chunk_size,overlap_rate=overlap_rate,ncpu=ncpu)
                    # 更新元数据中的去卷积文件名
                    md._setItemValue(it,Label('rlnDeconvTomoName'),deconv_tomo_name)
                md.write(star_file) # 写入更新后的元数据 将deconv处理过后.rec .mrc文件地址加入到tomograms.star中
            # 输出完成信息
            logging.info('\n######Isonet done ctf deconvolve######\n')

        except Exception:
            # 出现异常时记录错误信息
            error_text = traceback.format_exc()
            f =open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)
    #增加一个mask掩盖 突出我要图片的特征值
    def make_mask(self,star_file,
                mask_folder: str = 'mask',
                patch_size: int=4,
                mask_boundary: str=None,
                density_percentage: int=None,
                std_percentage: int=None,
                use_deconv_tomo:bool=True,
                z_crop:float=None,
                tomo_idx=None):
        """
        \ngenerate a mask that include sample area and exclude "empty" area of the tomogram. The masks do not need to be precise. In general, the number of subtomograms (a value in star file) should be lesser if you masked out larger area. \n
        isonet.py make_mask star_file [--mask_folder] [--patch_size] [--density_percentage] [--std_percentage] [--use_deconv_tomo] [--tomo_idx]
        :param star_file: path to the tomogram or tomogram folder
        :param mask_folder: path and name of the mask to save as
        :param patch_size: (4) The size of the box from which the max-filter and std-filter are calculated.
        :param density_percentage: (50) The approximate percentage of pixels to keep based on their local pixel density.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 50 will be used.
        :param std_percentage: (50) The approximate percentage of pixels to keep based on their local standard deviation.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 50 will be used.
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
        :param z_crop: If exclude the top and bottom regions of tomograms along z axis. For example, "--z_crop 0.2" will mask out the top 20% and bottom 20% region along z axis.
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
        生成一个包含样本区域并排除层析重建图像中"空白"区域的掩码。这些掩码不需要非常精确。一般来说,如果你遮掩了更大的区域,star文件中的子层析重建图像数量(a value in star file)应该更少。

        isonet.py make_mask star_file [--mask_folder] [--patch_size] [--density_percentage] [--std_percentage] [--use_deconv_tomo] [--tomo_idx]

        :param star_file: 层析重建图像或层析重建图像文件夹的路径
        :param mask_folder: 用于保存掩码的路径和名称
        :param patch_size: (4) 计算最大滤波和标准差滤波的框的大小。
        :param density_percentage: (50) 根据局部像素密度保留的近似像素百分比。
        如果没有设置这个值,程序将尝试从星文件中查找这个参数。
        如果没有设置并且星文件中也没有找到,将使用默认值50。
        :param std_percentage: (50) 根据局部标准差保留的近似像素百分比。
        如果没有设置这个值,程序将尝试从星文件中查找这个参数。
        如果没有设置并且星文件中也没有找到,将使用默认值50。
        :param use_deconv_tomo: (True) 如果在tomogram.star中找到了CTF去卷积的层析重建图像,则使用该层析重建图像。
        :param z_crop: 如果排除层析重建图像在z轴上的顶部和底部区域。例如,"--z_crop 0.2"将遮盖z轴上顶部20%和底部20%的区域。
        :param tomo_idx: (None) 如果设置了这个值,将只处理列在这个索引中的层析重建图像。例如1,2,4或5-10,15,16。
        """
        from IsoNet.bin.make_mask import make_mask
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        logging.info('\n######Isonet starts making mask######\n')
        try:
            # 判断是否有mask文件夹
            if not os.path.isdir(mask_folder):
                os.mkdir(mask_folder)
            # write star percentile threshold
            # 读取tomograms.star
            md = MetaData()
            md.read(star_file)

            if not 'rlnMaskDensityPercentage' in md.getLabels():
                md.addLabels('rlnMaskDensityPercentage','rlnMaskStdPercentage','rlnMaskName')
                for it in md:
                    md._setItemValue(it,Label('rlnMaskDensityPercentage'),50)
                    md._setItemValue(it,Label('rlnMaskStdPercentage'),50)
                    md._setItemValue(it,Label('rlnMaskName'),None)
            #读取tomograms.star文件中相关参数
            tomo_idx = idx2list(tomo_idx)
            for it in md:
                if tomo_idx is None or str(it.rlnIndex) in tomo_idx:
                    if density_percentage is not None:
                        md._setItemValue(it,Label('rlnMaskDensityPercentage'),density_percentage)
                    if std_percentage is not None:
                        md._setItemValue(it,Label('rlnMaskStdPercentage'),std_percentage)
                    if use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels() and it.rlnDeconvTomoName not in [None,'None']:
                        tomo_file = it.rlnDeconvTomoName
                    else:
                        tomo_file = it.rlnMicrographName
                    tomo_root_name = os.path.splitext(os.path.basename(tomo_file))[0]

                    if os.path.isfile(tomo_file):
                        logging.info('make_mask: {}| dir_to_save: {}| percentage: {}| window_scale: {}'.format(tomo_file,
                        mask_folder, it.rlnMaskDensityPercentage, patch_size))
                        
                        #if mask_boundary is None:
                        if "rlnMaskBoundary" in md.getLabels() and it.rlnMaskBoundary not in [None, "None"]:
                            mask_boundary = it.rlnMaskBoundary 
                        else:
                            mask_boundary = None
                              
                        mask_out_name = '{}/{}_mask.mrc'.format(mask_folder,tomo_root_name)
                        #执行蒙版操作
                        make_mask(tomo_file,
                                mask_out_name,
                                mask_boundary=mask_boundary,
                                side=patch_size,
                                density_percentage=it.rlnMaskDensityPercentage,
                                std_percentage=it.rlnMaskStdPercentage,
                                surface = z_crop)

                    md._setItemValue(it,Label('rlnMaskName'),mask_out_name)
                md.write(star_file)
            logging.info('\n######Isonet done making mask######\n')
        except Exception:
            error_text = traceback.format_exc()
            f =open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)
    #提取数据
    def extract(self,
        star_file: str,
        use_deconv_tomo: bool = True,
        subtomo_folder: str = "subtomo",
        subtomo_star: str = "subtomo.star",
        cube_size: int = 64,
        crop_size: int = None,
        log_level: str="info",
        tomo_idx = None
        ):

        """
        \nExtract subtomograms\n
        isonet.py extract star_file [--subtomo_folder] [--subtomo_star] [--cube_size] [--use_deconv_tomo] [--tomo_idx]
        :param star_file: tomogram star file
        :param subtomo_folder: (subtomo) folder for output subtomograms.
        :param subtomo_star: (subtomo.star) star file for output subtomograms.
        :param cube_size: (64) Size of cubes for training, should be divisible by 8, eg. 32, 64. The actual sizes of extracted subtomograms are this value adds 16.
        :param crop_size: (None) The size of subtomogram, should be larger then the cube_size The default value is 16+cube_size.
        :param log_level: ("info") level of the output, either "info" or "debug"
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.

        提取子层析重建图像
        isonet.py extract star_file [--subtomo_folder] [--subtomo_star] [--cube_size] [--use_deconv_tomo] [--tomo_idx]
        :param star_file: 层析重建图像的星文件
        :param subtomo_folder: (subtomo) 输出子层析重建图像的文件夹。
        :param subtomo_star: (subtomo.star) 输出子层析重建图像的星文件。
        :param cube_size: (64) 用于训练的立方体大小,应该能被8整除,例如32、64。实际提取的子层析重建图像大小将是这个值加16。
        :param crop_size: (None) 子层析重建图像的大小,应该大于cube_size。默认值为16+cube_size。
        :param log_level: ("info") 输出的日志级别,可以是"info"或"debug"。
        :param use_deconv_tomo: (True) 如果在层析重建图像的星文件中找到了CTF去卷积的层析重建图像,则使用该层析重建图像。
        """
        d = locals() # 将变量和对应的值做成字典赋值给d
        d_args = Arg(d)
        print("d_args",d_args)
        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])

        logging.info("\n######Isonet starts extracting subtomograms######\n")

        try:
            if os.path.isdir(subtomo_folder):
                logging.warning("subtomo directory exists, the current directory will be overwritten")
                import shutil
                shutil.rmtree(subtomo_folder)#删除subtomo文件夹
            os.mkdir(subtomo_folder)

            from IsoNet.preprocessing.prepare import extract_subtomos
            if crop_size is None:
                d_args.crop_size = cube_size + 16
            else:
                d_args.crop_size = crop_size
            d_args.subtomo_dir = subtomo_folder
            d_args.tomo_idx = idx2list(tomo_idx)
            extract_subtomos(d_args)
            logging.info("\n######Isonet done extracting subtomograms######\n")
        except Exception:
            error_text = traceback.format_exc()
            f = open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)

    #将模型进行反复迭代,更新模型。
    def refine(self,
        subtomo_star: str,
        gpuID: str = None,
        iterations: int = None,
        data_dir: str = None,
        other_data_dir: str = None,

        pretrained_model: str = None,
        log_level: str = None,
        result_dir: str='results',
        other_result_dir: str='other_results',
        remove_intermediate: bool =False,
        select_subtomo_number: int = None,
        preprocessing_ncpus: int = 16,
        continue_from: str=None,
        epochs: int = 10,
        batch_size: int = None,
        steps_per_epoch: int = None,

        noise_level:  tuple=(0.05,0.10,0.15,0.20),
        noise_start_iter: tuple=(11,16,21,26),
        noise_mode: str = None,
        noise_dir: str = None,
        other_noise_dir: str = None,
        learning_rate: float = None,
        drop_out: float = 0.3,
        convs_per_depth: int = 3,
        kernel: tuple = (3,3,3),
        pool: tuple = None,
        unet_depth: int = 3,
        filter_base: int = None,
        batch_normalization: bool = True,
        normalize_percentile: bool = True,

        use_unet: bool = True,
        use_other_unet: bool = True
    ):
        """
        \ntrain neural network to correct missing wedge\n
        isonet.py refine subtomo_star [--iterations] [--gpuID] [--preprocessing_ncpus] [--batch_size] [--steps_per_epoch] [--noise_start_iter] [--noise_level]...
        :param subtomo_star: (None) star file containing subtomogram(s).
        :param gpuID: (0,1,2,3) The ID of gpu to be used during the training. e.g 0,1,2,3.
        :param pretrained_model: (None) A trained neural network model in ".h5" format to start with.
        :param iterations: (30) Number of training iterations.
        :param data_dir: (data) Temporary folder to save the generated data used for training.
        :param log_level: (info) debug level, could be 'info' or 'debug'
        :param continue_from: (None) A Json file to continue from. That json file is generated at each iteration of refine.
        :param result_dir: ('results') The name of directory to save refined neural network models and subtomograms
        :param preprocessing_ncpus: (16) Number of cpu for preprocessing.

        ************************Training settings************************

        :param epochs: (10) Number of epoch for each iteraction.
        :param batch_size: (None) Size of the minibatch.If None, batch_size will be the max(2 * number_of_gpu,4). batch_size should be divisible by the number of gpu.
        :param steps_per_epoch: (None) Step per epoch. If not defined, the default value will be min(num_of_subtomograms * 6 / batch_size , 200)

        ************************Denoise settings************************

        :param noise_level: (0.05,0.1,0.15,0.2) Level of noise STD(added noise)/STD(data) after the iteration defined in noise_start_iter.
        :param noise_start_iter: (11,16,21,26) Iteration that start to add noise of corresponding noise level.
        :param noise_mode: (None) Filter names when generating noise volumes, can be 'ramp', 'hamming' and 'noFilter'
        :param noise_dir: (None) Directory for generated noise volumes. If set to None, the Noise volumes should appear in results/training_noise

        ************************Network settings************************

        :param drop_out: (0.3) Drop out rate to reduce overfitting.
        :param learning_rate: (0.0004) learning rate for network training.
        :param convs_per_depth: (3) Number of convolution layer for each depth.
        :param kernel: (3,3,3) Kernel for convolution
        :param unet_depth: (3) Depth of UNet.
        :param filter_base: (64) The base number of channels after convolution.
        :param batch_normalization: (True) Use Batch Normalization layer
        :param pool: (False) Use pooling layer instead of stride convolution layer.
        :param normalize_percentile: (True) Normalize the 5 percent and 95 percent pixel intensity to 0 and 1 respectively. If this is set to False, normalize the input to 0 mean and 1 standard dievation.
        :param use_unet: (True)
        :param use_swim_unet: (True)
        """
        """
        训练神经网络以校正缺失楔区

        isonet.py refine subtomo_star [--iterations] [--gpuID] [--preprocessing_ncpus] [--batch_size] [--steps_per_epoch] [--noise_start_iter] [--noise_level]...

        :param subtomo_star: (None) 包含子层析重建图像的星文件。
        :param gpuID: (0,1,2,3) 训练期间使用的GPU ID,例如0,1,2,3。
        :param pretrained_model: (None) 以".h5"格式提供一个训练好的神经网络模型作为起点。
        :param iterations: (30) 训练迭代的次数。
        :param data_dir: (data) 用于保存训练数据的临时文件夹。
        :param log_level: (info) 调试级别,可以是'info'或'debug'。
        :param continue_from: (None) 一个Json文件,用于从中继续训练。该Json文件在每次迭代时生成。
        :param result_dir: ('results') 保存精炼的神经网络模型和子层析重建图像的目录名称。
        :param preprocessing_ncpus: (16) 用于预处理的CPU数量。
        
        ************************训练设置************************
        
        :param epochs: (10) 每次迭代的训练世代数。
        :param batch_size: (None) 小批量的大小。如果为None,batch_size将是max(2 * gpu数量,4)。batch_size应该能被GPU数量整除。
        :param steps_per_epoch: (None) 每个世代的步数。如果未定义,默认值将是min(子层析重建图像数量*6/batch_size,200)。
        
        ************************去噪设置************************
        
        :param noise_level: (0.05,0.1,0.15,0.2) 在由noise_start_iter定义的迭代后,添加噪声的水平STD(添加噪声)/STD(数据)。
        :param noise_start_iter: (11,16,21,26) 开始添加相应噪声水平的迭代。
        :param noise_mode: (None) 生成噪声体积时使用的滤波器名称,可以是'ramp'、'hamming'和'noFilter'。
        :param noise_dir: (None) 生成噪声体积的目录。如果设置为None,噪声体积应该出现在results/training_noise中。
        
        ************************网络设置************************
        
        :param drop_out: (0.3) dropout率,用于减少过拟合。
        :param learning_rate: (0.0004) 网络训练的学习率。
        :param convs_per_depth: (3) 每个深度的卷积层数量。
        :param kernel: (3,3,3) 卷积核大小。
        :param unet_depth: (3) UNet的深度。
        :param filter_base: (64) 卷积后的基础通道数。
        :param batch_normalization: (True) 是否使用批量标准化层。
        :param pool: (False) 是否使用池化层代替步长卷积层。
        :param normalize_percentile: (True) 将5%和95%像素强度分别标准化为0和1。如果设置为False,则将输入标准化为0均值和1标准差。
        """
        from IsoNet.bin.refine import run
        d = locals()
        d_args = Arg(d)
        with open('log.txt','a+') as f:
            f.write(' '.join(sys.argv[0:]) + '\n')
        run(d_args)
    #进行预测
    def predict(self, star_file: str, model: str, output_dir: str='./corrected_tomos', gpuID: str = None, cube_size:int=64,
    crop_size:int=96,use_deconv_tomo=True, batch_size:int=None,normalize_percentile: bool=True,log_level: str="info", tomo_idx=None):
        """
        \nPredict tomograms using trained model\n
        isonet.py predict star_file model [--gpuID] [--output_dir] [--cube_size] [--crop_size] [--batch_size] [--tomo_idx]
        :param star_file: star for tomograms.
        :param output_dir: file_name of output predicted tomograms
        :param model: path to trained network model .h5
        :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
        :param cube_size: (64) The tomogram is divided into cubes to predict due to the memory limitation of GPUs.
        :param crop_size: (96) The side-length of cubes cropping from tomogram in an overlapping patch strategy, make this value larger if you see the patchy artifacts
        :param batch_size: The batch size of the cubes grouped into for network predicting, the default parameter is four times number of gpu
        :param normalize_percentile: (True) if normalize the tomograms by percentile. Should be the same with that in refine parameter.
        :param log_level: ("debug") level of message to be displayed, could be 'info' or 'debug'
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
        :raises: AttributeError, KeyError
        """
        """
        使用训练好的模型预测层析重建图像

        isonet.py predict star_file model [--gpuID] [--output_dir] [--cube_size] [--crop_size] [--batch_size] [--tomo_idx]
        
        :param star_file: 层析重建图像的星文件。
        :param output_dir: 输出预测层析重建图像的文件名。
        :param model: 训练好的神经网络模型路径(.h5文件)。
        :param gpuID: (0,1,2,3) 用于预测的GPU ID,例如0,1,2,3。
        :param cube_size: (64) 由于GPU内存限制,层析重建图像被分割成立方体进行预测。
        :param crop_size: (96) 从层析重建图像中裁剪出带有重叠的立方体的边长,如果看到分块伪影,可以增大此值。
        :param batch_size: 网络预测时分组的立方体批量大小,默认为GPU数量的4倍。
        :param normalize_percentile: (True) 是否按百分位数对层析重建图像进行标准化。应与refine参数中的设置相同。
        :param log_level: ("debug") 要显示的消息级别,可以是"info"或"debug"。
        :param tomo_idx: (None) 如果设置了该值,仅处理列在该索引中的层析重建图像,例如1,2,4或5-10,15,16。
        :param use_deconv_tomo: (True) 如果在层析重建图像星文件中找到了CTF去卷积的层析重建图像,则使用该层析重建图像。
        :raises: AttributeError, KeyError
        """
        d = locals()
        d_args = Arg(d)
        from IsoNet.bin.predict import predict

        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
            datefmt="%m-%d %H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
            datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        try:
            predict(d_args)
        except:
            error_text = traceback.format_exc()
            f =open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)
    #不知道有什么作用
    def resize(self, star_file:str, apix: float=15, out_folder="tomograms_resized"):
        '''
        This function rescale the tomograms to a given pixelsize
        这个函数将层析图重缩放到指定的像素大小。
        '''
        # 导入所需的库
        md = MetaData()     # 实例化 MetaData 类
        md.read(star_file)  # 读取 STAR 文件中的元数据信息
        #print(md._data[0].rlnPixelSize)
        from scipy.ndimage import zoom # 导入 zoom 函数用于调整数据大小
        #from skimage.transform import rescale
        #import numpy as np
        import mrcfile # 导入 mrcfile 库用于处理 MRC 文件
        if not os.path.isdir(out_folder): # 如果输出文件夹不存在，则创建
            os.makedirs(out_folder)
        # 遍历每个数据项
        for item in md._data:
            ori_apix = item.rlnPixelSize # 获取原始像素大小
            tomo_name = item.rlnMicrographName # 获取层析图文件名
            zoom_factor = float(ori_apix)/apix # 计算缩放系数
            new_tomo_name = "{}/{}".format(out_folder,os.path.basename(tomo_name)) # 构建新的层析图文件名
            with mrcfile.open(tomo_name, permissive=True) as mrc: # 使用 mrcfile 打开原始层析图文件
                data = mrc.data  # 读取数据
            print("scaling: {}".format(tomo_name))
            # 对数据进行缩放处理
            new_data = zoom(data, zoom_factor,order=3, prefilter=False)
            #new_data = rescale(data, zoom_factor,order=3, anti_aliasing = True)
            #new_data = new_data.astype(np.float32)

            # 将处理后的数据保存为新的 MRC 文件
            with mrcfile.new(new_tomo_name,overwrite=True) as mrc:
                mrc.set_data(new_data)
                mrc.voxel_size = apix

            # 更新元数据信息中的像素大小和文件名
            item.rlnPixelSize = apix
            print(new_tomo_name)
            item.rlnMicrographName = new_tomo_name
            print(item.rlnMicrographName)
        # 将修改后的元数据写回到 STAR 文件
        md.write(os.path.splitext(star_file)[0] + "_resized.star")
        print("scale_finished")

    def check(self):
        from IsoNet.bin.predict import predict
        from IsoNet.bin.refine import run
        import skimage
        import PyQt5
        import tqdm
        print('IsoNet --version 0.2 installed')
    #代开gui界面
    def gui(self):
        """
        \nGraphic User Interface\n
        """
        #打开gui界面
        import IsoNet.gui.Isonet_star_app as app
        app.main()

def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)
# 定义一个函数，利用进程池处理任务
def pool_process(p_func,chunks_list,ncpu):
    # 导入 multiprocessing 模块中的 Pool 类
    from multiprocessing import Pool
    # 创建进程池，包含 ncpu 个进程，每个进程执行 1000 个任务后销毁并重新创建
    with Pool(ncpu,maxtasksperchild=1000) as p:
        # 在进程池中执行任务
        # 这里可以添加具体的任务处理逻辑

        # results = p.map(partial_func,chunks_gpu_num_list,chunksize=1)
        results = list(p.map(p_func,chunks_list))
    # return results

if __name__ == "__main__":
    core.Display = Display
    # logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)
    if len(sys.argv) > 1:
        check_parse(sys.argv[1:])
    fire.Fire(ISONET)
    #
    # import IsoNet.gui.Isonet_star_app as app
    # app.main()




