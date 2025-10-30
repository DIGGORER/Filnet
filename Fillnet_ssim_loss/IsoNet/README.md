# Isotropic Reconstruction of Electron Tomograms with Deep Learning
# IsoNet version 0.2
[![DOI](https://zenodo.org/badge/222662248.svg)](https://zenodo.org/badge/latestdoi/222662248)
##
Update on July8 2022

We maintain an IsoNet Google group for discussions or news.

To subscribe or visit the group via the web interface please visit https://groups.google.com/u/1/g/isonet. 

If you do not have and are not willing to create a Google login, you can also request membership by sending an email to yuntao@g.ucla.edu

After your request is approved, you will be added to the group and will receive a confirmation email. Once subscribed, we request that you edit your membership settings so that your display name is your real name. 

To post to the forum you can either use the web interface or email to isonet@googlegroups.com


## Installation
python version at least 3.5 is required. If you download the package as a zip file from github, please rename the folder IsoNet-master to IsoNet.

1.  IsoNet relies on Tensorflow with version at least 2.0

Please find your cuda version, cuDNN version and corresponding tensorflow version here: https://www.tensorflow.org/install/source#gpu. 

For example, if you are using cuda 10.1, you should install tensorflow 2.3:
```
pip install tensorflow-gpu==2.3.0
```

2.  Install other dependencies

```
pip install -r requirements.txt
```
3.  Add environment variables: 

For example add following lines in your ~/.bashrc
```
export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 

export PYTHONPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH 
```

or you can run `source source-env.sh` in your terminal, which will export required variables into your environment.

4. Open a new terminal, enter your working directory and run 
```
isonet.py check
```

Tutorial data set and tutorial videos are on google drive https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp

# FAQ:
## 1. IsoNet refine raise OOM error.

This is caused by the insufficient GPU memory.
The solutions are:
1. Specify a smaller batch\_size or use more(powerful) GPUs. The default batch\_size is 4 if you use one GPU, otherwise the default batch\_size is 2 times the number of GPU. Please note the batch_size should be divisible by number of GPUs.
For example, if you have one GPU and get OOM error, please reduce the batch\_size to 1 or 2; If you use 4 GPUs and get OOM error, please reduce the batch\_size to 4.

2. Refine with a smaller cube\_size (not recommended).

## 2.  IsoNet extract ValueError: a must be greater than 0 unless no samples are taken
This could be due to the tomogram thickness is smaller than the size of subtomograms to be extracted. Please make your tomogram thicker in this case.

## 3. Can not see significant improvement after processing with IsoNet
IsoNet is kind of conservative in adding information into missing wedge region. If it can not find reasonable prediction, IsoNet may simply returns the original tomograms back to you. 
However, there are some ways to increase your success rate.
1. IsoNet performs better in high contrast tomograms. That means it will be helpful to tweak the parameters (especially snrfalloff) in CTF deconvolution step to make increase the weight of low resolution information. Or trying with the data acquired with phaseplate first. As far as we know, phaseplate data will always give you good result.

2. Missing wedge caused the nonlocal distributed information. You may observed the long shadows of gold beads in the tomograms, and those long rays can not be fully corrected with sub-tomogram based missing correction in IsoNet, because the receptive field of the network is limited to your subtomogram. This nonlocal information makes it particular difficult to recover the horizontal oriented membrane. There are several ways to improve. **First**, training with subtomograms with larger  cube size, the default cube size is 64, you may want to increase the size to 80, 96, 112 or 128, however this may lead to the OOM error Please refer to FAQ #1 when you have this problem. **Second**, bin your tomograms more. Sometimes we even bin our cellular tomograms to 20A/pix for IsoNet processing, this will of course increase your network receptive field, given the same size of subtomogram. 

3. IsoNet is currently designed to correct missing wedge for tomograms with -60 to 60 degrees tilt range. The other tilt scheme or when the tomograms have large x axis tilt. The results might not be optimal. 
## 4. Can not create a good mask during mask generation step
The mask is only important if the sample is sparsely located in the tomograms. And the mask do not need to be perfect to obtain good result, in other words, including many empty/unwanted subtomograms during the refinement can be tolerated. 

To obtain a good mask, the tomograms should have sufficient contrast, which can be achieved by CTF deconvolution. User defined mask can also be supplied by changing the mask_name field in the star file. Alternately, you can also use subtomograms extracted with other methods and skip the entire mask creation and subtomograms extraction steps.

If you want to exclude carbon area of the tomograms, you can try the new mask boundary feature in version 0.2. It allows you to draw a polygon in 3dmod so that the area outside the polygon will be excluded.

###########################################################################################################################################################

Update on July8 2022
我们设有一个 IsoNet 谷歌群组，用于讨论或发布新闻。

We maintain an IsoNet Google group for discussions or news.
要通过网络界面订阅或访问该群组，请访问 https://groups.google.com/u/1/g/iso。

To subscribe or visit the group via the web interface please visit https://groups.google.com/u/1/g/isonet.
要通过网络界面订阅或访问该小组，请访问 https://groups.google.com/u/1/g/isonet。

If you do not have and are not willing to create a Google login, you can also request membership by sending an email to yuntao@g.ucla.edu
如果您没有也不愿意创建 Google 登录，也可以发送电子邮件至 yuntao@g.ucla.edu 申请加入。

After your request is approved, you will be added to the group and will receive a confirmation email. Once subscribed, we request that you edit your membership settings so that your display name is your real name.
申请通过后，您将被添加到群组，并收到一封确认电子邮件。订阅后，我们要求您编辑会员设置，以便您的显示名是您的真实姓名。

To post to the forum you can either use the web interface or email to isonet@googlegroups.com
要在论坛上发帖，您可以使用网页界面或发送电子邮件至 isonet@googlegroups.com。

Installation安装
python version at least 3.5 is required. If you download the package as a zip file from github, please rename the folder IsoNet-master to IsoNet.
python 版本

IsoNet relies on Tensorflow with version at least 2.0
IsoNet 依赖于版本至少为 2.0 的 Tensorflow
Please find your cuda version, cuDNN version and corresponding tensorflow version here: https://www.tensorflow.org/install/source#gpu.
请在此处查找您的 cuda 版本、cuDNN 版本以及相应的 tensorflow 版本： https://www.tensorflow.org/install/source#gpu。

For example, if you are using cuda 10.1, you should install tensorflow 2.3:
例如，如果您使用的是 cuda 10.1，则应安装 tensorflow 2.3：

pip install tensorflow-gpu==2.3.0
Install other dependencies
安装其他依赖项
pip install -r requirements.txt
Add environment variables:
添加环境变量：
For example add following lines in your ~/.bashrc
例如，在 ~/.bashrc 中添加以下内容

export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 

export PYTHONPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH 
or you can run source source-env.sh in your terminal, which will export required variables into your environment.
或者在终端运行 source source-env.sh，将所需变量导出到环境中。

Open a new terminal, enter your working directory and run
打开新终端，输入工作目录并运行
isonet.py check
Tutorial data set and tutorial videos are on google drive https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp
教程数据集和教程视频在 google drive https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp 上。

FAQ:常见问题
1. IsoNet refine raise OOM error.1. IsoNet refine 引发 OOM 错误。
This is caused by the insufficient GPU memory. The solutions are:
这是由于 GPU 内存不足造成的。解决方法是

Specify a smaller batch_size or use more(powerful) GPUs. The default batch_size is 4 if you use one GPU, otherwise the default batch_size is 2 times the number of GPU. Please note the batch_size should be divisible by number of GPUs. For example, if you have one GPU and get OOM error, please reduce the batch_size to 1 or 2; If you use 4 GPUs and get OOM error, please reduce the batch_size to 4.
指定一个较小的批量大小或使用更多更强大的 GPU。如果使用一个 GPU，默认批处理大小为 4，否则默认批处理大小为 GPU 数量的 2 倍。请注意，批量大小应能被 GPU 数量整除。例如，如果您只有一个 GPU 并出现 OOM 错误，请将 batch_size 减小到 1 或 2；如果您使用 4 个 GPU 并出现 OOM 错误，请将 batch_size 减小到 4。

Refine with a smaller cube_size (not recommended).
使用较小的 cube_size（不建议）进行精炼。

2. IsoNet extract ValueError: a must be greater than 0 unless no samples are taken2. IsoNet extract ValueError: a must be greater than 0 unless no samples are taken（除非没有采集样本）。
This could be due to the tomogram thickness is smaller than the size of subtomograms to be extracted. Please make your tomogram thicker in this case.
这可能是由于断层扫描的厚度小于要提取的子断层扫描的大小。在这种情况下，请将断层扫描加厚。

3. Can not see significant improvement after processing with IsoNet3. 使用 IsoNet 处理后看不到明显改善
IsoNet is kind of conservative in adding information into missing wedge region. If it can not find reasonable prediction, IsoNet may simply returns the original tomograms back to you. However, there are some ways to increase your success rate.
IsoNet 在向缺失楔形区域添加信息时比较保守。如果找不到合理的预测，IsoNet 可能会简单地将原始断层图返回给您。不过，有一些方法可以提高成功率。

IsoNet performs better in high contrast tomograms. That means it will be helpful to tweak the parameters (especially snrfalloff) in CTF deconvolution step to make increase the weight of low resolution information. Or trying with the data acquired with phaseplate first. As far as we know, phaseplate data will always give you good result.
IsoNet 在高对比度断层扫描中表现更好。这意味着，在 CTF 解卷积步骤中调整参数（尤其是 snrfalloff）以增加低分辨率信息的权重会很有帮助。或者先用相位板获取的数据进行尝试。据我们所知，相位板数据总是会给你带来好的结果。

Missing wedge caused the nonlocal distributed information. You may observed the long shadows of gold beads in the tomograms, and those long rays can not be fully corrected with sub-tomogram based missing correction in IsoNet, because the receptive field of the network is limited to your subtomogram. This nonlocal information makes it particular difficult to recover the horizontal oriented membrane. There are several ways to improve. First, training with subtomograms with larger cube size, the default cube size is 64, you may want to increase the size to 80, 96, 112 or 128, however this may lead to the OOM error Please refer to FAQ #1 when you have this problem. Second, bin your tomograms more. Sometimes we even bin our cellular tomograms to 20A/pix for IsoNet processing, this will of course increase your network receptive field, given the same size of subtomogram.
楔形缺失会导致非局部分布信息。您可能会在层析成像图中观察到金珠的长条阴影，而这些长条射线并不能完全对应于楔块。

IsoNet is currently designed to correct missing wedge for tomograms with -60 to 60 degrees tilt range. The other tilt scheme or when the tomograms have large x axis tilt. The results might not be optimal.
IsoNet 目前的设计是纠正倾斜范围为 -60 至 60 度的断层图像的楔形缺失。其他倾斜方案或当断层扫描有较大的 x 轴倾斜时。结果可能并不理想。

4. Can not create a good mask during mask generation step4. 在掩膜生成步骤中无法创建良好的掩膜
The mask is only important if the sample is sparsely located in the tomograms. And the mask do not need to be perfect to obtain good result, in other words, including many empty/unwanted subtomograms during the refinement can be tolerated.
只有当样本稀疏地分布在断层图中时，掩膜才会变得重要。要获得良好的结果，掩膜不需要很完美，换句话说，在细化过程中包含许多空的/不需要的子断层图是可以容忍的。

To obtain a good mask, the tomograms should have sufficient contrast, which can be achieved by CTF deconvolution. User defined mask can also be supplied by changing the mask_name field in the star file. Alternately, you can also use subtomograms extracted with other methods and skip the entire mask creation and subtomograms extraction steps.
要获得良好的掩码，断层扫描图应具有足够的对比度，这可以通过 CTF 解卷积来实现。也可以通过更改星形文件中的 mask_name 字段来提供用户定义的掩膜。另外，也可以使用其他方法提取的子图，跳过整个掩膜创建和子图提取步骤。

If you want to exclude carbon area of the tomograms, you can try the new mask boundary feature in version 0.2. It allows you to draw a polygon in 3dmod so that the area outside the polygon will be excluded.
如果您想排除层析成像的碳区域，可以试试 0.2 版的新掩膜边界功能。通过该功能，您可以在 3dmod 中绘制一个多边形，这样多边形以外的区域将被排除在外。