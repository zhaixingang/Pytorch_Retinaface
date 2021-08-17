# RetinaFace in PyTorch self

## 创建docker环境

在dgx1（10.8.1.12）上运行docker命令如下：

```
nvidia-docker run --name tomzhai_pytorcj_gpu -it -v /data:/data -v /dcache:/dcache -p 10094:22 pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel /bin/bash
```

创建出来一个tomzhai_pytorcj_gpu容器，方便ssh远程登陆查看等等。进入容器，查看pytorch版本以及是否使用gpu等等。

```
root@1dc0653f1e3f:/workspace# python
Python 3.7.7 (default, May  7 2020, 21:25:33)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'Tesla V100-SXM2-32GB'
>>>
```

后续我们将ubuntu的源换为阿里云的源，方便快速更新下载东西等。

```
cp /etc/apt/sources.list /etc/apt/sources.list.bak
vi /etc/apt/sources.list
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
apt-get update
apt-get upgrade
```

接下来安装ssh相关

```
apt-get install openssh-client
apt-get install openssh-server
```

查看ssh是否启动

```
root@1dc0653f1e3f:/workspace# /etc/init.d/ssh start
 * Starting OpenBSD Secure Shell server sshd
 
 root@1dc0653f1e3f:/workspace# ps -e|grep ssh
 8667 ?        00:00:00 sshd
```

编辑sshd_config文件

```
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
vim /etc/ssh/sshd_config
PermitRootLogin yes
```

重启ssh service

```
service ssh restart
```

设置root的password

```
passwd root
```

使用pycharm打开这个工程，并设置解释器和deploy路径。#references)

## Installation
##### Clone and install
1. git clone https://github.com/biubug6/Pytorch_Retinaface.git

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

##### Data
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

##### Data1
We also provide the organized dataset we used as in the above directory structure.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

##### self

```
cp -r /data/zhaixingang/K510/Face_Detection/Pytorch_Retinaface/data/widerface/ ./
```

## Training
We provide restnet50 and mobilenet0.25 as backbone network to train model.
We trained Mobilenet0.25 on imagenet dataset and get 46.58%  in top 1. If you do not wish to train the model, we also provide trained model. Pretrain model  and trained model are put in [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and [baidu cloud](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq . The model could be put as follows:
```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``data/config.py and train.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --network resnet50 or
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25
  ```

3. 如果缺少对应的包请使用pip安装

   ```
   -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. 训练的log如下：

   ```
   Loading Dataset...
   Epoch:1/250 || Epochiter: 1/403 || Iter: 1/100750 || Loc: 4.6011 Cla: 10.1222 Landm: 20.3519 || LR: 0.00100000 || Batchtime: 2.7341 s || ETA: 3 days, 4:30:59
   Epoch:1/250 || Epochiter: 2/403 || Iter: 2/100750 || Loc: 4.3335 Cla: 9.7642 Landm: 20.5612 || LR: 0.00100000 || Batchtime: 0.2298 s || ETA: 6:25:50
   Epoch:1/250 || Epochiter: 3/403 || Iter: 3/100750 || Loc: 4.4592 Cla: 9.1041 Landm: 20.6768 || LR: 0.00100000 || Batchtime: 0.2171 s || ETA: 6:04:31
   Epoch:1/250 || Epochiter: 4/403 || Iter: 4/100750 || Loc: 4.5068 Cla: 8.7250 Landm: 19.6846 || LR: 0.00100000 || Batchtime: 0.2790 s || ETA: 7:48:28
   Epoch:1/250 || Epochiter: 5/403 || Iter: 5/100750 || Loc: 4.6380 Cla: 8.4013 Landm: 19.2466 || LR: 0.00100000 || Batchtime: 0.2230 s || ETA: 6:14:25
   Epoch:1/250 || Epochiter: 6/403 || Iter: 6/100750 || Loc: 4.6224 Cla: 7.9167 Landm: 19.4438 || LR: 0.00100000 || Batchtime: 0.2128 s || ETA: 5:57:21
   Epoch:1/250 || Epochiter: 7/403 || Iter: 7/100750 || Loc: 4.4048 Cla: 7.5806 Landm: 20.1771 || LR: 0.00100000 || Batchtime: 0.1806 s || ETA: 5:03:16
   Epoch:1/250 || Epochiter: 8/403 || Iter: 8/100750 || Loc: 4.4603 Cla: 6.8595 Landm: 20.0009 || LR: 0.00100000 || Batchtime: 0.2413 s || ETA: 6:45:11
   Epoch:1/250 || Epochiter: 9/403 || Iter: 9/100750 || Loc: 4.0585 Cla: 6.7934 Landm: 19.4913 || LR: 0.00100000 || Batchtime: 0.2151 s || ETA: 6:01:06
   Epoch:1/250 || Epochiter: 10/403 || Iter: 10/100750 || Loc: 4.5355 Cla: 6.1050 Landm: 18.5144 || LR: 0.00100000 || Batchtime: 0.3229 s || ETA: 9:02:08
   Epoch:1/250 || Epochiter: 11/403 || Iter: 11/100750 || Loc: 4.1846 Cla: 6.2162 Landm: 19.3465 || LR: 0.00100000 || Batchtime: 0.1819 s || ETA: 5:05:21
   Epoch:1/250 || Epochiter: 12/403 || Iter: 12/100750 || Loc: 4.0089 Cla: 5.7972 Landm: 18.4460 || LR: 0.00100000 || Batchtime: 0.1924 s || ETA: 5:22:59
   Epoch:1/250 || Epochiter: 13/403 || Iter: 13/100750 || Loc: 4.0790 Cla: 5.4977 Landm: 18.9646 || LR: 0.00100000 || Batchtime: 0.2318 s || ETA: 6:29:15
   Epoch:1/250 || Epochiter: 14/403 || Iter: 14/100750 || Loc: 4.2989 Cla: 5.5341 Landm: 17.5988 || LR: 0.00100000 || Batchtime: 0.2284 s || ETA: 6:23:24
   Epoch:1/250 || Epochiter: 15/403 || Iter: 15/100750 || Loc: 4.2228 Cla: 5.2870 Landm: 18.2571 || LR: 0.00100000 || Batchtime: 0.1833 s || ETA: 5:07:40
   Epoch:1/250 || Epochiter: 16/403 || Iter: 16/100750 || Loc: 4.1916 Cla: 5.0033 Landm: 18.6234 || LR: 0.00100000 || Batchtime: 0.1869 s || ETA: 5:13:46
   ```

   

## Evaluation

### Evaluation widerface val
1. 修改test_widerface.py

   ```
       with open(testset_list, 'r') as fr:
           test_dataset = fr.read().split()
       num_images = len(test_dataset)
   
       _t = {'forward_pass': Timer(), 'misc': Timer()}
   
       # testing begin
       for i, img_name in enumerate(test_dataset):
           if '.jpg' not in img_name:
               continue
   ```

2. Generate txt file
```Shell
python test_widerface.py --trained_model weight_file --network mobile0.25 or resnet50
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)
### Evaluation FDDB

1. Download the images [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to:
```Shell
./data/FDDB/images/
```

2. Evaluate the trained model using:
```Shell
python test_fddb.py --trained_model weight_file --network mobile0.25 or resnet50
```

3. Download [eval_tool](https://bitbucket.org/marcopede/face-eval) to evaluate the performance.

<p align="center"><img src="curve/1.jpg" width="640"\></p>

## TensorRT
-[TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)

## ONNX

```
python convert_to_onnx.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25 --long_side 320 --cpu --output_onnx ./onnx/mobilenet0.25_Final_320.onnx

python convert_to_onnx.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25 --long_side 640 --cpu --output_onnx ./onnx/mobilenet0.25_Final_640.onnx
```

## References

- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
