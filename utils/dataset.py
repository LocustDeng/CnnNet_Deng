import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict

#### 读wnids.txt文件
#### wnids.txt文件包含了一系列的WordNet ID（wnids），这些ID通常被用来标识和组织图像数据，每个ID对应一个类别
def read_wnids(data_dir):
    ### 将文件中的每一行都存储在列表wnids中
    with open(os.path.join(data_dir, 'wnids.txt'), 'r') as f:
        wnids = [line.strip() for line in f] # strip用于去除每行首尾的空白字符，包括\n
    return wnids


#### 读words.txt文件
#### words.txt文件包含了每个WordNet ID（wnid）对应的类别名称（或标签），通常是人类可读的形式，而不是单纯的标识符
def read_words(data_dir):
    words = {}
    with open(os.path.join(data_dir, 'words.txt'), 'r') as f:
        for line in f:
            wnid, word = line.split('\t')
            words[wnid.strip()] = word.strip()
    return words


#### 将wnids列表转换为字典，wnid为key，索引值为value
def get_class_to_idx(wnids):
    return {wnid: idx for idx, wnid in enumerate(wnids)}


#### 定义训练集/验证集/测试集数据加载器
def get_train_test_loaders(data_dir, batch_size):
    ### 完成对wnids.txt和words.txt文件的操作
    wnids = read_wnids(data_dir)
    words = read_words(data_dir)
    class_to_idx = get_class_to_idx(wnids)

    ### Compose将多个transforms操作整合在一起，便于对图片进行预处理
    transform = transforms.Compose([
        ## 将输入的图像调整（缩放）为指定的大小，这里将图像的短边调整为256像素，并保持宽高比不变
        transforms.Resize(256),
        ## 对图像进行中心裁剪，使得裁剪后的图像大小为 224x224 像素
        transforms.CenterCrop(224),
        ## 将数据转换为Tensor格式
        transforms.ToTensor(),
        ## 将图像的像素值归一化到[-1,1]之间，使模型更容易收敛
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ### 加载训练集、验证集和测试集数据
    ## 使用PyTorch中的datasets.ImageFolder类来加载图像文件夹数据集，并应用transform的预处理操作
    train_set = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_set = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)
    test_set = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

    ### 设置数据集的类别索引和类别标签信息
    train_set.class_to_idx = class_to_idx
    val_set.class_to_idx = class_to_idx
    test_set.class_to_idx = class_to_idx
    train_set.classes = wnids
    val_set.classes = wnids
    test_set.classes = wnids

    ### 设置数据加载器
    ## num_workers表示同时用四个线程加载数据
    ## 在PyTorch中，DataLoader是一个用于批量加载数据的工具，它从数据集中加载数据并以可迭代的方式提供数据
    ## 一般情况下，DataLoader 返回的每个批次数据是一个元组，其中包含两个元素：
    ## (inputs, targets)
    ## inputs: 一个张量或多个张量，包含了模型的输入数据。这些输入数据可以是图像、文本、序列数据等，具体取决于所使用的数据集和问题
    ## targets: 一个张量或多个张量，包含了对应的目标或标签数据。这些目标数据通常是分类任务中的类别标签，回归任务中的真实值等
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader