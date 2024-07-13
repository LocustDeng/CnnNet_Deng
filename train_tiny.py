import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from models.densenet import DenseNet
from dataset import train_dataset, val_dataset, train_dataloader, val_dataloader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    ### 配置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
 
    ### 配置图片预处理
    data_transform = {
        # 训练
        "train": transforms.Compose([
            # RandomResizedCrop(224)：将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为给定大小
            transforms.RandomResizedCrop(224),
            # RandomVerticalFlip()：以0.5的概率竖直翻转给定的PIL图像
            transforms.RandomHorizontalFlip(),
            # ToTensor()：数据转化为Tensor格式
            transforms.ToTensor(),
            # Normalize()：将图像的像素值归一化到[-1,1]之间，使模型更容易收敛
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        # 验证
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    ### 配置数据集路径
    # abspath()：获取文件当前目录的绝对路径，join()：用于拼接文件路径，可以传入多个路径，getcwd()：该函数不需要传递参数，获得当前所运行脚本的路径
    data_root = os.path.abspath(os.getcwd())
    # 得到数据集的路径
    image_path = os.path.join(data_root, "tiny_imagenet_data")
    # exists()：判断括号里的文件是否存在，可以是文件路径，如果image_path不存在，序会抛出AssertionError错误，报错为参数内容“ ”
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    ### 加载数据集
    ## 一次训练载入16张图像
    batch_size = 64
    ## 确定进程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    ## 加载数据
    # train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    # validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    ## 获取数据集长度
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    ## 获取批次数
    train_steps = len(train_dataloader)
    val_steps = len(val_dataloader)

    # ### 构建分类类别
    # # class_to_idx：获取分类名称对应索引
    # data_list = train_dataset.class_to_idx
    # # 循环遍历数组索引并交换val和key的值重新赋值给数组，这样模型预测的直接就是value类别值
    # cla_dict = dict((val, key) for key, val in data_list.items())
    # # 把字典编码成json格式
    # json_str = json.dumps(cla_dict, indent=4)
    # # 把字典类别索引写入json文件
    # with open('class_indices_tiny.json', 'w') as json_file:
    #     json_file.write(json_str)
    
    ### 模型实例化
    net = DenseNet(num_classes=200)
    net.to(device)
 
    ### 定义损失函数（交叉熵损失）
    loss_function = nn.CrossEntropyLoss()
 
    ### 定义adam优化器
    params = [p for p in net.parameters() if p.requires_grad] # 抽取模型参数
    optimizer = optim.Adam(params, lr=0.0001)
 
    ### 设置训练参数
    ## 迭代次数（训练次数）
    epochs = 25
    ## 用于判断最佳模型
    best_acc = 0.0
    ## 最佳模型保存地址
    save_path = 'DenseNet_Model_Tiny.pth'
    
    ### 训练&验证
    for epoch in range(epochs):
        ### 训练
        net.train()
        running_loss = 0.0
        ## tqdm：进度条显示
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            ## 前向传播
            # 获取数据
            images, labels = data
            # 计算训练值
            logits = net(images.to(device))
            # 计算损失
            loss = loss_function(logits, labels.to(device))

            ## 反向传播
            # 清空过往梯度
            optimizer.zero_grad()
            # 反向传播，计算当前梯度
            loss.backward()
            optimizer.step()
 
            ## 累积当前epoch的损失
            running_loss += loss.item()
 
            ## 进度条的前缀
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
 
        ### 验证
        net.eval()
        acc = 0.0
        val_running_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                loss = loss_function(outputs, val_labels.to(device))
                val_running_loss += loss.item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs, loss)
        val_accurate = acc / val_num

        ### 打印当前epoch相关信息
        print('[epoch %d] train_loss: %.3f val_loss: %.3f val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_running_loss / val_steps, val_accurate))
 
        ### 保存最好的模型权重
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
 
    print('Finished Training')
 
 
if __name__ == '__main__':
    main()