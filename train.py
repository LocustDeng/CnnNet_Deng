import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from utils.dataset import get_train_test_loaders
from models.densenet import DenseNet
from tqdm import tqdm

#####Epoch和Batch#####
# Epoch 表示整个训练数据集被模型使用一次的次数
# 在每个epoch中，模型会通过整个数据集进行一次前向传播和反向传播，并更新模型的参数一次（或多次，具体取决于优化器和学习率调度策略）
# Batch 表示在每次模型训练过程中，从训练数据集中取出的一部分数据
# 这部分数据被送入模型中进行前向传播和反向传播，并用来计算损失和更新模型参数
# 在训练过程中，数据集通常会被分割成多个 batch，每个 batch 包含一定数量的样本
# 模型在一个 batch 上计算得到的梯度被用来更新模型参数，从而使得模型向损失函数减小的方向优化
# 一个 epoch 中包含多个 batch，模型通过迭代每个 batch 来完成整个数据集的一次训练
# 例如，如果数据集有 1000 个样本，batch_size 设置为 100，那么一个 epoch 将包含 10 个 batch
# 训练过程中，通常会进行多个 epoch。每个 epoch 结束时，模型会重新遍历整个数据集，继续进行参数更新
# 这样，模型通过多个 epoch 的迭代，逐步优化其在训练数据上的性能，直至达到训练的终止条件（如预设的 epoch 数量、损失函数收敛等）
######################

#### train函数
def train(model, train_loader, criterion, optimizer, device):
    ### 将模型置于训练模式，便于进行Drop_out和Bn等操作
    model.train()
    ### 初始化用于累计当前batch损失的变量
    running_loss = 0.0
    ### 依次对每个batch进行处理，这里的inputs和labels就对应了一个batch
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) # 将数据移到cuda
        ## pytorch默认累积梯度，将梯度显式清0
        optimizer.zero_grad()
        ## 将inputs传入模型得到outputs
        outputs = model(inputs)
        ## 通过比较outputs和inputs的labels计算得loss
        loss = criterion(outputs, labels)
        ## 进行梯度回传
        loss.backward()
        ## 根据优化器策略和计算所得梯度，优化模型参数
        optimizer.step()
        ## 累加求当前batch的累积损失，inputs.size(0)是当前batch中的样本数量
        running_loss += loss.item() * inputs.size(0)
    ### 返回整个训练集上的平均损失值，即所有batch的累积损失除以训练集中的样本总数，这是一个指示模型在训练过程中性能的指标
    return running_loss / len(train_loader.dataset)


#### validate函数
def validate(model, val_loader, criterion, optimizer, device):
    ### 将模型置于验证模式，禁用Drop_out等操作
    model.eval()
    ### 初始化记录损失和样本分类正确的个数
    running_loss = 0.0
    correct = 0
    ### 在不计算梯度的情形下操作
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device) # 将数据移到cuda
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            ## torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)，返回一个元组 (values, indices)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data) # .data访问张量的底层数据，即实际存储的数值
    return running_loss / len(val_loader.dataset), correct.double() / len(val_loader.dataset)


#### 主函数
if __name__ == '__main__':
    print("Train begin.\n")
    ### 设置训练参数
    data_dir = 'data' # 数据集路径
    batch_size = 32 # 每次训练模型时用来处理的样本数，将数据集分为更小的batch来进行训练
    learning_rate = 0.001 # 学习率，学习率决定了每次参数更新的步长大小。如果学习率设置过高，可能导致模型无法收敛；如果设置过低，则训练时间可能会过长
    num_epochs = 25 # 训练的轮数，每一轮完成一次前向和反向传播，并更新参数

    ### 获取数据集加载器
    train_loader, val_loader, test_loader = get_train_test_loaders(data_dir, batch_size)

    ### 设置训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### 初始化DenseNet模型
    model = DenseNet(num_classes=len(test_loader.dataset.classes))
    model = model.to(device)

    ### 配置损失函数，使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    ### 配置优化器，使用梯度下降优化算法，model.parameters() 提供了需要优化的模型参数，lr=learning_rate 指定了优化器的初始学习率
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ### 迭代训练
    progress_bar = tqdm(range(0, num_epochs), desc="Training progress")
    for epoch in range(num_epochs):
        ## 进行训练，并返回训练的loss
        train_loss = train(model, train_loader, criterion, optimizer, device)
        ## 进行验证，并返回验证的loss和准确率
        val_loss, val_accuracy = validate(model, val_loader, criterion, optimizer, device)
        ## 打印当前轮次的训练信息
        print(f'Epoch{epoch + 1}/{num_epochs}, Train loss{train_loss:.4f}, Val loss{val_loss:.4f}, Val accuracy{val_accuracy:.4f}')
        ## 更新进度条
        with torch.no_grad(): # 在不计算梯度的情况下执行，节省内存并加快计算速度
            progress_bar.update(1)

    ### 训练结束，保存模型
    torch.save(model.state_dict(), "DenseNet_Model.pth")

    ### 关闭进度条
    progress_bar.close()
    print("Train complete.\n")