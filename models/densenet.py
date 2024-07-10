import torch
import torch.nn as nn
import torch.nn.functional as F

#### 定义DenseLayer
#### DeseLayer包括Bn+Relu+1*1ConV+Bn+Relu+3*3ConV， 其中1*1ConV用于减少通道数，3*3ConV用于在降维后的特征图上卷积提取特征
class DenseLayer(nn.Module):
    ### 初始化函数
    ## in_channels是输入的特征图的维度，growth_rate是输出的特征图的维度，
    ## bn_size是尺寸因子，决定1*1卷积层输出特征图的通道数，drop_rate是drop_out的丢弃率，用于在训练时随机丢弃一些神经元，防止过拟合
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        ## 调用父类的构造函数
        super(DenseLayer, self).__init__()
        ## 定义DenseLayer的各个层
        self.norm1 = nn.BatchNorm2d(in_channels) # Bn1，批量归一化
        self.relu1 = nn.ReLU(inplace=True) # Relu1，Relu层
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False) # 卷积层1
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate) # 输入conv1的输出
        self.relu2 = nn.ReLU(inplace=True) # 激活函数
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False) # 卷积层2
        ## 定义DenseLayer的丢弃率
        self.drop_rate = drop_rate

    ### forward函数
    ## x是当前层的输入
    def forward(self, x):
        out = self.conv2(self.relu2(self.norm2(self.conv1(self.relu1(self.norm1(x))))))
        ## 应用drop_out
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        ## 进行特征连接
        return torch.cat([x, out], 1)
    

#### 定义DenseBlock
#### DenseBlock包括一系列的DenseLayer
class DenseBlock(nn.Module):
    ### 初始化函数
    ## n_layers是每个DenseBlock包含的DenseLayer层数
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(in_channels, growth_rate, bn_size, drop_rate, n_layers)

    ### _make_layer函数，对DenseBlock中的layer进行初始化
    def _make_layer(self, in_channels, growth_rate, bn_size, drop_rate, n_layers):
        layers = []
        ## 依次初始化DenseLayer并在列表中追加
        for i in range(n_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        ## 用容器封装DenseBlock
        return nn.Sequential(*layers)
    
    ### forward函数
    def forward(self, x):
        return self.layer(x)
    

#### 定义TransitionLayer
#### TransitionLayer包括Bn+Relu+1*1ConV+2*2AvgPool，其中1*1ConV用于减少通道数，2*2AvgPool用于压缩特征
class TransitionLayer(nn.Module):
    ### 初始化函数
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    ### foward函数
    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.norm(x))))
        return out
    

#### 用DenseBlock和TransitionLayer构造DenseNet
class DenseNet(nn.Module):
    ### 初始化函数
    ### num_init_features是第一次卷积结束后特征图的通道数
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                num_init_features=64, bn_size=4, drop_rate=0, num_classes=4):
        super(DenseNet, self).__init__()
        ## 初始化growth_rate
        self.growth_rate = growth_rate
        ## 第一次卷积，直接对输入进行操作
        self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False) # 3代表输入图像的通道数，rgb
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 输出的特征图的通道数不会改变
        ## 组装DenseNet
        # 定义模型变量
        self.dense_blocks = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # 添加DenseBlock
            self.dense_blocks.append(DenseBlock(in_channels=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate, n_layers=num_layers))
            num_features += num_layers * growth_rate # 更新经过当前block后特征的通道数
            # 添加TransitionLayer，在两层DenseBlock之间添加
            if i != (len(block_config) - 1):
                self.trans_layers.append(TransitionLayer(in_channels=num_features, out_channels=num_features // 2))
                num_features = num_features // 2
        ## 最后一次正则化
        self.norm_f = nn.BatchNorm2d(num_features)
        ## 线性层
        self.classifier = nn.Linear(num_features, num_classes)

    ### forward函数
    def forward(self, x):
        ## 进行第一次卷积
        out = self.pool0(self.relu0(self.norm0(self.conv0(x))))
        ## 依次经过DenseBlock和TransitionLayer
        for i in range(len(self.dense_blocks)):
            out = self.dense_blocks[i](out)
            if i != len(self.dense_blocks) - 1:
                out = self.trans_layers[i](out)
        ## 进行最后一次正则化
        out = self.norm_f(out)
        ## 全局平均池化
        out = F.adaptive_avg_pool2d(out, (1, 1))
        ## 展平
        out = torch.flatten(out, 1)
        ## 线性层
        out = self.classifier(out)
        ## 返回结果
        return out 


# #### 测试模型
# if __name__ == '__main__':
#     # 假设模型期望的输入图像大小是 32x32
#     batch_size = 1
#     channels = 3  # RGB图像
#     height, width = 32, 32
#     input_data = torch.randn(batch_size, channels, height, width)

#     # 测试模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = DenseNet().to(device)
#     net.eval()  # 切换模型到评估模式，关闭 drop_out 等训练时特有的功能

#     # 将输入数据传递给模型
#     input_data = input_data.to(device)
#     output = net(input_data)

#     print("Output size:", output)  # 打印输出张量


