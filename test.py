import torch
from utils.dataset import get_train_test_loaders
from models.densenet import DenseNet

#### test函数
def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
    return correct.doouble() / len(test_loader.dataset)


#### 主函数
if __name__ == '__main__':
    ### 配置测试参数
    data_dir = 'data'
    batch_size = 32

    ### 获取数据集加载器
    _, _, test_loader = get_train_test_loaders(data_dir, batch_size)

    ### 配置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### 初始化模型并加载训练好的模型参数
    model = DenseNet(num_classes=len(test_loader.dataset.classes))
    model.load_state_dict(torch.load("DenseNet_Model.pth"))
    model = model.to(device)

    ### 进行测试
    test_accuracy = test(model, test_loader, device)

    ### 打印测试结果
    print(f"Test Accuracy:{test_accuracy:.4f}")