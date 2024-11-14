# pytorch
import torch
import torch.nn as nn # 导入神经网络模块，方便构建神经网络层
from torch.utils.data import Dataset, DataLoader # 导入dateset和dataloader用于处理和加载数据

# 数据预处理
import numpy as np
import csv  # 导入csv模块，用于读取和写入csv文件
import os # 导入os模块，用于处理文件和路径的操作

# 画图
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure # 导入Figure，方便创建自定义大小的图表

myseed = 42069 # 设置一个随机种子，用于保证代码的可复现性（确保每次运行时随机数相同）
torch.backends.cudnn.dateministic = True  # 确保使用确定的算法，保证结果可复现
torch.backends.cudnn.benchmark = False # 禁用cudnn的benchmak模式，以避免算法非确定性
np.random.seed(myseed) # 设置np的随机种子，保证np的随机操作可复现
torch.manual_seed(myseed) # 设置torch的随机种子

if torch.cuda.is_available(): # 如果当前设备支持cuda（即有gpu可用）
    torch.cuda.manual_seed_all(myseed) # 为所有GPU设置相同的=随机种子，保证gpu上的操作也可复现

plt.rcParams['font.family'] = ['Simsun']

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


class COVId19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        # 初始化函数，加载并预处理COVID19数据集
        # path：数据集路径
        # mode：指定数据集模式（train、dev、test）
        # target_only： 是否只使用目标特征
        self.mode = mode
        # 将数据读取到np中
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))  # 使用csv.reader读取csv文件
            data = np.array(data[1:])[:, 1:].astype(float)  # 将数据转换为np数组，跳过第一行表头和第一个特征列并将数据转换为浮点数

        if not target_only:
            feats = list(range(93))  # 如果不使用目标特征，选择所有93个特征
            # 灵活选择任意数量和组合的列，而不需要硬编码列索引
        else:
            # 应该使用与 40 个州相关的特征以及两个与确诊病例相关的特征（它们的索引为 57 和 75）
            pass  # 如果target_only为True，应使用特定的特征（训练集95列，验证集94列）

        if mode == 'test':
            # 测试数据：893行x93列（40个州的状态＋3天的数据集：18+18+17=53）
            data = data[:, feats]  # 选择指定的特征列
            self.data = torch.FloatTensor(data)  # 转换为pytorch张量
        else:
            # 训练数据（训练集+验证集）：2700行x94列（40个州的状态＋3天的数据集：18+18+18=54）
            target = data[:, -1]  # 标签为最后一列
            data = data[:, feats]  # 选择特征列

            # 将训练数据拆分为训练集和验证集
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]  # 列表推导式（找出相应的索引），选择90%的数据集作为训练集
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]  # 选择10%的数据集作为验证集

            # 将数据转换为pytorch张量
            self.data = torch.FloatTensor(data[indices])  # 选择指定索引的数据
            self.target = torch.FloatTensor(target[indices])  # 选择对应索引的标签

        # 归一化特征数据（可删除这一部分代码观察效果）
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0,
                                    keepdim=True)  # 对从第 40 列开始的特征进行标准化（减去均值除以标准差） dim=0沿行方向（对列）操作，keepdim=true 保留维度

        self.dim = self.data.shape[1]  # 数据的维度，即特征数量

        # 打印读取数据集的状态
        print('读取数据的模式为： {}  (样本数量为：{}  样本的维度为 = {})'.format(mode, len(self.data),
                                                                                self.dim))  # 显示读取的数据集模式、样本数量和每个样本的维度

    def __getitem__(self, index):
        # 根据索引返回一个样本
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]  # 训练时返回数据和对应的标签
        else:
            return self.data[index]  # 测试时只返回标签

    def __len__(self):
        return len(self.data)  # 返回数据集的大小，即样本数量

    # 它们的主要作用是使自定义数据集能够以更直观的方式与 PyTorch 的数据加载器（如 DataLoader）进行交互

def pred_dataloader(path, mode, batch_size, n_jobs, target_only=False):
    # 函数用于生产一个数据集，并将其放入dataloader中
    # batc_size : 批大小（每个batch中包含的样本数）
    # n_jobs:用于加载数据的工作线程数（默认为0，即不使用多线程）
    # target_only: 是否只使用目标特征

    dataset = COVId19Dataset(path, mode=mode, target_only=target_only) # 构建数据集对象
    dataloader = DataLoader(
        dataset, # 数据集对象
        batch_size, # batch包的大小
        shuffle=(mode == 'train'), # 如果是训练模式，则打乱数据顺序
        drop_last=False, # 不丢弃最后一个不满批次的样本
        num_workers=n_jobs, # 使用的工作线程
        pin_memory=True) # 如果使用GPU，设置pin_memory可以加速数据传输到gpu
    return dataloader # 返回构建好的dataloader对象


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        # 初始化函数，定义神经网络的结构
        # input_dim 输入特征的维度
        super(NeuralNet,self).__init__()
        # 调用 NeuralNet 的父类（nn.Module）的构造函数，确保神经网络的基本属性被正确初始化

        # 使用nn.sequential定义一个简单的全连接神经网络
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # 输入层到隐藏层的全连接，64个隐藏单元
            nn.ReLU(), # 激活函数，增加非线性
            nn.Linear(64, 1) # 隐藏层到输出层的全连接，输出一个数值（回归）
        )

        # 定义损失函数为均方误差
        self.criterion = nn.MSELoss(reduction='mean') # 对所有样本误差取平均

    def forward(self, x):
        # 前向传播函数，输入x的形状为（batch_size x input_dim）
        # 通过神经网络计算并返回，使用squeeze（1）将形状调整为（batch_size,） squeeze(dim) 方法用于从张量中移除大小为 1 的维度。
        return self.net(x).squeeze(1) #  squeeze(dim) 方法用于从张量中移除大小为 1 的维度,便于后续计算

    def cal_loss(self, pred, target):
        # 计算损失函数
        # pred：预测值
        # target：真是标签
        # 这里可以加入L2正则化（权重衰弱）来提升模型的泛化能力
        return self.criterion(pred, target)


def train(tr_set, dv_set, model, config, device):
    # 该函数用于训练深度神经网络（DNN）
    # model 要训练的神经网络模型
    # config 配置字典，包含训练参数
    # device 使用的设备（cpu或 cuda）

    n_epochs = config['n_epochs']  # 最大训练轮数（epochs）

    # 设置优化器，通过config参数中的optimizer名称和优化器超参数动态生成
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])

    min_mse = 1000  # 初始化最小均方误差为一个很大的值，用于保存最佳模型
    loss_record = {'train': [], 'dev': []}  # 用于记录训练和验证的损失值
    early_stop_cnt = 0  # 早停计数器
    epoch = 0  # 初始化当前epoch计数
    while epoch < n_epochs:  # 开始训练循环
        model.train()  # 将训练设置为训练模式
        for x, y in tr_set:
            optimizer.zero_grad()  # 将优化器的梯度清零，防止梯度累加
            x, y = x.to(device), y.to(device)  # 将输入和标签数据移到指定设备（cpu或cuda）
            pred = model(x)  # 计算预测值
            mse_loss = model.cal_loss(pred, y)  # 计算当前批次的均方误差
            mse_loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 使用优化器更新模型参数
            loss_record['train'].append(mse_loss.detach().cpu().item())  # 记录损失值

        # 在每个epoch结束后，使用验证集岑模型性能
        dev_mse = dev(dv_set, model, device)  # 计算验证集上的均方误差
        if dev_mse < min_mse:
            min_mse = dev_mse  # 更新最小验证损失
            print('保存模型(epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
            torch.save(model.state_dict, config['save_path'])  # 保存模型参数到指定路径
            early_stop_cnt = 0  # 重置早停计数器
        else:
            early_stop_cnt += 1  # 若验证集损失没有下降，增加早停计数器
        epoch += 1  # 进入下一轮训练
        loss_record['dev'].append(dev_mse)  # 记录验证损失
        if early_stop_cnt > config['early_stop']:
            break  # 若验证集损失在'early_stop'次训练中没有改善，提前终止训练

    print('在{}个循环后完成训练'.format(epoch))  # 打印训练结束后的轮数
    return min_mse, loss_record  # 返回最小验证集均方误差和损失记录

def dev(dv_set, model, device):
    # 该函数用于在验证集上评估模型的性能
    # model：训练好的神经网络模型

    model.eval() # 将模型设置为评估模式，关闭dropout和batch normalizetion等训练时使用的层
    total_loss = 0  # 初始化总损失为0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad(): # 禁用梯度计算，在验证时不需要反向传播
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)  # 累加损失，乘以当前批次的样本数
    total_loss = total_loss / len(dv_set.dataset) # 计算验证集的平均损失
    return total_loss

def test(tt_set, model, device):
    # 该函数用于在测试集上生产模型的预测结果
    model.eval()
    preds = []  # 初始化列表用于存储每个批次的预测结果
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())  # 将预测结果从计算设备转移到cpu，并存入preds列表中
    preds = torch.cat(preds, dim=0).numpy()  # 将所有批次的预测结果拼接成一个完整的tensor，并转换为numpy数组
    return preds

device = get_device() # 获取当前可用的设备
os.makedirs('models', exist_ok=True) # 创建存放模型的文件夹,如果文件已存在不报错
target_only = False # 在训练过程中使用所有特征

# 以下是模型训练的超参数配置
config = {
    'n_epochs': 3000,  # 最大训练轮数，模型将最多训练 3000 轮。如果模型在训练过程中提前满足条件，可能会早停。
    'batch_size': 270,  # DataLoader 中的小批量大小（每个批次中包含的样本数量为 270 个）。
    'optimizer': 'SGD',  # 使用随机梯度下降（SGD）作为优化算法，定义了用于更新模型权重的优化器。
    'optim_hparas': {  # 优化器的超参数
        'lr': 0.001,  # 学习率，决定每次权重更新的步幅大小。较小的学习率可以使模型更稳定，但学习速度较慢。
        'momentum': 0.9  # 动量，帮助优化器更快地收敛，同时减少震荡。
    },
    'early_stop': 200,  # 早停机制，表示如果在 200 个 epoch 内验证集损失没有改进，训练将提前停止。
    'save_path': 'models/model.pth'  # 模型将保存到该路径下，文件名为 'model.pth'。
}

tr_path = r'D:\桌面\软件学习\2021 ML\01 Introduction\作业HW1\covid.train.csv'
tt_patch = r'D:\桌面\软件学习\2021 ML\01 Introduction\作业HW1\covid.test.csv'
tr_set = pred_dataloader(tr_path, 'train', config['batch_size'], n_jobs=0, target_only = target_only) # 训练模式下数据将被打乱
dv_set = pred_dataloader(tr_path, 'dev', config['batch_size'], n_jobs=0, target_only =target_only) # 数据不会被打乱, 评估模型性能
tt_set = pred_dataloader(tt_patch, 'test', config['batch_size'], n_jobs=0, target_only = target_only) # 预测模型的输出
model = NeuralNet(tr_set.dataset.dim).to(device) # 构建模型，输入维度来自训练集的数据维度

model_loss, model_loss_record= train(tr_set, dv_set, model, config, device)

del model # 删除当前模型，释放内存

model = NeuralNet(tr_set.dataset.dim).to(device) # 重新初始化一个新的模型,输入维度相同
ckpt = torch.load(config['save_path'], map_location='cpu', weights_only=True) # 加载之前训练好的最佳模型权重，在cpu上计算
model.load_state_dict(ckpt) # 将加载的权重赋值给新创建的模型
plot_pred(dv_set, model, device) # 在验证集上展示模型的预测效果，观察模型的效果

def save_pred(preds, file):
    print('保存文件为：{}'.format(file))
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp) # 创建csv写入器
        writer.writerow(['id', 'tested_positive']) # 写入表头
        for i, p in enumerate(preds):
            writer.writerow([i, p]) # 将索引和预测值写入文件
preds = test(tt_set, model, device)
save_pred(preds, 'pred.csv')