import torch
import sys
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.simple_lstm import SimpleLSTMForecast
from dataset.MATLAB_Dataset import MatDataset


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)


def test_train_model():

    window_size = 2560                  # 人耳频率 20Hz~20480Hz，所以选最低20Hz
    forecast_length = window_size       # 序列预测序列
    hidden_size = 128                   # LSTM 隐藏层神经元数
    batch_size = 20                     # 1个batch为1s的数据
    input_size = 48                     # 输入：48个Features
    output_size = 10                    # 输出：10个Responses

    model = SimpleLSTMForecast(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        forecast_length=forecast_length
    )

    dataset = MatDataset(train_mode=1, root_path="D:\\VN_DL_Dataset", window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    k = len(dataloader)

    # 定义损失函数和优化器
    criterion = nn.MSELoss(reduction='none')  # 假设我们使用均方误差作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

    # 使用StepLR学习率调度器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # 设置训练的轮数
    num_epochs = 45

    # 检测GPU是否可用，并转移model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 开始训练
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0
        for inputs, targets in dataloader:
            # t_start = time.time()
            inputs = inputs.transpose(1, 0)     # nn.LSTM 默认输入顺序为(seq, batch, feature) batch_first 默认为 false
            targets = targets.transpose(1, 0)   # nn.LSTM 默认输出顺序为(seq, batch, feature) batch_first 默认为 false

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)
            targets_rms = []
            loss_rmse = []
            for i in range(10):
                targets_rms.append(torch.sqrt(torch.mean(targets[:, :, i] ** 2)))
            for i in range(10):
                loss_rmse.append(torch.sqrt(torch.mean(loss[:, :, i])))
            targets_rms = torch.stack(targets_rms)
            loss_rmse = torch.stack(loss_rmse)
            loss_final = torch.mean(loss_rmse / targets_rms)

            # 反向传播
            loss_final.backward()
            
            # 参数更新
            optimizer.step()

            # 累计损失
            running_loss += loss_final.item()
            # t_end = time.time()
            # print(f'train time {t_end - t_start :.4f}')

        # 变学习率
        scheduler.step()

        # 打印每轮的平均损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/k:.4f}  Learning Rate: {optimizer.param_groups[0]['lr']}")

    torch.save(model, 'model.pth')
    print('Training finished.')


if __name__ == '__main__' :
    test_train_model()
