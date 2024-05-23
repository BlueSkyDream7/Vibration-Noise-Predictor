import torch.nn as nn
from torch.nn import functional as f


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dr=0.0):
        super(SimpleLSTM, self).__init__()                      # 继承父类初始化函数
        self.linearIn = nn.Linear(input_size, hidden_size)      # 线性输入层（input_size个输入 hidden_size个输出）
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )                                                       # LSTM层（hidden_size个输入 hidden_size个输出 无dropout）
        self.linearOut = nn.Linear(hidden_size, output_size)    # 线性输出层（hidden_size个输入 output_size个输出）

    def forward(self, x):                                       # 定义神经网络的前向传播函数forward，x即为输入
        x0 = f.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        return self.linearOut(out_lstm)                         # 线性输入 -> ReLU -> LSTM -> 线性输出


class SimpleLSTMForecast(SimpleLSTM):
    def __init__(self, input_size, output_size, hidden_size, forecast_length, dr=0.0):
        super(SimpleLSTMForecast, self).__init__(input_size, output_size, hidden_size, dr)
        self.forecast_length = forecast_length
        self.output_size = output_size

    def forward(self, x):
        # 调用父类的forward方法获取完整的输出
        full_output = super(SimpleLSTMForecast, self).forward(x)

        # 重塑输出以匹配目标形状
        # forecast_output = full_output[-self.forecast_length:, :, :]
        forecast_output = full_output
        return forecast_output
