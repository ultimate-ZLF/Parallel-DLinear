import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class ParallelDLinear(nn.Module):
    """
    DLinear
    """
    def __init__(self, channels, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = 1

        # Decompsition Kernel Size
        kernel_size = 21
        self.decompsition = series_decomp(kernel_size)
        self.channels = channels

        self.linear_seasonal = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=seq_len,
            groups=channels,
            bias=True
        )
        
        self.linear_trend = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=seq_len,
            groups=channels,
            bias=True
        )

        with torch.no_grad():
            self.linear_seasonal.weight.data.fill_(1/seq_len)
            self.linear_trend.weight.data.fill_(1/seq_len)
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal, trend = self.decompsition(x)
        
        # [Batch, Channel, Seq_len]
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        
        seasonal_out = self.linear_seasonal(seasonal)  # [Batch, Channel, 1]
        trend_out = self.linear_trend(trend)  # [Batch, Channel, 1]

        x = seasonal_out + trend_out
        x = x.permute(0,2,1) # to [Batch, Output length, Channel]
        return x
