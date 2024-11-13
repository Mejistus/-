import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.WQ = nn.Parameter(torch.randn(input_size, input_size))
        self.WK = nn.Parameter(torch.randn(input_size, input_size))
        self.WV = nn.Parameter(torch.randn(input_size, input_size))

    def forward(self, x):
        # 计算注意力权重
        scores = torch.bmm(x, self.W.unsqueeze(0))
        attention_weights = torch.softmax(scores, dim=1)
        # 计算注意力输出
        output = torch.bmm(attention_weights, x)
        return output

# 测试代码
if __name__ == '__main__':
    input_tensor = torch.randn(3, 8)
    attention = SelfAttention(8)
    output = attention(input_tensor)
    print(output.shape)


