import torch,math 
from torch import nn
import torch.nn.functional as F
def get_1d_dct(i, freq, L): # freq指的是频率
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0: 
        return result 
    else: 
        return result * math.sqrt(2) 
def get_dct_weights( width, height, channel, fidx_u= [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3], fidx_v= [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]):
    # 该函数的目的是生成一个用于多频谱注意力的DCT权重张量，以适应输入图像的不同尺寸和通道数量
    # width : width of input 
    # height : height of input 
    # channel : channel of input 
    # fidx_u : horizontal indices of selected fequency 低频高频差异不大
    # according to the paper, should be [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3]
    # fidx_v : vertical indices of selected fequency 
    # according to the paper, should be [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]
    # [0,0],[0,1],[6,0],[0,5],[0,2],[1,0],[1,2],[4,0],
    # [5,0],[1,6],[3,0],[0,4],[0,6],[0,3],[2,2],[3,5],
    scale_ratio = width//7
    # 计算缩放比例，目的是确保频率在不同大小的输入图像中保持一致。7是作者选择的一个常数
    fidx_u = [u*scale_ratio for u in fidx_u]
    # 将水平和垂直方向的频率索引根据缩放比例进行调整，以适应输入图像的尺寸。
    fidx_v = [v*scale_ratio for v in fidx_v]
    # 将水平和垂直方向的频率索引根据缩放比例进行调整，以适应输入图像的尺寸。
    dct_weights = torch.zeros(1, channel, width, height) 
    # 初始化一个全零的张量，用于存储生成的DCT权重。张量的形状为 (1, channel, width, height)。
    c_part = channel // len(fidx_u) 
    # 计算每个频率分量分配给通道的数量，以便在后续的循环中进行通道划分。
    # split channel for multi-spectal attention 
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)): 
        # 遍历频率索引列表 fidx_u 和 fidx_v 中的每个元素，其中 u_x 是水平方向的索引，v_y 是垂直方向的索引。
        for t_x in range(width):  # 遍历输入图像的宽度。
            for t_y in range(height):   # 遍历输入图像的宽度。
                dct_weights[:, i * c_part: (i+1)*c_part, t_x, t_y]\
                =get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height) 
                # 计算一维DCT并将结果分配给相应位置的张量切片。这里使用了 get_1d_dct 函数，其中 t_x 表示水平方向的位置，t_y 表示垂直方向的位置。
    # Eq. 7 in our paper 
    return dct_weights  #  DCT 权重张量 dct_weights



class FcaLayer(nn.Module):
    def __init__(self,
                 channel,
                 reduction,width,height):
        super(FcaLayer, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer('pre_computed_dct_weights',get_dct_weights(self.width,self.height,channel)) 
        #self.register_parameter('pre_computed_dct_weights',torch.nn.Parameter(get_dct_weights(width,height,channel)))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x,(self.height,self.width))
        y = torch.sum(y*self.pre_computed_dct_weights,dim=(2,3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
