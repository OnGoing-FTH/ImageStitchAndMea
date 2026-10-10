import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from types import SimpleNamespace


# 当前脚本目录：secondly_paper_all_file/VIT_model/
cur_dir = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：secondly_paper_all_file/
project_root = os.path.abspath(os.path.join(cur_dir, ".."))

# 把根目录加入 Python 模块搜索路径
sys.path.append(project_root)
from models.pidinet import pidinet_tiny


class dipid_tiny(nn.Module): # 将主干输出的特征变成图像特征
    def __init__(self, type= "r-v15 " ,dil=True, sa=True): #     使用 CDCM（扩张卷积）	使用 CSAM（注意力 
        super().__init__()
        self.net_type = type
        self.dil = dil
        self.sa = sa
        args = SimpleNamespace(
            config=self.net_type,
            dil=self.dil,
            sa=self.sa
        )
        self.model_pidinet = pidinet_tiny(args)

   
    def forward(self, x):
        x=self.model_pidinet(x)
        return x #  (B, 1, 512, 512)
        

if __name__ == "__main__":
    # 模拟一个 1x3x512x512 的图像张量
    x = torch.randn(1, 3, 512, 512)
    model = dipid_tiny(type='r-v15', dil=True, sa=False)
    x= model(x)
    print(x.shape)