import torch
import cv2
import numpy as np
import os
import sys
from collections import OrderedDict

# project/
#   ├── edge_methods/
#   ├── models/  <-- 存放 pidinet_block_two_class.py
#   └── ...
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pidinet_block_two_class import dipid_tiny
from .base import EdgeExtractorBase


class PiDiNetEdge(EdgeExtractorBase):
    def __init__(self, model_path=None, device=None, thresh=0.5):
        self.thresh = thresh
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if dipid_tiny is not None:
            # 初始化模型
            self.model = dipid_tiny("r-v15", True, True).to(self.device)
            
            # 默认权重路径
            if model_path is None:
                model_path = r"D:\secondly_paper_all_file\实验\第二步实验\pidinet-master_bindary\logs\data2\epoch_200_weights.pth"
            
            if os.path.exists(model_path):
                # ================= 核心修改：智能加载权重 =================
                print(f"Loading PiDiNet from {model_path} ...")
                try:
                    # 1. 加载原始 state_dict
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # 兼容性处理：有些 pth 保存的是 {'state_dict': ...}，有些直接是 dict
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint

                    # 2. 创建一个新的有序字典，用于存放修正后的权重
                    new_state_dict = OrderedDict()
                    
                    # 3. 遍历权重，进行清洗
                    for k, v in state_dict.items():
                        name = k
                        
                        # 情况A: 去除 'module.' 前缀 (多卡训练产物)
                        if name.startswith('module.'):
                            name = name[7:] 
                        
                        # 情况B: 你的报错提示缺少 'model_pidinet.' 前缀
                        # 如果模型需要 'model_pidinet.xxx' 但权重里只有 'xxx'
                        # 则手动加上前缀
                        if not name.startswith('model_pidinet.'):
                            # 这里假设缺少的正是这个前缀，尝试拼接
                            # 注意：仅当你的 dipid_tiny 结构确实包含 model_pidinet 子模块时有效
                            name = 'model_pidinet.' + name
                            
                        new_state_dict[name] = v

                    # 4. 尝试加载 (strict=False 允许部分不匹配，但我们会打印未匹配的键)
                    # 先尝试严格加载，如果清洗得当应该没问题
                    try:
                        self.model.load_state_dict(new_state_dict, strict=True)
                        print("Success: Weights loaded strictly.")
                    except RuntimeError as e:
                        print(f"Strict loading failed, trying loose loading... Error: {e}")
                        # 如果上面的清洗逻辑不对 (比如不该加前缀)，退回到 strict=False
                        # 或者是尝试直接加载原始 dict (去除 module 后)
                        
                        # 备选方案：只去 module 前缀，不加 model_pidinet
                        fallback_dict = OrderedDict()
                        for k, v in state_dict.items():
                            name = k.replace('module.', '')
                            fallback_dict[name] = v
                        
                        self.model.load_state_dict(fallback_dict, strict=False)
                        print("Warning: Weights loaded with strict=False. Some layers might be missing.")

                except Exception as e:
                    print(f"CRITICAL ERROR loading weights: {e}")
                    self.model = None
                # =======================================================
                
                if self.model is not None:
                    self.model.eval()
            else:
                print(f"Error: Weights not found at {model_path}")
                self.model = None
        else:
            self.model = None

    @torch.no_grad()
    def extract(self, image, roi_info, step, bg_color=None):
        if self.model is None:
            return []

        # 1. 【核心优化】切出 ROI 区域，只对小图进行推理
        crop_img, off_x, off_y = self._get_roi_crop(image, roi_info, padding=32) # padding稍微大点，防止边缘被切断
        if crop_img is None: return []

        h, w = crop_img.shape[:2]

        # 2. 预处理 (参考你之前的 DataLoader 逻辑)
        # 转为 RGB -> Normalize -> Tensor
        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0 # 归一化到 [0, 1]
        
        # 转换为 Tensor [1, 3, H, W]
        input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # 3. 模型推理
        output = self.model(input_tensor)
        
        # PiDiNet 有时返回列表，取最后一个输出
        if isinstance(output, (list, tuple)):
            output = output[-1]

        # 4. 后处理 (Sigmoid + 二值化)
        # squeeze() 变成 [H, W]
        pred_mask = (torch.sigmoid(output) > self.thresh).float().squeeze().cpu().numpy()

        # 5. 提取边缘点坐标 (局部坐标)
        ys_local, xs_local = np.where(pred_mask > 0)
        
        if len(xs_local) == 0: return []

        # 6. 坐标还原为全局坐标 (向量化)
        gx = xs_local + off_x
        gy = ys_local + off_y

        # 7. 严格的几何掩膜过滤 (与 Canny/Sobel 保持一致)
        pts = []

        if len(roi_info) == 3 and not isinstance(roi_info[0], str):
            # --- 圆环过滤 ---
            (cx, cy), r_in, r_out = roi_info
            
            dists = np.hypot(gx - cx, gy - cy)
            mask = (dists >= r_in) & (dists <= r_out)
            
            valid_x = gx[mask]
            valid_y = gy[mask]
            pts = np.column_stack((valid_x, valid_y)).tolist()

        elif len(roi_info) == 6 and roi_info[0] == "line":
            # --- 直线过滤 ---
            _, p_s, p_e, d_min, d_max, nv = roi_info
            p_s = np.array(p_s)
            
            vecs = np.column_stack((gx, gy)) - p_s
            dists = np.dot(vecs, nv)
            
            mask = (dists >= d_min) & (dists <= d_max)
            
            valid_x = gx[mask]
            valid_y = gy[mask]
            pts = np.column_stack((valid_x, valid_y)).tolist()
        # print(pts)
        return pts