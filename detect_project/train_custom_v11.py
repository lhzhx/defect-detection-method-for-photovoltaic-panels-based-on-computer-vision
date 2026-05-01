import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import C3k2, SPPF
from ultralytics import YOLO

class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, kernel_size * kernel_size, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, autopad(kernel_size), bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.SiLU()

    def forward(self, x):
        # 为了在使用 AMP (混合精度) 时防止巨大的分辨率(1792)和注意力相乘导致 FP16 溢出 (NaN)
        # 我们在这里强行将这部分注意力计算锁定在 FP32 精度，算完后再转回原始精度
        x_float = x.float()
        weight = self.generate(x_float)
        b, c, h, w = x_float.size()
        weight = weight.view(b, -1, 1, 1).mean(dim=1, keepdim=True)
        out = x_float * weight
        return self.act(self.bn(self.conv(out))).type_as(x)

class RFCBAMConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.rfa = RFAConv(c1, c2, k, s)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c2 // 16, c2, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.rfa(x)
        # 同样为通道注意力增加 FP32 保护
        x_float = x.float()
        ca_weight = self.ca(x_float)
        return (x_float * ca_weight).type_as(x)

class LSKA(nn.Module):
    def __init__(self, dim, k_size=5):
        super().__init__()
        self.conv0_h = nn.Conv2d(dim, dim, (1, k_size), padding=(0, k_size//2), groups=dim)
        self.conv0_v = nn.Conv2d(dim, dim, (k_size, 1), padding=(k_size//2, 0), groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 1)
        
    def forward(self, x):
        attn = self.conv_spatial(self.conv0_v(self.conv0_h(x)))
        return x * nn.Sigmoid()(attn)

class SPPF_LSKA(SPPF):
    def __init__(self, c1, c2, k=5):
        super().__init__(c1, c2, k)
        self.lska = LSKA(c1 // 2, k)
        
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.lska(x)
        y2 = self.lska(y1)
        y3 = self.lska(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

# Registration logic removed, we swap modules directly.
if __name__ == "__main__":
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), "yolo11n_custom.yaml")
    
    # 1. Build the model with standard components first to leverage clean YOLO parsing
    model = YOLO(yaml_path)
    
    # *** 关键修复：加载预训练权重 ***
    # 之前因为是从头开始训练（随机初始化），导致 Precision 极低，这里加载官方权重进行迁移学习
    try:
        model.load("yolo11n.pt")
        print("Loaded pretrained weights from yolo11n.pt successfully!")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")

    net = model.model.model
    
    # 2. Dynamically inject our custom blocks into the specific layer indices
    
    def apply_rfcbam_to_layer(layer_idx):
        old_cv1 = net[layer_idx].cv1
        # Swap the bottleneck's first convolution with our RFCBAM mechanism
        new_cv1 = RFCBAMConv(old_cv1.conv.in_channels, old_cv1.conv.out_channels, 1)
        # 继承原本的 1x1 卷积权重，保留预训练特征
        if hasattr(new_cv1, 'conv') and hasattr(old_cv1, 'conv'):
            new_cv1.conv.weight.data = old_cv1.conv.weight.data.clone()
            if old_cv1.conv.bias is not None:
                new_cv1.conv.bias.data = old_cv1.conv.bias.data.clone()
        net[layer_idx].cv1 = new_cv1

    print(f"Applying RFCBAM upgrades to C3k2 modules...")
    apply_rfcbam_to_layer(8)
    apply_rfcbam_to_layer(13)
    apply_rfcbam_to_layer(28)
    
    print(f"Applying LSKA upgrades to SPPF...")
    old_sppf = net[9]
    new_sppf = SPPF_LSKA(old_sppf.cv1.conv.in_channels, old_sppf.cv2.conv.out_channels, k=5)
    # 继承 SPPF 的前后卷积层权重
    new_sppf.cv1 = old_sppf.cv1
    new_sppf.cv2 = old_sppf.cv2
    net[9] = new_sppf
    net[9].i, net[9].f, net[9].type = old_sppf.i, old_sppf.f, old_sppf.type
    
    print("Custom model created and upgraded successfully! Starting training...")
    
     # 针对小目标极度困难的情况，我们改用切片后的大特征数据集，imgsz改回标准640，批次可以拉大到16
    model.train(
        data=os.path.join(os.path.dirname(__file__), "yolo_dataset_sliced", "dataset.yaml"), # <--- 改用切片好的数据集！
        epochs=150,   
        imgsz=640,    # <--- 切片后的图像边长设定为640即可获得极高的物理分辨率
        batch=16,     # <--- 显存压力暴降，直接吃满批量！
        device=0, 
        workers=0, 
        patience=50,
        lr0=0.01,          # <--- 切片数据好学，调回标准较高学习率0.01
        lrf=0.01,     
        warmup_epochs=3.0, 
        optimizer="SGD",   
        momentum=0.937,    
        cos_lr=True,       
        weight_decay=0.0005,
        amp=True,  
        name="pv_yolov11n_custom_sliced_trained"
    )
