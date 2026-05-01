%%{init: {"theme": "default", "themeVariables": {"background": "#ffffff"}}}%%
graph TD
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000;
    classDef data fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000;
    classDef model fill:#cce5ff,stroke:#007bff,stroke-width:2px,color:#000;
    classDef train fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#000;
    classDef output fill:#f8d7da,stroke:#dc3545,stroke-width:3px,color:#000;

    subgraph Stage1[一、 数据准备与增强阶段]
        A[原始光伏电池缺陷数据集<br>数量: 40张, 高分辨率]:::data --> B[综合数据增强<br>旋转, 翻转, 缩放平移, 高斯噪声, HSV抖动]:::data
        B --> C[增强后数据集<br>数量: 400张]:::data
    end

    subgraph Stage2[二、 图像重构与切片阶段]
        C --> D[离线切片处理 Slice Dataset<br>应对2000x2000高分辨率与极小缺陷]:::data
        D --> E[切片目标检测数据集<br>划分 Train / Val / Test]:::data
    end

    subgraph Stage3[三、 YOLOv11 模块化改造阶段]
        F[YOLOv11n 原始网络基线]:::model
        F --> G1[骨干网络升级:<br>引入 LSKA 替换标准 SPPF<br>增强多尺度与不同方向特征依赖]:::model
        F --> G2[特征融合级联:<br>基于 RFCBAM 改进 C3k2 模块<br>引入大核空间注意力与通道注意力]:::model
        F --> G3[检测头扩增:<br>增加对浅层特征 P2 的分支<br>构建四尺度 Head 针对微小目标]:::model
        G1 --> H[YOLOv11n_custom 自定义改进模型]:::model
        G2 --> H
        G3 --> H
    end

    subgraph Stage4[四、 模型训练与评估优化阶段]
        E --> I[混合训练策略<br>将切片数据输入改进后的模型]:::train
        H --> I
        I --> J[多轮训练输出与指标监控<br>获取最佳权重 best.pt]:::train
        J --> K[召回率强化与阈值扫描<br>动态寻找最优置信度 Conf 与 IoU]:::train
        K --> L[SAHI 切片辅助推理验证<br>对比原图直推与切片推理的防漏检效果]:::train
        L --> M(((最终光伏微小缺陷检测方案<br>高召回率 & 低漏检率))):::output
    end
