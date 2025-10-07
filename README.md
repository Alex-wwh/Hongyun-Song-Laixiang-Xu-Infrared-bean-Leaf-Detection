ZDAM-for-Bean-Leaf-Disease-Identification
官方 PyTorch 实现 | 论文处于投刊阶段，标题：《ZDAM: An Efficient Deep Learning Model for Bean Leaf Disease Identification 》提出 ZDAM网络模型，基于 PyTorch 框架实现四类大豆常见病害与健康状态的高精度识别，兼顾推理效率与特征捕捉能力，助力大豆病害智能化诊断与防控。

1. 研究背景与模型定位
   大豆作为全球重要的经济作物，其叶片病害（如叶霉病    锈病    花叶病    白斑病等）易导致光合效率下降、纤维品质退化，传统人工检测依赖经验判断，存在效率低、误判率高、规模化应用难的问题。本文提出ZDAM（ZFNet-CBAM-Res-unit） 模型，通过三大核心模块协同优化：1) 改进 ZFNet 网络结构，精简冗余卷积层，提升模型推理速度，适配田间移动设备部署；2) 引入 CBAM（通道注意力 - 空间注意力）模块，强化大豆病害的细微特征（如锈病的锈状斑点、叶霉病的霉层纹理）表征能力；3) 融合 Res-unit 残差单元，解决深层网络梯度消失问题，提升模型对复杂病害场景（如叶片遮挡、光照不均）的鲁棒性。模型基于 PyTorch 2.4.1 框架实现，在 “四类病害 + 健康” 共 5 类大豆叶片数据集上实现 “高效推理 + 高精度识别”，为大豆病害自动化诊断提供技术支撑
2. ZDAM 核心创新点
   2.1 改进 ZFNet：提升推理效率针对原始 ZFNet 卷积层冗余、推理耗时的问题，通过两点优化适配大豆病害识别场景：精简卷积模块：移除第 5 卷积层后的冗余全连接层，将卷积核数量从 256 减至 192，在精度损失 < 0.8% 的前提下，推理速度提升 35%；适配输入尺寸：将默认输入从 224×224 微调至 256×256，更好捕捉大豆叶片边缘的病害特征（如白斑病的不规则白斑边界），同时避免计算量过度增加。
   2.2 CBAM 注意力机制：强化病害特征捕捉将 CBAM 模块嵌入 ZFNet 的卷积层之间，通过 “通道 + 空间” 双注意力聚焦大豆病害关键区域：通道注意力：突出病害特征相关的通道（如锈病的红色通道、花叶病的黄绿差异通道），抑制背景（如土壤、杂草）干扰通道；空间注意力：定位叶片上的病害区域（如叶霉病的局部霉层、枯萎病的褐色焦斑），减少无病害区域对特征提取的影响，细粒度识别精度提升 8%。
   2.3 Res-unit 残差优化：提高模型鲁棒性在 ZFNet 深层网络中插入 Res-unit 残差单元，解决梯度消失问题：残差连接设计：通过 “shortcut 路径” 传递浅层特征，确保深层网络能学习到大豆病害的复杂特征（如花叶病的黄绿斑驳纹理）；
   动态激活调整：结合大豆叶片的叶脉分布特点，在 Res-unit 中使用 LeakyReLU 替代 ReLU，减少暗部病害（如叶背霉层）的特征丢失，鲁棒性提升 12%（针对光照变化、叶片褶皱场景）。
3. 实验数据集：四类大豆病害数据集
   3.1 数据集概况
   本研究基于四类大豆病害识别数据集，包含大豆常见病害与健康状态，数据集需联系作者获取或后续更新至公开存储平台
   数据集名称	包含类别	图像总数	图像分辨率	数据分布（训练：验证：测试）
   四类大豆数据集	叶霉病（Leaf mould）、锈病（Rust）、 花叶病 （Mosaic）、白斑病（White spot）+ 健康叶片（Healthy）+	统一 resize 至 384×384（适配 ZFNet 输入）	3：1：1（通过代码自动划分）
   3.2 数据集获取与结构
   3.2.1 下载方式
   百度网盘链接及提取码：The following is the link to my dataset. https://pan.baidu.com/s/197Lyn2TGdIjLCE2gylsiHA?pwd=krpw 提取码: krpw
   3.2.2 文件夹组织（解压后放置于项目根目录，结构如下）
   plaintext
   cotton\_disease\_dataset/  
   ├── Leaf mould/           # 大豆叶霉病叶片图像  
   ├── Rust/     #大豆锈病叶片图像  
   ├── Mosaic/        # 大豆花叶病叶片图像  
   └── White spot/              # 大豆白斑病叶片图像
4. 实验环境配置
   4.1 依赖安装
   推荐使用 Anaconda 创建虚拟环境，确保 PyTorch 版本与 CUDA 环境匹配（支持 GPU/CPU，优先推荐 GPU 加速）：
   bash

# 1\. 创建并激活虚拟环境

conda create -n vitkab-pytorch python=3.10  
conda activate vitkab-pytorch

# 2\. 安装PyTorch 2.4.1（GPU版本，需CUDA 12.1；CPU版本见下方备注）

conda install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# （备注：CPU版本安装命令）

# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# 3\. 安装其他依赖库

pip install numpy~=2.0.1 matplotlib~=3.9.5 opencv-python~=4.12.0.88  
pip install pandas~=2.3.2 pillow~=11.3.0 scikit-learn~=1.5.2  
pip install tqdm~=4.66.5 tensorboard~=2.17.0 torchmetrics~=1.4.0  
5. 代码使用说明
5.1 模型训练
运行train.py脚本启动训练，支持通过命令行参数调整训练配置，示例命令如下：
bash
python train.py \\  
--data\_dir ./cotton\_disease\_dataset \\  # 数据集根目录（解压后的路径）  
--epochs 80 \\                          # 训练轮数  
--batch\_size 32 \\                      # 批次大小（根据GPU显存调整，16/32/64）  
--lr 5e-5 \\                            # 初始学习率  
--weight\_decay 1e-5 \\                  # 权重衰减（防止过拟合）  
--save\_dir ./weights \\                 # 模型权重保存目录（.pth格式）  
--log\_interval 20 \\                    # 每20个batch打印一次训练日志  
--device GPU                           # 训练设备（GPU/CPU）  
关键参数说明
参数名	含义	默认值
--data\_dir	数据集根目录路径	./cotton\_disease\_dataset
--epochs	训练轮数	80
--batch\_size	批次大小（GPU 显存不足时可设为 16）	32
--lr	初始学习率（采用余弦退火学习率调度）	5e-5
--save\_dir	权重保存目录（自动生成，.pth 格式）	./weights
--device	训练设备（GPU 需配置 CUDA 12.1+）	GPU
5.2 模型预测
使用训练好的权重进行单张大豆叶片图像预测，运行predict.py脚本，示例命令如下：
bash
python predict.py \\  
--image\_path ./examples/cotton\_brown\_spot.jpg \\  # 输入图像路径  
--weight\_path ./weights/best\_vitkab.pth \\         # 预训练权重路径（PyTorch .pth格式）  
--device CPU                                      # 预测设备（GPU/CPU）  
预测输出示例
plaintext
输入图像路径：./examples/cotton\_brown\_spot.jpg  
预测类别：锈病（Rust） 
置信度：0.982  
预测耗时：12.3ms（CPU）/ 2.1ms（GPU）  
6. 项目文件结构
zdam-for-soybean-leaf-disease-identification/

├── soybean\_disease\_dataset/  # 五类大豆叶片数据集（需联系作者获取）

├── examples/                # 预测示例图像（如 soybean\_rust.jpg、soybean\_leaf\_mold.jpg）

├── models/                  # ZDAM 模型核心模块实现

│   ├── zfnet\_improve.py     # 改进 ZFNet 网络（精简卷积+输入适配）

│   ├── cbam\_module.py       # CBAM 注意力模块（通道+空间注意力）

│   ├── res\_unit.py          # Res-unit 残差单元（LeakyReLU 激活+残差连接）

│   └── ZDAM.py              # ZDAM 主模型（整合三大核心模块，num\_classes=5）

├── dataset/                 # 数据处理文件夹

│   └── data\_loader.py       # 大豆数据集加载、预处理（Resize 256×256）与自动划分

├── train.py                 # 模型训练脚本（含余弦退火学习率、早停机制）

├── predict.py               # 模型预测脚本（支持单图预测+置信度输出）

├── weights/                 # 模型权重保存目录（训练时自动生成，如 best\_zdam.pth）

└── README.md                # 项目说明文档（本文档）

7\. 已知问题与注意事项
框架适配：本项目仅支持 PyTorch 2.4.1 及以上版本，不兼容 TensorFlow 或低版本 PyTorch（<2.0）；
输入尺寸：模型固定输入为 384×384×3（RGB 图像），预测时会自动 resize 输入图像，建议原始图像分辨率≥384×384，避免低分辨率导致的特征丢失；
数据集扩展：如需新增大豆病害类别，需补充对应类别图像数据，并修改models/ViTKAB.py中num\_classes参数（当前为 4，新增后需同步调整）；
GPU 依赖：训练时推荐使用 CUDA 12.1 及以上版本 GPU（显存≥8GB），CPU 训练耗时较长（单轮 epoch 约 120 分钟，GPU 约 15 分钟）；
权重格式：模型权重仅支持 PyTorch 的.pth格式，不兼容 TensorFlow 的.h5格式，请勿混用跨框架权重。
8. 引用与联系方式
8.1 引用方式
@article{zdam\_soybean\_disease,

title={ZDAM: An Efficient Deep Learning Model for Soybean  Bean Disease Identification},

author={\[作者姓名，待发表时补充]},

journal={\[期刊名称，待录用后补充]},

year={2025},

note={Manuscript submitted for publication}

}

8.2 联系方式
若遇到代码运行问题、数据集获取需求或学术交流，可通过以下方式联系：
邮箱：songhongyunhuuc@yeah.net（替换为实际邮箱）
GitHub Issue：直接在本仓库提交 Issue，会在 1-3 个工作日内回复；
学术交流：可发送主题为 “ZDAM - 学术交流” 的邮件，附个人简介及交流方向，将优先回复

