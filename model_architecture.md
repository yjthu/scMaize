
**scMaize 单细胞玉米基础模型的架构与预训练框架。**
该模型采用基于 Transformer 的深度学习架构，旨在捕捉玉米单细胞转录组中的复杂基因调控模式与功能特征。

* **1. 词表构建与动态采样 (Vocab Building)**：模型构建了一个包含 15,000 个基因的混合词表，由 13,000 个高可变基因 (HVGs) 和 2,000 个基于基因本体 (GO) 补充的功能性基因组成。在训练过程中，模型为每个细胞随机采样 $L=2,048$ 个基因，并根据其在总词表中的 HVG 排名进行排序，随后应用紧凑排名编码 (Tight rank encoding) 以保留基因的重要性层级信息。
* **2. 嵌入融合 (Embedding Fusion)**：模型通过四个分支集成多模态信息：(i) **基因身份 (Gene IDs)** 通过基因嵌入层与层归一化 (LayerNorm) 处理；(ii) **表达丰度 (Expression Values)** 经过一个由两层线性变换与 GELU 激活函数组成的 MLP 投影层，并进行层归一化；(iii) **批次标签 (Batch Labels)** 通过批次嵌入转化为条件偏置 (Conditioning Bias)，注入模型以感知实验背景；(iv) **功能嵌入 (Optional GO variant)** 为可选模块，通过功能投影层引入基因功能先验知识。
* **3. Transformer 编码器堆栈 (Transformer Encoder Stack)**：核心骨干网络由 $N$ 层编码器块组成。每层包含多头自注意力机制 (Multi-Head Self-Attention, MHA) 和前馈网络 (Feed-Forward Network, FFN)，辅以残差连接与层归一化。序列头部引入了一个可学习的 **[CLS] Token**，用于聚合全细胞层级的全局特征。
* **4. 多任务输出 (Multi-task Outputs)**：
* **a. 掩码基因建模 (Masked Gene Modeling, MGM)**：模型通过掩码隐状态预测基因表达值。为了应对单细胞数据的稀疏性，模型采用了 **加权 MGM 损失函数 (Weighted MGMLoss)**，对非零表达值赋予更高的权重（如 5.0 倍），以防止模型由于零值过多而产生平凡解并保留生物学动态范围。
* **b. 细胞表征 (Cell Representation)**：经过输出归一化处理后的 [CLS] 隐状态 ($H_{CLS}$) 被提取为最终的**细胞嵌入 (Cell Embedding)**，作为代表该细胞生物学状态的全局指纹，用于下游的聚类、细胞类型鉴定及发育轨迹分析。