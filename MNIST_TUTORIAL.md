# Rust 手写数字识别 (MNIST) 极简教程

本文档将详细解释如何从零开始使用 Rust 构建一个神经网络来识别手写数字。我们将涵盖**数学原理**、**公式推导**以及**每一行代码的含义**。

---

## 1. 神经网络架构 (Architecture)

我们构建的是一个简单的 **多层感知机 (MLP)**，它包含三层：

1.  **输入层 (Input Layer)**:
    *   **大小**: 784 个节点。
    *   **原因**: MNIST 图片是 28x28 像素的灰度图。我们将二维图片展平 (Flatten) 成一维向量：$28 \times 28 = 784$。
    *   每个节点代表一个像素点的亮度值 (归一化到 0.0 ~ 1.0)。

2.  **隐藏层 (Hidden Layer)**:
    *   **大小**: 128 个节点 (这个数字是可以调整的超参数)。
    *   **作用**: 提取特征。
    *   **激活函数**: Sigmoid。

3.  **输出层 (Output Layer)**:
    *   **大小**: 10 个节点。
    *   **原因**: 数字是 0 到 9，共 10 个类别。
    *   **激活函数**: Softmax。
    *   每个节点输出该图片属于对应数字的**概率**。

### 1.1 通俗类比 (Imagine this...)
*   **输入层 (Input Layer) = 眼睛**:
    它负责“看到”图片。784个节点就像784个视网膜细胞，接收图像的明暗信息。
*   **隐藏层 (Hidden Layer) = 大脑神经元**:
    它负责“思考”。各个神经元连接在一起，尝试从原始像素中提取出特征（比如“这里有一横”，“那里有个圈”）。
*   **输出层 (Output Layer) = 嘴巴**:
    它负责“回答”。最后有10个节点，就像10个人分别举牌子打分。如果“节点5”举的牌子分数最高（比如0.9分），那就代表网络认为这是数字5。

---

## 2. 数学原理与公式 (Theory & Math)

### 2.1 符号定义
*   $X$: 输入矩阵，形状 $[N, 784]$，其中 $N$ 是批量大小 (Batch Size)。
*   $W1$: 第一层权重，形状 $[784, 128]$。
*   $b1$: 第一层偏置，形状 $[1, 128]$。
*   $W2$: 第二层权重，形状 $[128, 10]$。
*   $b2$: 第二层偏置，形状 $[1, 10]$。
*   $Z$: 线性变换结果 ($X \cdot W + b$)。
*   $A$: 激活后结果 (Activation)。

### 2.2 前向传播 (Forward Propagation)
数据从输入层流向输出层的过程。

1.  **第一层 (隐藏层)**:
    *   **线性计算**:
        $$Z_1 = X \cdot W_1 + b_1$$
        这里 $\cdot$ 代表矩阵乘法。$b_1$ 会被广播 (Broadcast) 加到每一行。
    *   **激活**: 使用 Sigmoid 函数将值压缩到 $(0, 1)$。
        $$A_1 = \sigma(Z_1) = \frac{1}{1 + e^{-Z_1}}$$

2.  **第二层 (输出层)**:
    *   **线性计算**:
        $$Z_2 = A_1 \cdot W_2 + b_2$$
    *   **激活**: 使用 Softmax 函数将输出转换为概率分布（和为1）。
        $$A_2 = \text{softmax}(Z_2)$$
        对于每一个样本 $i$ 的第 $j$ 个类别：
        $$A_{2_{i,j}} = \frac{e^{Z_{2_{i,j}}}}{\sum_{k} e^{Z_{2_{i,k}}}}$$

### 2.3 损失函数 (Loss Function)
我们需要衡量模型预测的概率 $A_2$ 与真实标签 $Y$ (One-Hot 编码) 之间的差距。
我们使用 **交叉熵损失 (Cross Entropy Loss)**：

$$Loss = - \frac{1}{N} \sum (Y_{true} \times \ln(A_2))$$

直观理解：如果真实标签是 5 (即 $Y$ 的第5个位置是1)，我们就希望 $A_2$ 的第5个位置越接近 1 越好。$\ln(1)=0$，损失最小；$\ln(0) \to -\infty$，损失最大。

### 2.3.1 Loss 是什么？(通俗解释)
**Loss (损失)** 就是 **“模型现在有多笨”** 的分数。

*   **Loss 越大**，不仅代表模型猜错了，而且代表模型**错得很离谱**（明明是 5，它信誓旦旦说是 3）。
*   **Loss 越小**，代表模型越聪明，猜得越准。
*   **训练的目标**：就是想尽一切办法（比如调整 $W$ 和 $b$）让 Loss 这个数字不仅要变小，而且要无限接近于 0。

想象你在考试。
*   **满分 (Loss = 0)**：你全对了，而且非常有信心。
*   **低分 (Loss 很大)**：你全错了，或者你对错误答案还特别自信。
神经网络的每一次“反向传播”，就是在根据这一次考试的错题（Loss），去修改脑子里的知识（权重参数），争取下次考得更好。

### 2.4 反向传播 (Backpropagation)
这是训练的核心：根据损失函数 $L$ 对参数 $W, b$ 求导，从而更新参数。我们使用**链式法则**。

**目标**: 求 $\frac{\partial L}{\partial W_2}, \frac{\partial L}{\partial b_2}, \frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial b_1}$

1.  **输出层误差 (Error at Output)**:
    交叉熵损失配合 Softmax 的导数非常简洁：
    $$dZ_2 = A_2 - Y_{true}$$
    *推导*: 这是 Softmax + CrossEntropy 的经典结论，直接记结论即可。结果是一个 $[N, 10]$ 的矩阵，代表预测值与真实值的差。

2.  **第二层梯度 (Gradients for Layer 2)**:
    $$dW_2 = \frac{1}{N} (A_1^T \cdot dZ_2)$$
    $$db_2 = \frac{1}{N} \sum dZ_2$$
    *解释*: $W_2$ 连接 $A_1$ 和 $Z_2$，所以 $dW$ 取决于输入 $A_1$ 和 输出误差 $dZ_2$。

3.  **传播回隐藏层 (Backprop to Hidden Layer)**:
    我们需要求 $dZ_1$。首先求 $dA_1$ (对 $A_1$ 的导数)：
    $$dA_1 = dZ_2 \cdot W_2^T$$
    然后乘以激活函数的导数 ($\sigma'(x) = \sigma(x)(1-\sigma(x))$)：
    $$dZ_1 = dA_1 \times \sigma'(Z_1) = dA_1 \times (A_1 \times (1 - A_1))$$
    这里的 $\times$ 是逐元素相乘。

4.  **第一层梯度 (Gradients for Layer 1)**:
    同理：
    $$dW_1 = \frac{1}{N} (X^T \cdot dZ_1)$$
    $$db_1 = \frac{1}{N} \sum dZ_1$$

5.  **参数更新 (Gradient Descent)**:
    $$W = W - \alpha \times dW$$
    $$b = b - \alpha \times db$$
    $\alpha$ 是学习率 (Learning Rate)，控制更新步长。

### 2.5 Learning Rate 是什么？(通俗解释)
**lr (学习率)** 控制着模型**“学得有多快”**。

*   **想象下山**：训练模型就像是在大雾天从山上（Loss很高的地方）往下走，找山谷（Loss最低点）。
*   **学习率就是步长**：
    *   **太大 (太快)**：你会像巨人一样一步跨几公里，可能直接跨过了山谷，甚至走到对面的山上去了（Loss 震荡不收敛）。
    *   **太小 (太慢)**：你会像蚂蚁一样挪动，要走几万年才能到山谷（训练巨慢，或者卡在半山腰下不来）。
    *   **合适**：既能稳步下降，又不会走过头。
*   在本代码中，我们设为 `0.5`，这是一个相对激进但对 MNIST 手写数字很有效的值。

### 2.6 权重 (Weights) 和 偏置 (Biases) 是什么？
*   **W (Weight, 权重) = 连接强度**:
    *   想象 Input 是“输入信号”，Hidden 是“接收神经元”。
    *   权重就是连接它们的**粗细**。如果 $W$ 很大，说明这部分输入对结果影响很大；如果 $W$ 是 0，说明完全没关系。
*   **b (Bias, 偏置) = 激活门槛**:
    *   就像每个人的笑点不一样。偏置决定了神经元**容不容易被激活**（输出 1）。
    *   即使输入信号很弱，如果偏置很高，神经元也可能输出高分。

**为什么会有两套 (W1, b1 和 W2, b2)？**
因为我们的网络有**三层**（Input -> Hidden -> Output），中间有两个“空隙”需要连接：
1.  **Input -> Hidden**: 第一跳，需要第一套参数 **$W_1, b_1$**。这层负责从像素中提取简单特征。
2.  **Hidden -> Output**: 第二跳，需要第二套参数 **$W_2, b_2$**。这层负责组合特征，给出最终答案（是0还是9）。
每一“跳” (Layer Connection) 都需要自己独立的参数矩阵。

---

## 3. 代码详解 (Code Explanation)

### 3.0 ndarray 基础概念 (Basics of ndarray)
在本项目中，我们使用 `ndarray` 库来进行矩阵运算。代码中经常出现的四个核心组件含义如下：

*   **`Array1`**: **一维数组（向量）**。
    *   *例子*: `[1.0, 2.0, 3.0]`
    *   *用途*: 用来存储标签向量 `Labels` (大小为 `[N]`) 或单个偏置 `b`。
*   **`Array2`**: **二维数组（矩阵）**。
    *   *例子*: `[[1.0, 2.0], [3.0, 4.0]]`
    *   *用途*: 用来存储输入数据 `X` (大小 `[Batch, 784]`) 或 权重矩阵 `W`。
*   **`Axis`**: **轴（维度方向）**。
    *   *`Axis(0)`*: 也就是 **第 0 维**，通常对应 **“行” (Rows)** 的方向。
        *   `sum_axis(Axis(0))`: 把所有行加在一起（压缩行），结果保留列数。
    *   *`Axis(1)`*: 也就是 **第 1 维**，通常对应 **“列” (Columns)** 的方向。
        *   `sum_axis(Axis(1))`: 把所有列加在一起（压缩列）。
*   **`s!`**: **切片宏 (Slicing Macro)**。
    *   它的语法非常像 Python 的 NumPy 切片。
    *   *例子*: `s![0..10, ..]`
        *   `0..10`: 取第 0 到 第 9 行。
        *   `..`: 取所有列。
    *   *用途*: 在 Mini-batch 训练中，我们需要从大矩阵中切出一小块数据 `x_batch = x_train.slice(s![i..end, ..])`。

### 3.0.1 随机化工具 (Randomization)
`use rand::seq::SliceRandom;`

这是一个 **Trait (特征)**，引入它之后，Rust 的数组（Slice）和向量（Vec）就会多出一些随机相关的方法。
*   **核心方法**: `shuffle`。
    *   *代码*: `indices.shuffle(&mut rand::rng());`
    *   *解释*: 它会把一个数组里的元素顺序彻底打乱。
    *   *在本代码中的作用*: 每一轮训练 (Epoch) 我们都要把数据的顺序打乱。
    *   **为什么要打乱？(Why Shuffle?)**:
        1.  **打破相关性**: 如果数据是按顺序排列的（比如先把所有的 "0" 训练完，再训练所有的 "1"），模型会倾向于在一段时间内只预测 "0"，导致梯度方向严重偏离全局最优，产生剧烈震荡。
        2.  **防止死记硬背**: 打乱顺序让模型无法通过“顺序”这个特征来作弊。
        3.  **平滑收敛**: 随机梯度下降 (SGD) 假设每一批数据 (Mini-batch) 都能代表整体数据的分布。只有随机打乱，才能保证每一批数据里各种数字都有，从而让 Loss 曲线下降得更平滑。

### 3.1 核心引擎: `src/neural_network.rs`

这个文件实现了上述所有数学公式。

```rust
// 定义神经网络结构体
// 定义神经网络结构体
pub struct NeuralNetwork {
    // === 网络超参数 ===
    pub input_size: usize,   // 输入层大小 (784)
    pub hidden_size: usize,  // 隐藏层大小 (128)
    pub output_size: usize,  // 输出层大小 (10)
    pub lr: f32,             // 学习率 (Learning Rate): 每次更新参数的步长。

    // === 网络参数 (需要训练的) ===
    pub w1: Array2<f32>,     // W1 [784, 128]: 输入层到隐藏层的权重
    pub b1: Array2<f32>,     // b1 [1, 128]:   隐藏层的偏置
    pub w2: Array2<f32>,     // W2 [128, 10]:  隐藏层到输出层的权重
    pub b2: Array2<f32>,     // b2 [1, 10]:    输出层的偏置

    // === 中间计算缓存 (用于反向传播) ===
    // 在前向传播 (Forward) 时计算并存下来，供反向传播 (Backward) 直接使用。
    // Option<> 是因为初始化时它们是空的 (None)。
    z1: Option<Array2<f32>>, // 第一层线性结果 (X * W1 + b1)
    a1: Option<Array2<f32>>, // 第一层激活结果 (Sigmoid(Z1))
    z2: Option<Array2<f32>>, // 第二层线性结果 (A1 * W2 + b2)
    a2: Option<Array2<f32>>, // 第二层激活结果 (Softmax(Z2)) - 也是最终预测值
}

impl NeuralNetwork {
    // 前向传播: 实现 Z1->A1->Z2->A2 的过程
    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        // Z1 = X * W1 + b1
        let z1 = x.dot(&self.w1) + &self.b1;
        // A1 = Sigmoid(Z1)
        let a1 = Self::sigmoid(&z1);

        // Z2 = A1 * W2 + b2
        let z2 = a1.dot(&self.w2) + &self.b2;
        // A2 = Softmax(Z2)
        let a2 = Self::softmax(&z2);

        // 保存中间结果，供反向传播使用
        self.z1 = Some(z1); self.a1 = Some(a1); 
        self.z2 = Some(z2); self.a2 = Some(a2.clone());

        a2 // 返回预测概率
    }

    // 反向传播: 计算梯度并更新权重
    pub fn backward(&mut self, x: &Array2<f32>, y_true_encoded: &Array2<f32>) {
        // dZ2 = A2 - Y
        let dz2 = a2 - y_true_encoded;

        // dW2 = A1_Transpose * dZ2
        let dw2 = a1.t().dot(&dz2) / m;
        // db2 = Sum(dZ2)
        let db2 = dz2.sum_axis(Axis(0)).insert_axis(Axis(0)) / m;

        // dA1 = dZ2 * W2_Transpose
        let da1 = dz2.dot(&self.w2.t());
        // dZ1 = dA1 * Sigmoid_Derivative(Z1)
        let dz1 = &da1 * &Self::sigmoid_derivative(z1);

        // dW1 = X_Transpose * dZ1
        let dw1 = x.t().dot(&dz1) / m;
        // db1 = Sum(dZ1)
        let db1 = dz1.sum_axis(Axis(0)).insert_axis(Axis(0)) / m;

        // 梯度下降更新: W = W - lr * dW
        self.w1 = &self.w1 - &(dw1 * self.lr);
        // ... (其他参数同理)
    }
}
```

### 3.2 数据加载: `src/lib.rs`

负责读取 MNIST 的二进制文件。MNIST 文件格式是独特的二进制格式，包含 Magic Number 和 维度信息。

```rust
// 读取 32位 大端序 (Big Endian) 整数
// 文件的前16个字节是头信息，包含：[Magic Num, 图片数量, 行数, 列数]
let num_imgs = u32::from_be_bytes(img_header[4..8].try_into().unwrap()) as usize;

// 归一化 (Normalization)
// 原始像素值是 0 (黑) 到 255 (白) 的整数。
// 我们除以 255.0，把它们变成 0.0 到 1.0 之间的浮点数。
let img_data: Vec<f32> = img_buf.into_iter().map(|x| x as f32 / 255.0).collect();
```
这一步非常关键。
**为什么要归一化？(Why Normalize?)**
1.  **数值稳定性**: 神经网络喜欢小一点的数字（通常在 0 到 1 之间）。如果输入是 0-255 这样的大数字，计算出来的中间结果 $Z$ 会非常大，导致 Sigmoid 函数输出直接饱和（变成 1 或 0），梯度消失，网络根本学不动。
2.  **加速收敛**: 小范围的数据让梯度下降走得更“顺滑”，能更快找到最优解。

### 3.3 训练主循环: `src/main.rs`

这是程序的入口。

1.  **加载数据**: 调用 `lib.rs` 中的函数读取数据。
2.  **预处理**: 将标签转为 One-Hot 编码 (例如标签 5 变成 `[0,0,0,0,0,1,0,0,0,0]`)。
3.  **训练循环 (Epoch Loop)**:
    *   **Shuffle**: 每轮开始前打乱数据顺序，防止模型记忆顺序，增加随机性 (SGD 的核心)。
    *   **Mini-batch**: 每次只取一小批数据 (如 64 张) 进行训练。相比全量训练，这能更快收敛且内存占用更小。
    *   **Forward -> Loss -> Backward**: 标准的三步走。
    *   **进度条**: 使用 `indicatif` 库实时显示训练 Loss。
4.  **保存模型**: 训练完成后将权重序列化为 JSON 文件。

### 3.4 验证脚本: `src/bin/verify_model.rs`

用于评估模型在从未见过的测试集 (Test Set) 上的表现。

1.  **加载模型**: 读取 `model.json` 恢复权重。
2.  **加载测试集**: 读取 `t10k` 开头的文件。
3.  **推理 (Inference)**: 也就是只运行 `forward` 过程，不运行 `backward`。
4.  **ArgMax**: `output` 是 10 个概率值，我们取最大值的**索引**作为预测结果。
5.  **计算准确率**: (预测正确的数量 / 总数量)。

---

## 4. 总结

你刚刚用 Rust 从零手写了一个深度学习框架的核心！虽然它只支持全连接层，但它包含了深度学习的所有关键要素：
*   **矩阵运算**: 利用 `ndarray` 进行高效计算。
*   **非线性激活**: Sigmoid 引入非线性，使网络能拟合复杂函数。
*   **梯度下降**: 通过反向传播自动调整参数。

现在的准确率应该能达到 **96%** 左右，对于这样简单的模型来说已经非常优秀了！
