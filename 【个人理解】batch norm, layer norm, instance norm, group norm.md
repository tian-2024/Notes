一般来说，layer norm是用在NLP中，input的维度是`[N, F]`。

- `N`：批量大小。
- `F`：特征数量。

而其他三种norm是用在CV中，input的维度是`[N, C, H, W]`。

- `N`：批量大小（Batch Size），即一批中的样本数量。
- `C`：通道数（Channels），对于图像来说，例如RGB图像有3个通道。
- `H`：高度（Height），图像或特征图的垂直尺寸。
- `W`：宽度（Width），图像或特征图的水平尺寸。

batch norm 计算会考虑多个样本，而其他三类 norm 计算时只考虑单个样本。

因此在batch norm不适用于 batch size很小的情况，而其他三类 norm 不受影响。

下面来介绍四种 norm 的具体实现。

## layer norm

layer norm 的实现代码如下：

```python
def layer_norm(input): % input: [N, F]
    mean = input.mean(axis=1, keepdims=True)
    variance = input.var(axis=1, keepdims=True)
    normalized = (input - mean) / (variance + 1e-5)**0.5
    return normalized
```

![[Pasted image 20231128163800.png]]

## instance norm

instance norm 和 layer norm 很像，计算时都是只考虑单个样本。

不过由于他是用在CV里的，所以考虑了通道属性。

而且多个通道是分开独立计算的。

instance norm的实现代码如下：

```python
def instance_norm(input): % input: [N, C, H, W]
    mean = input.mean(axis=[2, 3], keepdims=True)
    variance = input.var(axis=[2, 3], keepdims=True)
    normalized = (input - mean) / (variance + 1e-5)**0.5
    return normalized
```


![[d67d6a0587a84a61895f88ecef32c53.jpg]]


group norm 也是用在CV里的，和 instance norm 一样都考虑了通道，并且计算时也是只考虑单个样本。不过他会把通道分组，在每个分组内计算所有特征的均值。所以像是一种 layer norm （计算单个样本的所有特征）和 instance norm（计算单个样本的单个通道的所有特征）的一种折中。

group norm 的实现代码：
```python
def group_norm(input, num_groups):
    N, C, H, W = input.shape
    groups = input.reshape(N, num_groups, C // num_groups, H, W)
    mean = groups.mean(axis=[2, 3, 4], keepdims=True)
    variance = groups.var(axis=[2, 3, 4], keepdims=True)
    normalized = (groups - mean) / (variance + 1e-5)**0.5
    return normalized.reshape(N, C, H, W)
```

![[75ef0e4f575e1c5cbf8f5c21c02b165.jpg]]

使用场景上，instance norm一般用于风格迁移，而group norm一般用于检测，分割，视频分类任务。

batch norm 是提出时间最早的 norm。计算 batch norm 时，会考虑batch 内所有样本，而不是单个样本。

具体实现如下：

```python
def batch_norm(input): % input: [N, C, H, W]
    mean = input.mean(axis=0)
    variance = input.var(axis=0)
    normalized = (input - mean) / (variance + 1e-5)**0.5
    return normalized
```

batch norm 一般会用在图像分类任务上，以及 batch_size 比较大的情况。

![[1762d9a08c2578a5bdf405c6dc0daf0.jpg]]





引子：
- 在 Transformer 模型里，需要用到一个技术，叫 layer norm。
- 用 layer norm 的目的，和用其他类型的 norm 方法类似，都是为了缩放数据到相同的分布中。
- 

1. 好难写啊，我不知道要怎么写了
2. 我不知道要了解哪些东西









这篇文章主要是为了快速了解几种常用的 normalization 方法，


1. Normalization是一种数据预处理技术，用于将数据缩放到相同的范围或分布中。
2. 目的是为了提高模型的训练速度和准确性，同时减少过拟合的风险。
3. 常见的归一化技术包括Batch Normalization、Layer Normalization、Instance Normalization和Group Normalization等。


normalization 就是对特征算均值和方差，接着每个特征减去这个均值，除以这个标准差，再乘以gamma，加上beta，




常用的 normalization 方法分为两类：
1. 样本内计算
2. 样本间计算

样本内计算的 normalization 包含三类：
1. layer norm
2. instance norm
3. group norm

样本间计算的 normalization 一般就是 batch norm。

## Layer norm

对

通常用于自然语言处理任务，如语言模型、机器翻译等。

## Batch Norm

要点：
1. 对每个特征在整个批量数据中进行标准化
2. `gamma` 和 `beta` 是可学习的缩放和平移参数

```python
def batch_norm(x, gamma, beta):
    mean = np.mean(x, axis=0)
    variance = np.var(x, axis=0)
    x_normalized = (x - mean) / np.sqrt(variance + epsilon)
    out = gamma * x_normalized + beta
    return out
```

### 为什么需要 `gamma` 和 `beta`？

1. **保持网络的表达能力**：归一化操作通过标准化输入数据（使其均值为0，方差为1）来稳定学习过程。但这种标准化可能会限制网络层学习某些特征的能力，因为它强制每个层的输出遵循相同的分布。`gamma` 和 `beta` 允许模型调整这些标准化后的数据，恢复网络的表达能力。
    
2. **加强学习的灵活性**：通过这两个参数，模型可以学习最适合数据的最优缩放和平移量，这有助于提高整体性能。
    

### 如何学习 `gamma` 和 `beta`？

`gamma` 和 `beta` 是通过网络的训练过程学习的，就像网络中的其他参数（如权重和偏置）一样：

1. **初始化**：在训练开始时，`gamma` 通常初始化为1，`beta` 初始化为0。这意味着在初始阶段，归一化层的输出不会被缩放或平移，保持了归一化的效果。
    
2. **反向传播和梯度下降**：在训练过程中，网络通过损失函数计算误差，并通过反向传播算法来计算每个参数（包括`gamma` 和 `beta`）对损失的贡献（即梯度）。然后，使用梯度下降（或其他优化算法）来更新这些参数。这意味着`gamma` 和 `beta` 会根据它们对减少网络误差的效果来进行调整。
    
3. **迭代更新**：通过多次迭代训练，`gamma` 和 `beta` 会逐渐调整到使得网络性能最优的值。

## instance norm

要点：
1. 对每个样本的每个特征通道独立进行标准化。
## layer norm

要点：
1. 对每个样本的所有特征进行标准化

```python
def layer_norm(x, gamma, beta):
    mean = np.mean(x, axis=1, keepdims=True)
    variance = np.var(x, axis=1, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(variance + epsilon)
    out = gamma * x_normalized + beta
    return out
```

