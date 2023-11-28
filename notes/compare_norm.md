# 【个人理解】batch norm, layer norm, instance norm, group norm

一般来说，layer norm 用在NLP任务中，input的维度是`[N, F]`。

- `N`：批量大小。
- `F`：特征数量。

而其他三种norm用在CV任务中，input的维度是`[N, C, H, W]`。

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
![20231128163800](https://github.com/jaycee-tian/Notes/assets/135324241/dbd2aa42-099f-4366-a496-1b8e5f6078b9)


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
![d67d6a0587a84a61895f88ecef32c53](https://github.com/jaycee-tian/Notes/assets/135324241/5bfbfd8e-2e76-4945-862b-24ea47fa8e59)




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
![75ef0e4f575e1c5cbf8f5c21c02b165](https://github.com/jaycee-tian/Notes/assets/135324241/c1fe70ba-b943-4090-a992-e8bd01031e72)


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
![1762d9a08c2578a5bdf405c6dc0daf0](https://github.com/jaycee-tian/Notes/assets/135324241/73ac0ee2-732d-4ee7-93fd-d417aaea258a)

