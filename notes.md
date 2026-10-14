# Let's build GPT 学习笔记（一）：Bigram 基线与张量维度

本篇对应 Andrej Karpathy 的 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) 中从字符数据到 Bigram 语言模型的阶段，重点是弄清楚**每一个轴代表什么、形状何时改变、为什么这样改变**。这里的 Bigram 是后续加入注意力机制前的基线：它只看当前位置的字符，预测下一个字符。

> 本仓库的 `gpt_dev.ipynb` 已写到字符编码、训练/验证切分、单段错位样本和 Bigram 类定义。随机批次 `get_batch`、训练循环与 `generate` 尚未写进该 notebook；下文这些部分按视频中的完整 Bigram 流程说明，不代表 notebook 已经运行到那里。

## 目录

- [一、先建立维度地图](#shape-map)
- [二、从文本到训练批次](#data-pipeline)
- [三、Bigram 查表与前向传播](#bigram-forward)
- [四、交叉熵为什么要调整形状](#cross-entropy-shapes)
- [五、训练和自回归生成](#train-and-generate)
- [六、全流程维度速查](#shape-cheatsheet)
- [七、面试问题与回答](#interview)

---

<a id="shape-map"></a>

## 一、先建立维度地图

这四个字母贯穿整篇。每看到一个张量，先问“每个轴在数什么”。

| 符号 | 含义 | 本篇示例 |
| --- | --- | --- |
| $N$ | 整份语料的字符数 | 假设约 $1{,}000{,}000$ |
| $B$ | batch size，同时处理的独立片段数 | $2$ |
| $T$ | 每个片段的 token 数，也叫上下文长度或 `block_size` | $3$ |
| $V$ | 词表大小，不同字符的总数 | 以 $65$ 为例，实际以 `vocab_size` 为准 |
| $C$ | `logits.shape` 最后一轴的长度，表示每个位置的类别打分数 | 在这个 Bigram 中 $C=V$ |

`(B,T)` 是一张整数 ID 表：行对应不同训练片段，列对应片段中的时间位置。`(B,T,V)` 则是在每个整数 ID 后面多放了一个长度为 $V$ 的打分向量；**最后一轴是候选字符类别，不是又多出来的时间步**。

在这个 Bigram 模型里，`nn.Embedding(V, V)` 的输出直接就是对下一个字符的 $V$ 个 logits。因此这里的 $C$ 恰好等于 $V$。将来做 Transformer 时，中间隐藏状态常写成 `(B,T,D)`，$D$ 是隐藏宽度；再经输出投影得到 `(B,T,V)`。不要把隐藏宽度与词表大小当成永远相同。

以 $B=2$、$T=3$、$V=65$ 为例，整条训练路径先记成：

```text
文本 → token ID (N,)
     → 输入 xb / 标签 yb (2, 3)
     → Bigram 查表 logits (2, 3, 65)
     → logits 展平 (6, 65)，targets 展平 (6,)
     → 平均交叉熵 loss ()
```

括号里的 `()` 表示标量，`(6,)` 表示长度为 6 的一维张量。

<a id="data-pipeline"></a>

## 二、从文本到训练批次

### 1. 字符编码：字符串变成一维整数张量

本例采用**字符级**分词。`chars = sorted(set(text))` 收集语料里不同的字符，再通过 `stoi` 与 `itos` 建立双向映射：

```python
chars = sorted(set(text))
vocab_size = len(chars)            # V
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
```

若原文长 $N$ 个字符，`data.shape == (N,)`。其中每个元素都是 $[0,V)$ 范围内的**整数索引**；它还不是 embedding，也没有“语义向量”这一轴。`torch.long` 对应后续 `nn.Embedding` 查表及类别索引标签需要的整型 dtype。

你的 notebook 用前 90% 的连续文本作为 `train_data`，后 10% 作为 `val_data`。验证集用于估计模型面对未用于参数更新的文本时的损失。这里的划分是按语料位置切开，不是从每个 batch 中拿出 10%。

### 2. 错位一个位置：一段文本提供多个预测任务

设 `block_size = T = 3`，取连续的四个字符 `h e l l`：

```text
原文位置：       0    1    2    3
字符：           h    e    l    l
输入 x：         h    e    l          shape (3,)
目标 y：         e    l    l          shape (3,)
预测关系：       h→e  e→l  l→l
```

代码对应 `x = data[i:i+T]` 与 `y = data[i+1:i+T+1]`。输入和标签长度一样，目标整体比输入向右偏移一个字符。你 notebook 中 `x = train_data[:block_size]`、`y = train_data[1:block_size+1]` 正是在演示这个关系。

视频中还会打印 `x[:t+1]`：第 $t$ 个预测任务从“长度 $t+1$ 的前缀”出发。但**当前 Bigram 只使用该前缀的最后一个字符**；展示前缀，是为后面能利用更长上下文的 Transformer 做铺垫。

### 3. 抽取 $B$ 个片段并堆叠

完整批处理会随机选择 $B$ 个起点；每个起点产生一对长度为 $T$ 的切片，再用 `torch.stack` 加上 batch 轴：

```python
def get_batch(split):
    source = train_data if split == 'train' else val_data
    starts = torch.randint(len(source) - block_size, (batch_size,))
    xb = torch.stack([source[i:i + block_size] for i in starts])
    yb = torch.stack([source[i + 1:i + block_size + 1] for i in starts])
    return xb, yb
```

例如 $B=2$、$T=3$，一批样本可表示为：

```text
xb  (2, 3)           yb  (2, 3)
[[15, 23, 40],       [[23, 40, 40],
 [ 8, 19, 52]]        [19, 52, 10]]
```

`xb[b,t]` 的目标是 `yb[b,t]`。同一行中的 3 个位置对应 3 次下一字符预测；两行是可并行处理的两个片段。`torch.stack` 会新增第 0 维，而不是把两段串成长为 `2*T` 的单段文本。

<a id="bigram-forward"></a>

## 三、Bigram 查表与前向传播

Bigram 的核心只有一张可训练的表：

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
logits = self.token_embedding_table(idx)
```

该表的权重形状为 `(V,V)`：第一个 $V$ 是“当前字符可能是谁”，第二个 $V$ 是“下一个字符各候选的打分”。若 `idx.shape == (B,T)`，PyTorch 会对每个整数索引取出相应的一整行，于是：

```text
idx[b,t]：一个整数，例如 15
      ↓ 取权重表第 15 行
logits[b,t,:]：长度 V 的浮点向量

idx:       (B, T)       = (2, 3)
权重表:    (V, V)       = (65, 65)
logits:    (B, T, V)    = (2, 3, 65)
```

`logits[b,t,c]` 表示第 $b$ 个片段、第 $t$ 个位置对“下一个字符是类别 $c$”给出的**未归一化打分**。它可以为负，也不要求沿类别轴求和为 1；交叉熵内部会处理从 logits 到概率的转换。此处 `C=V`，所以 notebook 中的 `B, T, C = logits.shape` 可以读成“批次、时间、候选类别”。

最容易误会的地方：虽然 `logits` 有时间轴 $T$，这个 Bigram **没有跨时间位置交换信息**。某位置的整行预测只由 `idx[b,t]` 决定；把先前字符改掉而当前字符不变，该位置的 logits 仍相同。训练时把 $B*T$ 个位置并行算出来，并不意味着模型已经学会利用长度为 $T$ 的上下文。

<a id="cross-entropy-shapes"></a>

## 四、交叉熵为什么要调整形状

### 1. 先看 PyTorch 怎样识别类别轴

`F.cross_entropy` 可以接收类别索引标签。常用二维形式要求：

```text
input logits: (N_examples, N_classes)
target IDs:   (N_examples,)
```

它**也支持高维输入**，形状为 `(N_examples, N_classes, d1, d2, ...)`，此时类别轴仍必须在第 1 维（从 0 开始数的第二维）。[PyTorch 官方形状说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy)。

我们手中的 `logits` 为 `(B,T,V)`，类别轴 $V$ 在最后。若直接把 `(B,T,V)` 和整型 `targets (B,T)` 传入，PyTorch 会把中间的 $T$ 解释成类别数、最后的 $V$ 解释成额外位置维，于是要求标签形状为 `(B,V)`。通常会因标签形状不符而报错；即使某些数值碰巧相等，轴的语义也仍然错了。

### 2. 把 batch 轴和时间轴合并

每一个 `(b,t)` 都对应一次“预测下一个字符”的分类任务。把 `B` 行、每行 `T` 个位置按同样顺序展平，总共有 $B\times T$ 次预测：

```python
B, T, C = logits.shape      # Bigram 中 C == vocab_size == V
flat_logits = logits.view(B * T, C)
flat_targets = targets.view(B * T)
loss = F.cross_entropy(flat_logits, flat_targets)
```

在 $B=2$、$T=3$、$V=65$ 的例子中：

```text
原 logits:       (2, 3, 65)
原 targets:      (2, 3)
展平 logits:     (6, 65)  → 6 次预测，每次 65 个类别打分
展平 targets:    (6,)     → 6 个正确类别的整数 ID
平均 loss:       ()       → 1 个标量
```

展平**不会抹掉预测与标签的对应关系**：两边都按相同次序排列，原来的 `(b,t)` 会成为新行号 `b*T+t`。例如 `(b=1,t=2)` 变为第 `1*3+2=5` 行；这行的 65 个 logits 对应 `flat_targets[5]`。

对于这个无需手动写 softmax 的损失，单个位置的负对数似然和默认的平均损失可写作：

$$
\ell_{b,t}=-\log p\bigl(y_{b,t}\mid x_{b,t}\bigr),
\qquad
L=\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\ell_{b,t}.
$$

其中 $p$ 是 logits 经过 softmax 后的概率。在 Bigram 阶段，条件中只有当前位置字符 $x_{b,t}$；后续注意力模型才会实际综合当前位置之前的上下文。

**另一个合法写法**是保留三维结构、把类别轴换到中间：

```python
loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
# input: (B, V, T); target: (B, T)
```

这和展平方案计算的是同一批 $B*T$ 个位置的平均交叉熵。视频使用展平，是因为每个 token 对应一行类别打分，形状更直观。若对 `permute` 后的张量继续使用 `.view(...)`，要留意内存是否连续；`.reshape(...)` 会更方便。

<a id="train-and-generate"></a>

## 五、训练和自回归生成

### 1. 训练时：所有时间位置并行计算损失

训练一步的顺序是：取一个批次 `(B,T)` → 查表得到 `(B,T,V)` → 展平算出标量 loss → 清空旧梯度 → 反向传播 → 更新权重。

```python
xb, yb = get_batch('train')
logits, loss = model(xb, yb)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

这张 `(V,V)` 表是可学习参数。每条训练样本推动“当前字符”对应的表格行，提高真实下一个字符的相对打分、调整其他候选的相对打分。验证时取 `val_data` 的批次计算 loss，不对其调用参数更新。

注意：你当前 notebook 的 `forward(self, idx, targets)` 总会展平 logits 并计算 loss，末尾还用了尚未定义的 `xb`、`yb` 和 `out`。因此直接运行当前最后一个单元会报错。这不影响理解这里的维度流，但实现生成前需要把 `targets` 改为可选，并在没有标签时保留三维 logits；视频的 Bigram 代码就是这种结构：

```python
def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)  # (B, T, V)
    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
    return logits, loss                    # 返回形状始终为 (B, T, V)
```

这里仅展示理解用的推荐写法；本篇笔记没有改动 notebook 代码。

### 2. 生成时：只取当前序列最后一个位置的预测

假设起始输入 `idx.shape == (B,1)`。一次循环中的形状如下：

```python
logits, _ = model(idx)                         # (B, T_current, V)
last_logits = logits[:, -1, :]                 # (B, V)
probs = F.softmax(last_logits, dim=-1)        # (B, V)
idx_next = torch.multinomial(probs, 1)        # (B, 1)
idx = torch.cat((idx, idx_next), dim=1)        # (B, T_current + 1)
```

`[:, -1, :]` 的意思是：保留全部 $B$ 条序列、只取最后一个时间位置、保留全部 $V$ 类的打分。`softmax(dim=-1)` 沿类别轴把打分变成概率；`multinomial` 给每条序列采样一个新字符 ID；`cat(dim=1)` 把它接到**时间轴**末尾。

```text
起始 idx:       (B, 1)
第 1 次追加后:   (B, 2)
第 2 次追加后:   (B, 3)
……
```

视频中可用 `torch.zeros((1,1), dtype=torch.long)` 作为起始上下文；其中的 0 是词表索引 0，**是否对应换行符取决于词表的排序**，不是模型天然拥有的特殊起始标记。

还有一个关键限制：Bigram 生成时虽然把完整历史不断拼回 `idx`，下一次预测实际上只取决于最后一个字符。保存增长的序列是生成接口的形式，尚不代表模型在用长程上下文。

<a id="shape-cheatsheet"></a>

## 六、全流程维度速查

| 阶段 | 张量 / 操作 | 形状（示例） | 轴的含义 |
| --- | --- | --- | --- |
| 文本编码 | `data` | `(N,)` | 全书按顺序排列的字符 ID |
| 单段输入与目标 | `x`, `y` | `(T,)`，即 `(3,)` | 目标相对输入右移一个位置 |
| 批次输入与目标 | `xb`, `yb` | `(B,T)`，即 `(2,3)` | 片段 × 时间，均为整数 ID |
| Bigram 权重 | `embedding.weight` | `(V,V)`，即 `(65,65)` | 当前字符 × 下个字符候选 |
| 模型原始输出 | `logits` | `(B,T,V)`，即 `(2,3,65)` | 片段 × 时间 × 类别打分 |
| 展平预测 | `flat_logits` | `(B*T,V)`，即 `(6,65)` | 预测位置 × 类别打分 |
| 展平目标 | `flat_targets` | `(B*T,)`，即 `(6,)` | 每个预测位置的正确类别 ID |
| 交叉熵 | `loss` | `()` | 所有预测位置的平均损失 |
| 生成取最后一步 | `logits[:, -1, :]` | `(B,V)` | 每条序列的下一字符打分 |
| 采样下一字符 | `idx_next` | `(B,1)` | 每条序列一个新 ID |
| 拼接生成序列 | `idx` | `(B,T+1)` | 时间轴增长一个位置 |

最容易记的两条线：**训练把 `(B,T)` 当作 $B*T$ 个并行分类位置；生成把 `(B,T)` 的时间轴每轮延长 1。** 前者展平是为了方便计算损失，后者保留时间轴是为了记录已经生成的序列。

<a id="interview"></a>

## 七、面试问题与回答

### 问题 1：`B`、`T`、`C` 各是什么？为什么 Bigram 中 `C=V`？

**回答：** `B` 是批次中的独立片段数，`T` 是每段的 token 数，`C` 是 logits 最后一轴的类别打分数。此处 `nn.Embedding(V,V)` 直接将当前字符 ID 映射成对下一字符 $V$ 个候选的 logits，因此 `C=V`。以后隐藏层宽度与词表大小可以不同，须经输出层投影到 $V$ 个类别。

### 问题 2：`get_batch` 中为什么输入与标签错开一个字符？

**回答：** 自回归目标是用当前位置之前可见的信息预测下一个 token。对位置 $t$，`x[t]` 是输入，`y[t]=data[i+t+1]` 是紧随其后的字符；两者形状都为 `(T,)`。堆叠后成为 `(B,T)`，这样一次前向能计算 $B*T$ 个监督位置。

### 问题 3：为什么 `(B,T,V)` 的 logits 不能原样配 `(B,T)` 的整数标签传给 `F.cross_entropy`？

**回答：** PyTorch 对高维输入把**第二维**视为类别轴。原始 `(B,T,V)` 会被解释成“$T$ 类、$V$ 个额外位置”，与 `(B,T)` 标签不对应。可把两者展平为 `(B*T,V)` 与 `(B*T,)`，或将 logits 换轴为 `(B,V,T)` 后保留标签 `(B,T)`。

### 问题 4：展平后会不会打乱样本与标签的对应？

**回答：** 不会。对两者采用相同的展平顺序，原坐标 `(b,t)` 都映射到 `b*T+t`。于是 `flat_logits[b*T+t]` 与 `flat_targets[b*T+t]` 仍是一对。这一步改的是张量视图，不是把文本重新洗牌。

### 问题 5：`logits` 是概率吗？为什么不先写 `softmax` 再交叉熵？

**回答：** logits 是未归一化的实数打分，可以为负，也不必和为 1。`F.cross_entropy` 接收 logits，内部完成数值稳定的归一化与负对数损失。生成时需要从分布采样，才显式对最后一步的 logits 做 `softmax(dim=-1)`。

### 问题 6：Bigram 有 `(B,T,V)`，为何还说它只看一个字符？

**回答：** `T` 是并行计算了多少个位置，不是单个位置使用了多少历史。查表时 `logits[b,t,:]` 只依赖 `idx[b,t]`；其他时间位置的字符不会改变这一行。真正利用上下文需要后续的注意力等跨位置计算。

### 问题 7：生成时为什么取 `logits[:, -1, :]`，又为什么沿 `dim=1` 拼接？

**回答：** 我们只需要当前序列末尾位置对**下一个字符**的打分，所以取最后时间步，得到 `(B,V)`。采样得到 `(B,1)` 后沿时间轴 `dim=1` 拼接，让输入由 `(B,T)` 变成 `(B,T+1)`；batch 轴 $B$ 和类别轴 $V$ 都不是序列增长的方向。

### 问题 8：训练阶段与生成阶段对时间轴的处理有什么不同？

**回答：** 训练时真实目标已知，可同时对 $T$ 个位置算 logits 和损失；生成时下一个字符未知，必须先采样再把它作为下一轮输入，因此按时间一步步循环。Bigram 的下一步预测只用最后字符，但这个自回归接口会延续到后续模型。

### 问题 9：当前 notebook 为什么还不能直接得到 loss 或生成文本？

**回答：** 最后单元引用了尚未构造的 `xb`、`yb`，最后的 `print(out.shape)` 也引用了未定义的 `out`。此外现有 `forward` 要求传入 `targets` 并直接返回展平后的 logits，不适合生成时用 `[:, -1, :]` 取最后时间步。先补 `get_batch` 并修正打印变量；实现生成时让 `targets=None` 可选、保留原始 `(B,T,V)` 输出。

---

资料：[Karpathy 课程与视频](https://karpathy.ai/zero-to-hero.html) · [视频配套 Bigram 源码](https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py) · [PyTorch `cross_entropy` 形状说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy)
