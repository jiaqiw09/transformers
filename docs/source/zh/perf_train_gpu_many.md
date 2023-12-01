<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 在多个 GPU 上高效训练

在单个 GPU 上训练模型过慢或模型权重无法容纳在单个 GPU 内存中时，考虑转向多 GPU 设置可能是个可行的选择。在进行这种转变之前，务必充分探索[在单个 GPU 上进行高效训练的方法和工具](perf_train_gpu_one)，因为这个教程适用于任意数量的 GPU 上的模型训练。一旦您应用了这些策略，发现在单个 GPU 上并不足以满足您的需求，可以考虑转向多个 GPU。

从单个 GPU 转向多个 GPU 需要引入某种形式的并行处理，因为工作负载必须分配到多个资源上。可以采用多种技术来实现并行处理，比如数据并行、张量并行和管道并行。需要注意的是，并没有一种适用于所有情况的解决方案，最佳设置取决于您使用的具体硬件配置。

本指南提供了对个别并行处理类型的深入概述，以及如何结合各种技术和选择适当方法。关于分布式训练的逐步教程，请参考[🤗 Accelerate 文档](https://huggingface.co/docs/accelerate/index)。

<Tip>

虽然本指南讨论的主要概念可能适用于多个框架，但这里我们着重于基于 PyTorch 的实现。

</Tip>

在深入探讨每种技术的具体细节之前，让我们先了解一下在大型基础架构上训练大模型时的大致决策过程。

## 可扩展性策略

首先估算训练模型所需的显存量。对于托管在 🤗 Hub 上的模型，可以使用我们的[模型内存计算器](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)，该计算器可以准确计算，误差在几个百分点内。

**单节点/多 GPU 设置下的并行策略**

在单节点多 GPU 上训练模型时，选择并行策略会对性能产生显著影响。以下是各种选项的详细说明：

**情况1：模型适合单个 GPU**

如果您的模型可以轻松适配单个 GPU，有两个主要选择：

1. DDP - 分布式数据并行
2. ZeRO - 根据所用的情况和配置，这种方法可能快也可能不快，然而值得尝试一下。

**情况2: 模型不适合单个 GPU**

如果您的模型太大无法在单个 GPU 上运行，有几种替代方案可供考虑：

1. PipelineParallel (PP)
2. ZeRO
3. TensorParallel (TP)

拥有非常快的节点间连接（例如，NVLINK 或 NVSwitch）时，这三种策略（PP、ZeRO、TP）应该表现出类似的性能。然而，如果没有这些连接，PP 将比 TP 或 ZeRO 更快。TP 的张量并行的程度也可能有所不同。最好根据您的具体设置进行实验，以确定最合适的策略。

TP 几乎总是在单个节点内使用。即 TP 大小 <= 每个节点的 GPU 数量。


**情况3: 模型中最大的层不适合单个 GPU**

1. 如果您不使用 ZeRO，必须使用 TensorParallel（TP），因为单独使用 PipelineParallel（PP）无法满足大型层的需求。
2. 如果您使用 ZeRO，此外还要采用来自 [在单个 GPU 上进行高效训练的方法和工具](perf_train_gpu_one) 的技术。


**多节点/多 GPU 设置下的并行策略**

* 当您拥有快速的节点间连接（例如，NVLINK 或 NVSwitch）时，考虑使用以下选项之一：

    1. ZeRO - 因为它几乎不需要对模型进行修改
    2. PipelineParallel（PP）与TensorParallel（TP）和DataParallel（DP）的结合 - 这种方法将减少通信，但需要对模型进行重大改动

* 当您的节点间连接速度较慢，同时 GPU 内存也不足时：

    1. 使用DataParallel（DP）与PipelineParallel（PP）、TensorParallel（TP）和ZeRO的结合。

在本指南的接下来部分，我们将更深入地探讨这些不同的并行方法是如何工作的。

## Data Parallelism

即使只有两个 GPU，你也可以轻松地利用 PyTorch 内置功能提供的加速训练能力，比如 `DataParallel`（DP）和 `DistributedDataParallel`（DDP）。需要注意的是[PyTorch文档](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)推荐多 GPU 训练时优先选择 `DistributedDataParallel`（DDP），因为它适用于所有模型。让我们看看这两种方法是如何工作的，以及它们之间的区别。

### DataParallel vs DistributedDataParallel

为了了解这两种方法之间 GPU 间通信开销的主要差异，让我们来回顾每批次的处理过程：

[DDP](https://pytorch.org/docs/master/notes/ddp.html):

- 开始时，主进程将模型从 GPU 0 复制到其他 GPU。
- 然后针对每批次：
   1. 每个 GPU 直接处理其所分配的小批量数据。
   2. 在`backward`期间，一旦本地梯度准备就绪，它们会在所有进程之间进行平均。

[DP](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html):

对于每个批次：
   1. GPU 0 读取数据批次，然后将一个小批量数据发送到每个 GPU。
   2. 最新的模型从 GPU 0 复制到每个 GPU。
   3. 执行`forward`，并且来自每个 GPU 的输出被发送到 GPU 0 来计算loss。
   4. loss从 GPU 0 分发到所有 GPU，并运行`backward`。
   5. 每个 GPU 的梯度发送到 GPU 0 并进行平均。

关键差异包括：
1. DDP每批次只执行一次通信 - 发送梯度，而DP每批次执行五种不同的数据交换。
   DDP使用 [torch.distributed](https://pytorch.org/docs/master/distributed.html) 进行数据复制，而DP通过Python线程在进程内部复制数据（这会引入与全局解释器锁相关的限制）。因此，**`DistributedDataParallel`（DDP）通常比`DataParallel`（DP）更快**，除非你的 GPU 之间连接速度较慢。
2. 在 DP 下，GPU 0 执行的工作明显多于其他 GPU，导致 GPU 利用率较低。
3. DDP支持跨多台机器的分布式训练，而DP不支持。

这并不是关于DP和DDP差异的详尽列表，其他微妙之处超出了本指南的范围。
您可以通过阅读这篇[文章](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)来更深入地了解这些方法。

让我们通过一个实验来阐明DP和DDP之间的差异。我们将用一个实验来评估DP和DDP之间的差异，并增加NVLink存在的背景信息：

* 硬件: 2x TITAN RTX 24GB each + NVlink with 2 NVLinks (`NV2` in `nvidia-smi topo -m`).
* 软件: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`.

为了在其中一个基准测试中禁用 NVLink 功能，我们使用 `NCCL_P2P_DISABLE=1`。

以下是基准测试的代码和输出：

**DP**

```
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 110.5948, 'train_samples_per_second': 1.808, 'epoch': 0.69}
```

**DDP w/ NVlink**

```
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}
```

**DDP w/o NVlink**

```
rm -r /tmp/test-clm; NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

以下是相同的基准测试结果，整理成表格以方便查看：

| Type   | NVlink | Time |
| :----- | -----  | ---: |
| 2:DP   | Y      | 110s |
| 2:DDP  | Y      | 101s |
| 2:DDP  | N      | 131s |

如您所见，在这种情况下，DP 比使用 NVLink 的 DDP 慢约 10%，但比没有使用 NVLink 的 DDP 快约 15%。
真正的差异将取决于每个 GPU 需要与其他 GPU 同步的数据量 - 需要同步的数据量越大，慢速链接就会阻碍整体运行时间。

## ZeRO Data Parallelism

ZeRO驱动的数据并行（ZeRO-DP）在这篇[博文](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)中的以下图表中有所说明。

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png" alt="DeepSpeed-Image-1"/>
 </div>

虽然它可能看起来复杂，但它与 `DataParallel`（DP）是非常类似的概念。不同之处在于，每个 GPU 不再复制完整的模型参数、梯度和优化器状态，而是每个 GPU 只存储其中的一部分。然后在运行时，当特定层时需要整个层参数时，所有 GPU 将同步以提供彼此缺失的部分。

举个例子来说明这个概念，考虑一个具有 3 个层（La、Lb 和 Lc）的简单模型，每个层有 3 个参数。例如，层 La 拥有权重 a0、a1 和 a2：

```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```

如果我们有 3 个 GPU，ZeRO-DP 将模型分割到 3 个 GPU 上，如下所示：

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

在某种程度上，这与张量并行相同，是水平切片，与将整个层组放置在不同 GPU 上的垂直切片相对应。现在让我们看看它是如何工作的：

每个 GPU 将像 DP 中一样获得常规的小批次数据：

```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

输入被传递而不进行修改，就好像它们将被原始模型处理。

首先，输入到达层 `La`。在这一点上会发生什么呢？

在 GPU0 上：x0 小批量数据需要 a0、a1、a2 参数才能通过该层进行前向传播，但是 GPU0 只有 a0。它会从 GPU1 获取 a1，从 GPU2 获取 a2，将模型的各部分汇聚到一起。

与此同时，GPU1 获取另一个小批量数据- x1。GPU1 有参数 a1，但需要 a0 和 a2，因此它从 GPU0 和 GPU2 获取这些参数。GPU2 也同样接收到小批量 数据x2。它从 GPU0 和 GPU1 获取 a0 和 a1。

这样， 3 个 GPU 都重新构建了完整的张量，并使用自己的小批量数据进行了前向传播。一旦计算完成，不再需要的数据就会被丢弃 - 它只在计算过程中使用。通过预取方式高效地进行重构。

然后整个过程会按照前向传播的顺序依次进行 Lb 层，然后是 Lc 层，最后是反向传播 Lc -> Lb -> La。

<Tip>

这种机制类似于高效的团体背包策略：A 带着帐篷，B 带着炉子，C 带着斧头。每晚，他们分享各自携带的物品，并从其他人那里获取自己所需的，第二天早上他们打包自己分配到的装备，然后继续前进。这就是 ZeRO DP/Sharded DDP。将这种策略与每个人必须自己携带帐篷、炉子和斧头的简单策略进行对比（类似于 PyTorch 中的 DataParallel（DP 和 DDP）），后者效率要低得多。

</Tip>

在阅读有关此主题的文献时，您可能会遇到以下同义词：Sharded、Partitioned。

如果您仔细观察 ZeRO 如何分区模型的权重，它看起来与后面将讨论的张量并行非常相似。这是因为它分割/分区了每个层的权重，与即将讨论的垂直模型并行不同。

实现方式：

- [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/) ZeRO-DP stages 1+2+3
- [`Accelerate` 集成](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)
- [`transformers` 集成](main_classes/trainer#trainer-integrations)
- 

## 从 Naive Model Parallelism 到 Pipeline Parallelism

为了解释Pipeline parallelism,，我们首先来看一下Naive Model Parallelism（MP），也称为垂直模型并行。这种方法涉及将模型层组分配到多个 GPU 上，通过使用 `.to()` 将特定层分配给特定的 GPU。当数据流经这些层时，它会移动到与层相同的 GPU 上，而其他层保持不变。

我们将这种模型并行称为“垂直”，是因为模型通常是垂直分割的。例如，下面的图示展示了一个由 8 层组成的模型被垂直分为两个部分，将层 0-3 放在 GPU0 上，层 4-7 放在 GPU1 上：

```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        GPU0                 GPU1
```

在这个例子中，当数据从层 0 移动到层 3 时，它与普通的前向传播没有区别。但是，将数据从层 3 传递到层 4 需要将其从 GPU0 移动到 GPU1，引入了通信开销。如果参与的 GPU 在同一计算节点上（例如同一台物理机），这种复制是快速的，但是如果 GPU 分布在不同的计算节点上（例如多台机器），通信开销可能会大大增加。

接着，层 4 到 7 的工作方式与原始模型中的一样。在完成第 7 层之后，通常需要将数据发送回到第 0 层，那里有labels（或者将labels发送到最后一层）。现在可以计算loss，并且优化器可以开始工作。

Naive Model Parallelism 有一些缺点:
- **在任何给定时刻，除了一个 GPU，其他 GPU 都是空闲的**：如果使用 4 个 GPU，这几乎等同于将单个 GPU 的内存增加四倍，并忽略其他硬件。
- **设备之间数据传输的开销**：例如，4 张 6GB 的显卡能够容纳与 1 张 24GB 显卡相同大小的数据，但使用简单的模型并行，一张 24GB 的显卡会更快地完成训练，因为它没有数据复制的开销。但是，比如，如果你有 40GB 的显卡，需要容纳一个 45GB 的模型，你可以使用 4 张 40GB 的显卡（但由于梯度和优化器状态的缘故，可能会很勉强）。
- **复制共享的embeddings**：共享的embeddings可能需要在 GPU 之间来回复制。

现在你对简单的模型并行工作方式及其缺点比较熟悉，让我们来看一下pipeline并行（Pipeline Parallelism，PP）。
PP 几乎与简单的模型并行相同，但它通过将传入的批次分成微批次，并人为地创建pipeline来解决 GPU 空闲的问题，这样不同的 GPU 可以同时参与计算过程。

以下图片来自[GPipe 论文](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)，顶部展示了简单的MP，底部展示了PP：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png" alt="MP vs PP"/>
</div>

在图表的底部，你可以观察到pipeline并行（PP）方法最小化了空闲 GPU 区域的数量，被称为“bubble”。图表的两部分都展示了并行级别为 4，表示有 4 个 GPU 参与了管道。你可以看到有一个由 4 个pipeline阶段组成的前向路径（F0、F1、F2 和 F3），然后是反向路径，以相反的顺序（B3、B2、B1 和 B0）。

PP引入了一个新的超参数来调整 - `chunks`，它决定了有多少数据块以一个序列的方式通过相同的pipeline阶段发送。例如，在底部的图表中，你可以看到 `chunks=4`。GPU0 对于 chunk 0、1、2 和 3（F0,0、F0,1、F0,2、F0,3）执行相同的前向路径，然后等待其他 GPU 完成他们的工作。只有当其他 GPU 开始完成他们的工作时，GPU0 才开始再次工作，执行 chunks 3、2、1 和 0 的反向路径（B0,3、B0,2、B0,1、B0,0）。

需要注意的是，这与梯度累积步骤的概念是相同的。PyTorch 使用 `chunks`，而DeepSpeed 将相同的超参数称为梯度累积步骤。

由于存在 `chunks`，PP引入了微批次（MBS）的概念。DP将全局数据批次大小分成小批次，因此，如果你的 DP 级别为 4，全局批次大小为 1024，那么它会分成 4 个每个 256 的小批次（1024/4）。如果 `chunks`（或 GAS）的数量为 32，我们最终得到微批次大小为 8（256/32）。每个pipeline阶段一次只处理一个微批次。要计算 DP + PP 设置的全局批次大小，使用公式：`mbs * chunks * dp_degree`（`8 * 32 * 4 = 1024`）。

如果 `chunks=1`，你得到的是简单的 MP，效率较低。如果 `chunks` 的值较大，你得到的是微小的微批次大小，这也是低效的。因此，我们鼓励尝试不同的 `chunks` 值，找到能够实现最有效 GPU 利用率的值。

你可能会注意到图表上有一个无法并行化的“死”时间bubble，因为最后的 `forward` 阶段必须等待 `backward` 完成管道。找到最佳的 `chunks` 值的目的是实现所有参与 GPU 高并发利用，从而最小化这个bubble的大小。

Pipeline API 的解决方案已经在以下平台中实现：
- PyTorch
- DeepSpeed
- Megatron-LM

这些解决方案存在一些缺点：
- 它们需要对模型进行相当大的修改，因为 Pipeline 要求将模块的正常流重写为相同模块的 `nn.Sequential` 序列，这可能需要对模型设计进行更改。
- 目前 Pipeline API 的限制非常严格。如果在管道的第一个阶段中传递了一堆 Python 变量，你将不得不找到解决方法。目前，pipeline接口只接受单个张量或张量元组作为唯一的输入和输出。这些张量必须将批处理大小作为第一个维度，因为管道将将小批量划分为微批次。可能的改进正在这里讨论：https://github.com/pytorch/pytorch/pull/50693
- 在pipeline阶段级别上无法进行条件控制流，例如像 T5 这样的编码器-解码器模型需要特殊的解决方案来处理条件编码器阶段。
- 它们必须安排每个层，以便一个层的输出成为另一个层的输入。

更近期的解决方案包括：
- Varuna
- Sagemaker

我们还没有对 Varuna 和 SageMaker 进行实验，但它们的论文报告称它们已经克服了上面提到的问题清单，并且它们只需要对用户模型进行较小的更改。

实现方案：
- [PyTorch](https://pytorch.org/docs/stable/pipeline.html)（pytorch-1.8 中首次提供支持，逐渐在 1.9 以及更多的 1.10 版本中得到改进）。一些[例子](https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/pipeline/pipe.py)
- [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)有一个内部实现 - 不需要API。
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972) - 这是一种专有解决方案，只能在 AWS 上使用。
- [OSLO](https://github.com/tunib-ai/oslo) - 这是基于 Hugging Face Transformers 实现的。

🤗 Transformers 的现状：截止到目前，没有任何模型支持完整PP。GPT2 和 T5 模型支持简单的MP。主要障碍在于无法将模型转换为 `nn.Sequential` 并且要求所有输入都是张量。这是因为当前的模型包含许多功能，使得转换变得非常复杂，并需要将其移除才能实现。

DeepSpeed 和 Megatron-LM 的集成已经在[🤗 Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/deepspeed)中可用。

其他方法：

DeepSpeed、Varuna 和 SageMaker 使用[Interleaved Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html)的概念。

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-sagemaker-interleaved-pipeline.png" alt="Interleaved pipeline execution"/>
</div>

这里通过优先考虑后向传递来进一步减少空闲时间（bubble）。Varuna 进一步尝试通过使用模拟来发现最高效的调度来改进时间表。

OSLO 基于 Transformers 实现了pipeline并行，而没有进行 `nn.Sequential` 转换。

## Tensor Parallelism

在张量并行中，每个 GPU 处理张量的一部分，并且仅在需要时对整个张量进行聚合操作。
为描述这种方法，本指南的这部分依赖于[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 论文中的概念和图表：[Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)。

任何 Transformer 的主要组件是一个完全连接的 `nn.Linear`，然后是一个非线性激活 `GeLU`。
根据 Megatron 论文的符号表示，它的点乘部分可以写成 `Y = GeLU(XA)`，其中 `X` 是输入向量，`Y` 是输出向量，`A` 是权重矩阵。

如果我们以矩阵形式看计算，你可以看到矩阵乘法可以在多个 GPU 之间进行拆分：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_gemm.png" alt="Parallel GEMM"/>
</div>

如果我们在 `N` 个 GPU 上按列拆分权重矩阵 `A` 并并行执行矩阵乘法 `XA_1` 到 `XA_n`，那么我们将得到 `N` 个输出向量 `Y_1, Y_2, ..., Y_n`，这些可以独立地输入到 `GeLU` 中：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-independent-gelu.png" alt="Independent GeLU"/>
</div>

利用这个原理，我们可以更新任意深度的多层感知器，而无需在最后一步之前同步多个 GPU，直到我们需要从片段中重构输出向量。Megatron-LM 论文的作者为此提供了一个有用的图示：

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_shard_processing.png" alt="Parallel shard processing"/>
</div>

并行化多头注意力层甚至更简单，因为拥有多个独立的注意力头，它们已经天生是并行的！

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_self_attention.png" alt="Parallel self-attention"/>
</div>

特殊考虑：TP 需要非常快速的网络，因此不建议在多个节点上进行 TP。实际上，如果一个节点有 4 个 GPU，那么最高的 TP 程度就是 4。如果您需要 8 的 TP 程度，你需要使用至少有 8 个 GPU 的节点。

这部分是基于原始更为[详细的 TP 概述](https://github.com/huggingface/transformers/issues/10321#issuecomment-783543530)，由 [@anton-l](https://github.com/anton-l) 提供。

替代名称：
- DeepSpeed 称其为[张量切片](https://www.deepspeed.ai/training/#model-parallelism)

实现：
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 有一个内部实现，因为它非常适应特定模型
- [parallelformers](https://github.com/tunib-ai/parallelformers)（目前仅支持推理）
- [SageMaker](https://arxiv.org/abs/2111.05972) - 这是一种专有解决方案，只能在 AWS 上使用。
- [OSLO](https://github.com/tunib-ai/oslo) 基于 Transformers 实现了张量并行。

SageMaker将TP与DP结合，以实现更高效的处理。

🤗 Transformers 状态：
- 核心库：核心库尚未实施此功能。
- 但如果需要推理，[parallelformers](https://github.com/tunib-ai/parallelformers) 提供了大多数模型的支持。在此功能被实现到核心库之前，可以使用该工具。希望训练模式也会得到支持。
- Deepspeed-Inference 也支持BERT、GPT-2和GPT-Neo模型的超快速 CUDA 内核推理模式，详情请见[这里](https://www.deepspeed.ai/tutorials/inference-tutorial/)

🤗 Accelerate 与 [Megatron-LM 的 TP](https://huggingface.co/docs/accelerate/v0.23.0/en/usage_guides/megatron_lm)集成。

## Data Parallelism + Pipeline Parallelism

以下来自 DeepSpeed [pipeline tutorial](https://www.deepspeed.ai/tutorials/pipeline/) 的图示展示了如何将DP与PP结合使用。


<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png" alt="DP + PP-2d"/>
</div>

这里重要的是要看到 DP rank 0 看不到 GPU2，而 DP rank 1 看不到 GPU3。对于 DP 来说，只有 GPU0 和 GPU1，它将数据传递，就好像只有 2 个 GPU。GPU0 “秘密地”将一部分负载转移到 GPU2 使用 PP。GPU1 也通过将 GPU3 编入队伍做同样的操作。

因为每个维度至少需要 2 个 GPU，所以这里至少需要 4 个 GPU。

实施情况：
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/tunib-ai/oslo)

🤗 Transformers 状态：尚未实施。

## Data Parallelism + Pipeline Parallelism + Tensor Parallelism

要实现更高效的训练，使用了 3D 并行，将 PP 与 TP 和 DP 结合使用。以下是对应的示意图。

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png" alt="dp-pp-tp-3d"/>
</div>

这张图是来自 [3D parallelism: Scaling to trillion-parameter models](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/) 的博客文章，也值得一读。

因为每个维度至少需要 2 个 GPU，所以这里至少需要 8 个 GPU。

实施情况：
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - DeepSpeed 还包括了更高效的 DP，称为 ZeRO-DP。
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/tunib-ai/oslo)

🤗 Transformers 状态：尚未实施，因为我们尚未使用 PP 和 TP。

## ZeRO Data Parallelism + Pipeline Parallelism + Tensor Parallelism

DeepSpeed 的主要功能之一是 ZeRO，它是 DP 的super-scalable扩展。已在 [ZeRO Data Parallelism](#zero-data-parallelism) 中讨论过。通常情况下，它是一个独立的功能，不需要 PP 或 TP。但它可以与 PP 和 TP 结合使用。

当 ZeRO-DP 与 PP（和可选的 TP）结合时，通常只启用 ZeRO 阶段 1（优化器分片）。

尽管理论上可以将 ZeRO 阶段 2（梯度分片）与 Pipeline Parallelism 结合使用，但这将对性能产生负面影响。每个微批次都需要额外的 reduce-scatter 集合，以汇总梯度，然后进行分片，这会增加潜在的显著通信开销。由于 Pipeline Parallelism 的性质，会使用较小的微批次，重点是尝试平衡算术强度（微批次大小）和最小化 Pipeline bubble（微批次数量）。因此，这些通信成本将影响性能。

此外，由于 PP，层次会比正常情况下少，因此内存节省并不会很大。PP 已经将梯度大小减少了 ``1/PP``，因此相比于纯粹的 DP，再加上梯度分片的节省就不那么显著了。

ZeRO 阶段 3 也不是一个好选择，原因同样是需要更多的节点间通信。

而且，由于我们有 ZeRO，另一个好处是 ZeRO-Offload。由于这是阶段 1，优化器状态可以转移到 CPU 上。

实现方式：
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) 和 [BigScience 的 Megatron-Deepspeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)，这是前者存储库的 fork。
- [OSLO](https://github.com/tunib-ai/oslo)

重要论文：
- [使用 DeepSpeed 和 Megatron 训练 Megatron-Turing NLG 530B，一个大规模生成式语言模型](https://arxiv.org/abs/2201.11990)

🤗 Transformers 状态：尚未实现，因为我们还没有 PP 和 TP。

## FlexFlow

[FlexFlow](https://github.com/flexflow/FlexFlow) 采用了稍有不同的方法来解决并行化问题。

论文：["Beyond Data and Model Parallelism for Deep Neural Networks" by Zhihao Jia, Matei Zaharia, Alex Aiken](https://arxiv.org/abs/1807.05358)

它实现了一种4D并行：Sample-Operator-Attribute-Parameter。

1. Sample = 数据并行（逐样本并行）
2. Operator = 将单个操作并行化为多个子操作
3. Attribute = 数据并行（按长度并行）
4. Parameter = 模型并行（无论是水平还是垂直维度）

例子：
* Sample

以10个序列长度为512的批次为例。如果我们按样本维度将它们并行化到两个设备，那么10 x 512 将变成 5 x 2 x 512。

* Operator

如果我们进行层归一化，首先计算标准差，然后计算均值，最后归一化数据。操作符并行化允许同时计算标准差和均值。因此，如果我们按操作符维度将它们并行化到两个设备（cuda:0，cuda:1），首先将输入数据复制到两个设备，cuda:0 在计算标准差，cuda:1 同时计算均值。

* Attribute

我们有10个长度为512的批次。如果我们按属性维度将它们并行化到两个设备，10 x 512 将变成 10 x 2 x 256。

* Parameter

这类似于张量模型并行化或者朴素的分层模型并行化。

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-flexflow.jpeg" alt="flex-flow-soap"/>
</div>

该框架的重要意义在于它将资源（1）GPU/TPU/CPU vs（2）RAM/DRAM vs（3）快速内部连接/慢速跨连接，并自动通过算法优化决定在何处使用哪种并行化。

非常重要的一点是，FlexFlow 设计用于优化具有静态和固定工作负载的 DNN 并行化，因为具有动态行为的模型可能在迭代中更喜欢不同的并行化策略。

因此，它的承诺非常吸引人 - 它在所选择的集群上运行了30分钟的模拟，并给出了最佳策略来利用特定环境。如果添加/移除/替换任何部分，它会重新运行并重新优化该计划。然后您可以进行训练。不同的设置将有自己的定制优化。

🤗 Transformers 状态: Transformers 模型可以通过 [transformers.utils.fx](https://github.com/huggingface/transformers/blob/master/src/transformers/utils/fx.py) 进行 FX-跟踪，这是 FlexFlow 的先决条件，但是需要在 FlexFlow 方面进行更改以使其与 Transformers 模型配合。