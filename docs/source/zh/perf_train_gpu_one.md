<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 单个GPU上高效训练的方法和工具

本指南演示了可以用来增加模型训练效率的实用技术，包括优化内存利用、加速训练或两者兼顾。如果你想了解训练过程中 GPU 的利用情况，请先参考 [模型训练剖析](model_memory_anatomy) 概念指南。本指南侧重于实用技巧。

<Tip>

如果您可以访问带有多个 GPU 的计算机，这些方法仍然有效，并且您可以利用 [多 GPU 章节](perf_train_gpu_many) 中详细介绍的其他方法。

</Tip>

在训练大型模型时，同时应考虑两个方面：

* 数据吞吐量/训练时间
* 模型性能

最大化吞吐量（样本量/秒）有助于降低训练成本。通常通过尽可能充分利用 GPU来将GPU 内存填充到极限来实现这一点。如果期望的数据批量大小超过了 GPU 内存的限制，内存优化技术（例如gradient accumulation）可以解决此类问题。

然而，如果内存大小足够容纳期望的数据批量大小，就没有理由应用内存优化技术，因为它们可能会减慢训练速度。可以使用大批量数据，并不一定意味着应该这样做。作为超参数调整的一部分，您应该确定哪种数据批量大小能产生最佳结果，然后相应地的优化资源。

本指南涵盖的方法和工具可以根据它们对训练过程的影响进行分类：

| Method/tool                                                | Improves training speed | Optimizes memory utilization |
|:-----------------------------------------------------------|:------------------------|:-----------------------------|
| [Batch size choice](#batch-size-choice)                    | Yes                     | Yes                          |
| [Gradient accumulation](#gradient-accumulation)            | No                      | Yes                          |
| [Gradient checkpointing](#gradient-checkpointing)          | No                      | Yes                          |
| [Mixed precision training](#mixed-precision-training)      | Yes                     | (No)                         |
| [Optimizer choice](#optimizer-choice)                      | Yes                     | Yes                          |
| [Data preloading](#data-preloading)                        | Yes                     | No                           |
| [DeepSpeed Zero](#deepspeed-zero)                          | No                      | Yes                          |
| [torch.compile](#using-torchcompile)                       | Yes                     | No                           |

<Tip>

注意：在使用混合精度时，对于小模型和大批量数据，会有一些内存节省，但对于大模型和小批量数据，内存使用量会更大。

</Tip>

您可以组合使用这些方法以获得累积效果。无论您是使用[`Trainer`]训练模型还是在纯PyTorch原生训练循环，都可以配置这些优化。在这种情况下，您可以使用 🤗 Accelerate 配置这些优化方法。

如果这些方法没有带来足够的收益，您可以尝试以下选项：
* [考虑使用具有高效预构建软件的自定义Docker容器](#efficient-software-prebuilds)
* [考虑使用混合专家（MoE）模型](#mixture-of-experts)
* [将您的模型转换为BetterTransformer以利用PyTorch native attention](#using-pytorch-native-attention)

最后，即使所有这些方法仍然不足以解决问题，即使在切换到服务器级GPU（如A100）后，也可以考虑迁移到多GPU设置。这些方法在多GPU设置中仍然有效，此外，您可以利用[多GPU章节](perf_train_gpu_many)中列出的其他并行技术。

## Batch size 选择

为了实现最佳性能，首先要确定合适的数据批量大小。建议使用2^N的批量大小和输入/输出神经元数量。通常它是8的倍数，但取决于所使用的硬件和模型的数据类型。

参考NVIDIA关于[输入/输出神经元数量](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features)和[批量大小](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size)的建议，对于涉及GEMMs（通用矩阵乘法）的完全连接层。

[张量核心要求](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc)根据数据类型和硬件定义乘数。例如，对于fp16数据类型，建议是8的倍数，而A100 GPU建议使用64的倍数。

对于较小的参数，还需要考虑[维度量化效果](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization)。这是其中tiling发生的地方，正确的乘数可以带来显著的速度提升。

## Gradient Accumulation

**梯度累积** 方法旨在逐步计算梯度，而不是一次性计算整个批次的梯度。这种方法涉及通过模型进行迭代计算小批次的前向和反向传播，并在此过程中累积梯度。一旦累积了足够数量的梯度，便执行模型的优化步骤。通过使用梯度累积，可以将**有效批量大小**增加到 GPU 内存容量所限制的范围之外。但是，需要注意的是梯度累积引入的额外前向和后向传递可能会减慢训练过程。

您可以通过向 [`TrainingArguments`] 添加 `gradient_accumulation_steps` 参数来启用梯度累积：

```py
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
```

在上面的例子中，你的有效数据批量大小是 4。

或者，使用 🤗 Accelerate 来完全控制训练循环。在本指南的[后面部分](#using-accelerate)找到 🤗 Accelerate 示例。

虽然建议尽可能充分利用 GPU，但大量的梯度累积步骤可能会导致训练速度明显放缓。考虑以下示例。假设 `per_device_train_batch_size=4`（不使用梯度累积）已经达到了 GPU 的限制。如果您想要使用大小为 64 的批次进行训练，不要将 `per_device_train_batch_size` 设置为 1，也不要将 `gradient_accumulation_steps` 设置为 64。相反，保持 `per_device_train_batch_size=4`，并设置 `gradient_accumulation_steps=16`。这样可以得到相同的有效批量大小，同时更好地利用可用的 GPU 资源。

了解更多信息，请参考关于 [RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004392537) 和 [A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1005033957) 的批量大小和梯度累积基准数据。

## Gradient Checkpointing

即使将批量大小设置为 1 并使用梯度累积，一些大模型仍可能面临内存问题。这是因为还有其他组件也需要内存存储。

保存前向传递中的所有激活值以在后向传递期间计算梯度可能导致显著的内存开销。另一种方法不保存激活值，并在后向传递期间重新计算它们，但这样会引入相当大的计算开销并减慢训练过程。

**梯度checkpoint** 在这两种方法之间寻求折中，通过在计算图中选择性保存激活值，只需重新计算部分激活值以获得梯度。要深入了解梯度checkpoint，请参考 [这篇精彩的文章](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)。

要在 [`Trainer`] 中启用梯度checkpoint，请向 [`TrainingArguments`] 传递相应的标志：

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)
```

或者，使用 🤗 Accelerate - 在本指南的[后续章节](#using-accelerate)找到 🤗 Accelerate 示例。

<Tip>

虽然`gradient checkpointing`可能会提高内存效率，但会使训练速度大约减慢 20%。

</Tip>

## 混合精度训练

**混合精度训练** 是一种旨在通过利用低精度数值格式处理某些变量，从而优化训练模型的计算效率的技术。传统上，大多数模型使用 32 位浮点精度（fp32 或 float32）来表示和处理变量。然而，并非所有变量都需要这种高精度级别才能获得准确的结果。通过将某些变量的精度降低到较低的数值格式，如 16 位浮点（fp16 或 float16），我们可以加快计算速度。因为在这种方法中，一些计算是在半精度下进行的，而一些仍然是在完整精度下进行的，所以这种方法被称为混合精度训练。

最常见的混合精度训练是使用 fp16（float16）数据类型，但某些 GPU 架构（如安培架构）提供了 bf16 和 tf32（CUDA 内部数据类型）数据类型。查看 [NVIDIA 博客](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/) 了解这些数据类型之间的区别。

### fp16

混合精度训练的主要优势来自在半精度（fp16）下保存激活值。但尽管梯度也是以半精度计算，它们在优化步骤中会转换回完整精度，因此在这一步并未节省内存。混合精度训练可以加快计算，但也可能导致 GPU 内存的更多利用，特别是对于小批量大小。这是因为现在模型同时以 16 位和 32 位精度存在于 GPU 上（GPU 上原模型的 1.5 倍）。

要启用混合精度训练，请将 `fp16` 标志设置为 `True`：

```py
training_args = TrainingArguments(per_device_train_batch_size=4, fp16=True, **default_args)
```

如果你更喜欢使用 🤗 Accelerate，请在本指南的[后续部分](#using-accelerate)找到 🤗 Accelerate 示例。

### BF16

如果您有安培架构或更新的硬件，可以在模型混合精度训练和评估中使用 bf16。虽然 bf16 的精度比 fp16 更差，但它具有更大的动态范围。在 fp16 中，您可以拥有的最大数是 `65535`，任何超过这个范围的数字都会导致溢出。而 bf16 数可以达到 `3.39e+38`(!)，这与 fp32 大致相同 - 因为两者都使用 8 位来表示数值范围。

您可以通过以下方式在 🤗  Trainer 中启用 BF16：

```python
training_args = TrainingArguments(bf16=True, **default_args)
```

### TF32

安培架构硬件使用了一种神奇的数据类型叫做 tf32。它具有与 fp32 相同的数值范围（8 位），但精度不是 23 位，而是只有 10 位（与 fp16 相同），总共只使用了 19 位。它在这种意义上是“神奇的”，因为您可以使用正常的 fp32 训练和/或推理代码，并通过启用 tf32 ，可以获得高达 3 倍的吞吐量提升。你只需要在代码中添加以下内容：

```
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

CUDA 将在可能的情况下自动切换到使用 tf32 而不是 fp32，假设使用的 GPU 属于安培架构系列。

根据 [NVIDIA 研究](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)，大多数机器学习训练负载在 tf32 训练中显示出与 fp32 相同的复杂度和收敛性。如果您已经在使用 fp16 或 bf16 混合精度，这也可能有助于吞吐量。

您可以在 🤗  Trainer 中启用这种模式：

```python
TrainingArguments(tf32=True, **default_args)
```

<Tip>

tf32 不能直接通过 `tensor.to(dtype=torch.tf32)` 开永，因为它是一个内部 CUDA 数据类型。您需要 `torch>=1.7` 才能使用 tf32 数据类型。

</Tip>

关于tf32与其他精度的更多信息，请参考以下基准测试：
[RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803) 和
[A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1004543189)。

## Flash Attention 2

你可以通过在 transformers 中使用 Flash Attention 2 来加快训练吞吐量。查看 [单 GPU 章节](./perf_infer_gpu_one#Flash-Attention-2) 中相应部分，了解如何加载带有 Flash Attention 2 模块的模型。

## Optimizer 选择

用于训练 Transformer 模型最常用的优化器是 Adam 或 AdamW（带有权重衰减的 Adam）。Adam 通过存储先前梯度的滚动平均值实现良好的收敛性；但是，它会增加大约等于模型参数数量的额外内存占用。为了解决这个问题，您可以使用另一种优化器。例如，如果您已安装了 [NVIDIA/apex](https://github.com/NVIDIA/apex)，`adamw_apex_fused` 将为你提供所有支持的 AdamW 优化器中最快的训练体验。

[`Trainer`] 集成了各种优化器，可以直接使用：`adamw_hf`、`adamw_torch`、`adamw_torch_fused`、`adamw_apex_fused`、`adamw_anyprecision`、`adafactor` 或 `adamw_bnb_8bit`。通过第三方实现，也可以插入更多的优化器。

让我们更仔细地看看两种替代 AdamW 优化器：
1. `adafactor`，它在 [`Trainer`] 中可用。
2. `adamw_bnb_8bit` 也在 Trainer 中可用，但下面提供了第三方集成示例。

举例来说，对于一个 30 亿参数的模型，比如 "t5-3b"：
* 标准的 AdamW 优化器将需要 24GB 的 GPU 内存，因为它为每个参数使用了 8 字节（8*3 => 24GB）。
* Adafactor 优化器将需要超过 12GB。它为每个参数使用略多于 4 字节，所以是 4*3，再加上一些额外内存。
* 如果所有优化器状态都被量化的话，8 位 BNB 量化优化器只需使用 (2*3) 6GB。

### Adafactor

Adafactor 不会为权重矩阵中的每个元素存储滚动平均值。相反，它保留聚合信息（逐行和逐列的滚动平均和），大幅减少了内存占用。但是，与 Adam 相比，Adafactor 在某些情况下可能收敛速度较慢。

您可以通过在 [`TrainingArguments`] 中设置 `optim="adafactor"` 来切换到 Adafactor：

```py
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adafactor", **default_args)
```

结合其他方法（梯度累积、梯度checkpointing和混合精度训练），在保持吞吐量的同时，您可以看到高达 3 倍的改进！然而，如前所述，Adafactor 的收敛速度可能不如 Adam。

### 8-bit Adam

与 Adafactor 聚合优化器状态不同，8 位 Adam 保留完整状态并对其进行量化。量化意味着以较低的精度存储状态，并仅在优化时对其进行反量化。这类似于混合精度训练的思想。

使用`adamw_bnb_8bit`， 您可以简单的在[`TrainingArguments`]设置`optim="adamw_bnb_8bit"` 

```py
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adamw_bnb_8bit", **default_args)
```

然而，我们也可以以演示为目，使用第三方实现的 8 位优化器，以了解如何进行集成。

首先，按照 GitHub [repo](https://github.com/TimDettmers/bitsandbytes) 中的安装指南安装实现 8 位 Adam 优化器的 `bitsandbytes` 库。

接下来需要初始化优化器。这涉及两个步骤：
* 首先，将模型的参数分为两组 - 一组应用权重衰减，另一组则不应用。通常，偏置和层归一化参数不进行权重衰减。
* 然后进行一些参数设置，以使用与之前使用的 AdamW 优化器相同的参数。

```py
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)

decay_parameters = get_parameter_names(model, [nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": training_args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer_kwargs = {
    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    "eps": training_args.adam_epsilon,
}
optimizer_kwargs["lr"] = training_args.learning_rate
adam_bnb_optim = bnb.optim.Adam8bit(
    optimizer_grouped_parameters,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    lr=training_args.learning_rate,
)
```

最后，将自定义的优化器作为参数传递给 `Trainer`：

```py
trainer = Trainer(model=model, args=training_args, train_dataset=ds, optimizers=(adam_bnb_optim, None))
```

结合其他方法（梯度累积、梯度checkpointing和混合精度训练），您可以期望获得大约 3 倍的内存改进，甚至数据吞吐量会略高于使用 Adafactor 。

### multi_tensor

pytorch-nightly 引入了 `torch.optim._multi_tensor`，它应该显著加速具有大量小特征张量的优化器。它最终应该会成为默认设置，但如果您想更早尝试，可以查看这个 GitHub [issue](https://github.com/huggingface/transformers/issues/9965)。


## 数据预加载

达到良好的训练速度的一个重要要求是能够以 GPU 能够处理的最大速度提供数据。默认情况下，所有操作都发生在主进程中，这可能无法从磁盘快速读取数据，从而造成瓶颈，导致 GPU 利用不充分。可以配置以下参数以减少瓶颈：

- `DataLoader(pin_memory=True, ...)` - 确保数据预加载到 CPU 的固定内存中，通常会使得从 CPU 到 GPU 内存的传输速度更快。
- `DataLoader(num_workers=4, ...)` - 启动多个工作进程以更快地预加载数据。在训练过程中，观察 GPU 利用率统计；如果远低于 100%，尝试增加工作进程的数量。当然，问题可能出在其他地方，所以增加工作进程并不一定会带来更好的性能。

当使用 [`Trainer`] 时，相应的 [`TrainingArguments`] 包括：`dataloader_pin_memory`（默认为 `True`），以及 `dataloader_num_workers`（默认为 `0`）。


## DeepSpeed ZeRO

DeepSpeed 是一个与 🤗 Transformers 和 🤗 Accelerate 集成的开源深度学习优化库。它提供了广泛的功能和优化，旨在改善大规模深度学习训练的效率和可扩展性。

如果您的模型适合单个 GPU，并且有足够的空间适应小批量数据，那么您不需要使用 DeepSpeed，因为它只会减慢速度。然而，如果模型不能适配单个 GPU 或者不能处理小批量数据，您可以利用 DeepSpeed ZeRO + CPU Offload来处理，对于更大的模型可以使用NVMe Offload 。在这种情况下，您需要分别 [安装库](main_classes/deepspeed#installation)，然后按照指南创建配置文件并启动 DeepSpeed：

* 对于 DeepSpeed 与 [`Trainer`] 的深度集成指南，请查阅 [相关文档](main_classes/deepspeed)，特别是[单个 GPU 部署章节](main_classes/deepspeed#deployment-with-one-gpu)。在`notebook`中使用 DeepSpeed 需要进行一些调整，请参阅 [相关指南](main_classes/deepspeed#deployment-in-notebooks)。
* 如果您希望使用 🤗 Accelerate，请参考 [🤗 Accelerate DeepSpeed 指南](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)。


## 使用 torch.compile

PyTorch 2.0 引入了一个新的编译函数，不需要对现有 PyTorch 代码进行任何修改，但可以通过添加一行代码来优化您的代码：`model = torch.compile(model)`。

如果使用 [`Trainer`]，您只需要在 [`TrainingArguments`] 中传递 `torch_compile` 选项：

```python
training_args = TrainingArguments(torch_compile=True, **default_args)
```

`torch.compile` 使用 Python 的frame评估 API 从现有的 PyTorch 程序自动创建图。在捕获图后，可以部署不同的后端以将图降低到优化引擎。您可以在 [PyTorch 文档](https://pytorch.org/get-started/pytorch-2.0/) 中找到更多详细信息和基准测试。

`torch.compile` 有一个不断增长的后端列表，可以通过调用 `torchdynamo.list_backends()` 找到，每个后端都有其可选依赖项。

通过在 [`TrainingArguments`] 中使用 `torch_compile_backend` 来选择要使用的后端。一些常用的后端包括：

**调试后端**：
* `dynamo.optimize("eager")` - 使用 PyTorch 运行提取的 GraphModule。这在调试 TorchDynamo 问题时非常有用。
* `dynamo.optimize("aot_eager")` - 使用 AotAutograd 并不编译，即只是使用 PyTorch eager 用于 AotAutograd 提取的前向和后向图。这对调试很有用，但不太可能带来加速。

**训练和推理后端**：
* `dynamo.optimize("inductor")` - 使用 TorchInductor 后端，利用 codegened Triton kernels 的 AotAutograd 和 cudagraphs 进行优化 [阅读更多](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
* `dynamo.optimize("nvfuser")` -  使用 TorchScript 的 nvFuser。[阅读更多](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_nvfuser")` -  使用 AotAutograd 的 nvFuser。[阅读更多](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_cudagraphs")` - 使用 AotAutograd 的 cudagraphs。[阅读更多](https://github.com/pytorch/torchdynamo/pull/757)

**仅推理后端**：
* `dynamo.optimize("ofi")` - 使用 Torchscript 的 optimize_for_inference。 [阅读更多](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
* `dynamo.optimize("fx2trt")` - 使用 NVIDIA TensorRT 进行推理优化。 [阅读更多](https://pytorch.org/TensorRT/tutorials/getting_started_with_fx_path.html)
* `dynamo.optimize("onnxrt")` - 使用 ONNXRT 在 CPU/GPU 上进行推理。 [阅读更多](https://onnxruntime.ai/)
* `dynamo.optimize("ipex")` - 使用 IPEX 在 CPU 上进行推理。 [阅读更多](https://github.com/intel/intel-extension-for-pytorch)

关于如何在 🤗 Transformers 中使用 `torch.compile` 的示例，请查看[博客文章，使用最新的 PyTorch 2.0 特性对 BERT 模型进行文本分类微调](https://www.philschmid.de/getting-started-pytorch-2-0-transformers)


## 使用 🤗 Accelerate

通过 [🤗 Accelerate](https://huggingface.co/docs/accelerate/index)，您可以使用上述方法，并完全掌控训练循环，实质上可以使用纯 PyTorch 编写循环，只需进行少量修改。

假设您已经将上述方法结合到 [`TrainingArguments`] 中，如下所示：

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)
```

🤗 Accelerate 的完整示例训练循环仅有少量代码行：

```py
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

accelerator = Accelerator(fp16=training_args.fp16)
model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

model.train()
for step, batch in enumerate(dataloader, start=1):
    loss = model(**batch).loss
    loss = loss / training_args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % training_args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

首先，我们将数据集包装在 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 中。然后，我们可以通过调用模型的 [`~PreTrainedModel.gradient_checkpointing_enable`] 方法启用梯度checkpointing。在初始化 [`Accelerator`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator) 时，可以指定是否要使用混合精度训练，它会在 [`prepare`] 调用中为我们处理。在 [`prepare`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.prepare) 调用期间，如果使用多个 GPU，数据加载器也会分布到多个工作进程中。我们使用先前示例中的相同 [8 位优化器](#8-bit-adam)。

最后，我们可以添加主要的训练循环。请注意，`backward` 调用由 🤗 Accelerate 处理。我们还可以看到梯度累积的工作原理：归一化损失，以便在累积结束时得到平均值，一旦步数足够，运行优化。

使用 🤗 Accelerate 实现这些优化技术只需要少量代码行，带来更多的训练循环灵活性。要了解所有功能的完整文档，请查看 [Accelerate 文档](https://huggingface.co/docs/accelerate/index)。

## 高效的软件预构建

PyTorch的[pip和conda版本](https://pytorch.org/get-started/locally/#start-locally)是使用cuda工具包预先构建的，这足以运行PyTorch，但如果需要构建cuda扩展程序则不够。

有时，预先构建某些组件可能需要额外的工作。例如，如果您使用像`apex`这样的库，这些库并没有预先编译。在其他情况下，找出如何系统范围内安装正确的cuda工具包可能会很复杂。为了解决这些情况，PyTorch和NVIDIA发布了一个新版本的NGC Docker容器，该容器已经预先构建好一切。您只需要将你的程序安装在上面，它就能立刻运行。

如果您想调整PyTorch源代码和/或制作一个新的定制构建，这种方法也很有用。要找到您想要的docker映像版本，可以从[PyTorch发布说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)开始，选择最新的月度版本。进入所需版本的发布说明，检查环境组件是否满足您的需求（包括NVIDIA驱动要求），然后在该文档的顶部转到相应的NGC页面。如果您对此有所困惑，这里是[所有PyTorch NGC映像的索引](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)。

接下来，按照说明下载并部署docker映像。

## 混合专家模型

最近的一些论文报告称，将混合专家模型（MoE）技术集成到Transformer模型中可以实现4-5倍的训练加速和更快的推理速度。

由于发现参数越多，性能越好，这种技术可以在不增加训练成本的情况下将参数数量提高一个数量级。

在这种方法中，每隔一个前馈神经网络（FFN）层都被MoE层替代，MoE层由许多experts组成，其中包含一个门控函数，根据序列中输入标记的位置均衡地训练每个experts。

![MoE Transformer 2x block](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perf-moe-transformer.png)

(source: [GLAM](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html))

您可以在本节末尾列出的论文中找到详尽的细节和比较表格。

这种方法的主要缺点是它需要惊人数量的GPU内存 - 几乎比其密集等效模型多一个数量级。有各种蒸馏和方法被提出来以克服更高的内存需求。

不过，这其中存在直接的trade-off，您可以使用少量experts和2-3倍较小的基础模型，而不是几十或上百名experts。这会使模型小5倍，因此适当提高训练速度的同时也仅仅适度增加了内存需求。

大多数相关的论文和实现都是围绕Tensorflow/TPUs构建的。

- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [GLaM: Generalist Language Model (GLaM)](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)

针对 PyTorch，DeepSpeed 也构建了一个模型： [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596), [Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/) - 博客文章:  [1](https://www.microsoft.com/en-us/research/blog/deepspeed-powers-8x-larger-moe-model-training-with-high-performance/), [2](https://www.microsoft.com/en-us/research/publication/scalable-and-efficient-moe-training-for-multitask-multilingual-models/) 以及基于Transformer的大型自然语言生成模型的特定部署： [博客文章](https://www.deepspeed.ai/2021/12/09/deepspeed-moe-nlg.html), [Megatron-Deepspeed branch](https://github.com/microsoft/Megatron-DeepSpeed/tree/moe-training).

## 使用 PyTorch 原生 attention 和 Flash Attention

PyTorch 2.0发布了一个原生的[`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA)，允许使用融合的GPU核心，例如[高效使用内存的attention](https://arxiv.org/abs/2112.05682)和[flash attention](https://arxiv.org/abs/2205.14135)。

在安装[`optimum`](https://github.com/huggingface/optimum)包后，可以替换相关的内部模块以使用PyTorch的原生attention：

```python
model = model.to_bettertransformer()
```

一旦转换完成，请像往常一样训练模型。

<Tip warning={true}>

PyTorch原生的`scaled_dot_product_attention`操作符只能在没有提供`attention_mask`的情况下转换到Flash Attention。

默认情况下，在训练模式下，BetterTransformer集成**删除了对mask的支持，只能用于不需要填充mask的批处理训练**。例如，在掩码语言建模或因果语言建模期间。BetterTransformer不适用于需要填充mask的任务的微调模型。

</Tip>

检查这个[博客帖子](https://pytorch.org/blog/out-of-the-box-acceleration/)，了解更多关于使用SDPA进行加速和节省内存的信息。