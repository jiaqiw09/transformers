<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 在CPU上高效训练

这个指南侧重于在 CPU 上高效训练大模型。

## 使用 IPEX 进行混合精度处理

IPEX针对具有AVX-512或更高版本的CPU进行了优化，并且在仅具有AVX2功能的CPU上也能正常工作。因此，预计对于具有AVX-512或更高版本的Intel CPU，IPEX将带来性能优势，而仅具有AVX2功能的CPU（例如AMD CPU或较旧的Intel CPU）可能在IPEX下实现更好的性能，但并不能保证。 IPEX为在CPU上使用Float32和BFloat16进行训练提供了性能优化。 BFloat16的使用是以下部分的主要关注点。

低精度数据类型BFloat16已经在支持AVX512指令集的第3代Xeon® Scalable Processors（也称为Cooper Lake）上原生支持，并将在下一代支持Intel&#174; Advanced Matrix Extensions（Intel&#174; AMX）指令集的Intel&® Xeon&® Scalable Processors上进一步提供更高的性能。从PyTorch-1.10开始启用了用于CPU后端的自动混合精度。同时，在Intel® Extension for PyTorch中大规模启用了对CPU和BFloat16的自动混合精度的支持以及BFloat16操作符的优化，并部分迁移到了PyTorch主分支。用户可以通过IPEX自动混合精度获得更好的性能和用户体验。

请查看自动混合精度的详细信息。

### IPEX安装n:

IPEX 发布与 PyTorch 同步，可通过 pip 安装：

| PyTorch Version   | IPEX version   |
| :---------------: | :----------:   |
| 1.13              |  1.13.0+cpu    |
| 1.12              |  1.12.300+cpu  |
| 1.11              |  1.11.200+cpu  |
| 1.10              |  1.10.100+cpu  |

```
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```

查看更多有关 [IPEX 安装](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html) 的方法。

### 在训练器中的使用方法
在训练器中启用 IPEX 的自动混合精度，用户应该在训练命令参数中添加 `use_ipex`、`bf16` 和 `no_cuda`。

以 [Transformers 问答](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 为例：

- 在 CPU 上使用 IPEX 进行 BF16 自动混合精度训练：
<pre> python run_qa.py \
--model_name_or_path bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/debug_squad/ \
<b>--use_ipex \</b>
<b>--bf16 --no_cuda</b></pre> 

### 实践示例

博客: [Accelerating PyTorch Transformers with Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids)
