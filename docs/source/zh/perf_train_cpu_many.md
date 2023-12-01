<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 多 CPU 上的高效训练

当在单个 CPU 上进行训练速度太慢时，我们可以利用多个 CPU。这个指南侧重于基于 PyTorch 的 DDP，实现分布式 CPU 训练的高效性。

## Intel® oneCCL 与 PyTorch 的绑定

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL)（集合通信库）是用于实现诸如 allreduce、allgather、alltoall 等集合操作的高效分布式深度学习训练的库。欲了解更多关于 oneCCL 的信息，请参阅 [oneCCL 文档](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html) 和 [oneCCL 规范](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html)。

模块 `oneccl_bindings_for_pytorch`（在版本 1.12 之前为 `torch_ccl`）实现了 PyTorch C10D ProcessGroup API，并可以作为外部 ProcessGroup 动态加载，目前仅在 Linux 平台上可用。

请查看 [oneccl_bind_pt](https://github.com/intel/torch-ccl) 获取更详细的信息。


### Intel® oneCCL Bindings for PyTorch 的安装方式：

Wheel文件支持以下版本python：

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :---------: |
| 1.13.0            |            | √          | √          | √          | √           |
| 1.12.100          |            | √          | √          | √          | √           |
| 1.12.0            |            | √          | √          | √          | √           |
| 1.11.0            |            | √          | √          | √          | √           |
| 1.10.0            | √          | √          | √          | √          |             |

```
pip install oneccl_bind_pt=={pytorch_version} -f https://developer.intel.com/ipex-whl-stable-cpu
```
`{pytorch_version}` 应该是您的 PyTorch 版本，例如 1.13.0。
查看 [oneccl_bind_pt 安装](https://github.com/intel/torch-ccl) 获取更多方法。oneCCL 和 PyTorch 的版本必须匹配。

<Tip warning={true}>

`oneccl_bindings_for_pytorch` 1.12.0 预构建的 wheel 不适用于 PyTorch 1.12.1（适用于 PyTorch 1.12.0）。
PyTorch 1.12.1 应当能够使用 `oneccl_bindings_for_pytorch` 1.12.100。

</Tip>

## Intel® MPI 库
使用这个基于标准的 MPI 实现，在 Intel® 架构上实现灵活、高效、可扩展的集群消息传递。这个组件是 Intel® oneAPI HPC Toolkit 的一部分。

`oneccl_bindings_for_pytorch` 会与 MPI 工具集一起安装。在使用之前需要配置环境。

对于Intel® oneCCL >= 1.12.0。

```
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
```

对于Intel® oneCCL whose version < 1.12.0
```
torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source $torch_ccl_path/env/setvars.sh
```

#### IPEX安装:

IPEX 为 CPU 训练提供了针对 Float32 和 BFloat16 的性能优化，您可以参考 [单 CPU 章节](./perf_train_cpu)。

以下的 "在训练器中的使用" 以 Intel® MPI 库中的 mpirun 为例。


## 在训练器中的使用
在训练器中启用 ccl 后端的多 CPU 分布式训练，用户应在命令参数中添加 **`--ddp_backend ccl`**。

让我们以 [问答示例](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) 为例：

以下命令在一个 Xeon 节点上使用 2 个进程进行训练，每个进程在一个处理器插槽上运行。可以调整OMP_NUM_THREADS/CCL_WORKER_COUNT 变量以获得最佳性能。
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=127.0.0.1
 mpirun -n 2 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex
```
以下命令在两个 Xeon（node0 和 node1，以 node0 为主进程）上使用总共四个进程进行训练，ppn（每个节点的进程数）设置为 2，每个处理器插槽上运行一个进程。可以调整OMP_NUM_THREADS/CCL_WORKER_COUNT 变量以获得最佳性能。

在 node0，您需要创建一个配置文件，其中包含每个节点的 IP 地址（例如 hostfile），并将该配置文件的路径作为参数传递。
```shell script
 cat hostfile
 xxx.xxx.xxx.xxx #node0 ip
 xxx.xxx.xxx.xxx #node1 ip
```
现在，在 node0 执行以下命令，将在 node0 和 node1 上启用 4DDP，并使用 BF16 自动混合精度：
```shell script
 export CCL_WORKER_COUNT=1
 export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
 mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
 python3 run_qa.py \
 --model_name_or_path bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12  \
 --learning_rate 3e-5  \
 --num_train_epochs 2  \
 --max_seq_length 384 \
 --doc_stride 128  \
 --output_dir /tmp/debug_squad/ \
 --no_cuda \
 --ddp_backend ccl \
 --use_ipex \
 --bf16
```
