<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU推理

通过一些优化技术，可以在 CPU 上高效地运行大型模型推理。其中一种优化技术涉及将 PyTorch 代码编译成高性能环境（如 C++）的中间格式。另一种技术是将多个操作融合为一个内核，以减少单独运行每个操作的开销。

您可以学习如何使用 [BetterTransformer](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/) 进行更快的推理，并了解如何将 PyTorch 代码转换为 [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)。如果您正在使用英特尔 CPU，您还可以使用 [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/index.html) 中的 [图优化](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features.html#graph-optimization) 来进一步提高推理速度。最后，学习如何使用 🤗 Optimum 通过 ONNX Runtime 或 OpenVINO（如果您正在使用英特尔 CPU）来加速推理。

## BetterTransformer

BetterTransformer 通过其快速路径（Transformer 函数的原生 PyTorch 专用实现）加速推理。快速路径执行的两个优化是：

1. 融合（fusion），将多个顺序操作合并为单个“核心”，以减少计算步骤的数量
2. 跳过`padding tokens`的固有稀疏性，以避免与`nested tensors`的不必要计算

BetterTransformer 还将所有attention操作转换为更节省内存的 [scaled dot product attention](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention)。

<Tip>

BetterTransformer 并非适用于所有模型。请检查此 [列表](https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models)，查看模型是否支持 BetterTransformer。


</Tip>

在开始之前，请确保已安装 🤗 Optimum [installed](https://huggingface.co/docs/optimum/installation)。

使用 [`PreTrainedModel.to_bettertransformer`] 方法启用 BetterTransformer：

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder")
model.to_bettertransformer()
```

## TorchScript

TorchScript 是 PyTorch 的中间模型表示形式，可在对性能要求很高的生产环境中运行。您可以在 PyTorch 中训练模型，然后将其导出到 TorchScript 以摆脱模型在 Python 性能上的限制。PyTorch [traces](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) 模型以返回一个 [`ScriptFunction`]，它通过即时编译（JIT）进行优化。与默认的即时模式相比，PyTorch 中的 JIT 模式通常通过操作融合等优化技术获得更好的推断性能。

要了解 TorchScript 的基础知识，请参阅 [Introduction to PyTorch TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) 教程。

使用 [`Trainer`] 类，您可以通过设置 `--jit_mode_eval` 标志为 CPU 推断启用 JIT 模式：

```bash
python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
--jit_mode_eval
```

<Tip warning={true}>

对于 PyTorch >= 1.14.0，JIT 模式对于预测和评估任何模型都可能有益，因为 `jit.trace` 支持字典输入。

对于 PyTorch < 1.14.0，如果模型的前向参数顺序与 `jit.trace` 中的元组输入顺序匹配，例如问答模型，那么 JIT 模式可能有益。如果模型的前向参数顺序与 `jit.trace` 中的元组输入顺序不匹配，例如文本分类模型，`jit.trace` 将失败，并且我们会捕获这个异常来进行回退。我们使用日志记录来通知用户。

</Tip>

## IPEX图优化

Intel® Extension for PyTorch（IPEX）为 Intel CPU 提供 JIT 模式下的进一步优化，并建议将其与 TorchScript 结合使用以获得更快的性能。IPEX 的图优化可以合并诸如多头注意力、Concat Linear、Linear + Add、Linear + Gelu、Add + LayerNorm 等操作。

要充分利用这些图优化，请确保已经[安装](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html)了 IPEX。

```bash
pip install intel_extension_for_pytorch
```

在 `Trainer` 类中设置 `--use_ipex` 和 `--jit_mode_eval` 标志，以启用带图优化的 JIT 模式：

```bash
python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
--use_ipex \
--jit_mode_eval
```

## 🤗 Optimum

<Tip>

了解如何在[Optimum Inference with ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models)指南中使用ORT和🤗 Optimum。这部分仅提供了简要的示例。

</Tip>

ONNX Runtime (ORT)是一个模型加速器，默认在 CPU 上运行推理。ORT 受到🤗 Optimum 的支持，可以在 🤗 Transformers 中使用，而无需对代码进行太多更改。您只需要用其等价的 [`~optimum.onnxruntime.ORTModel`] 替换🤗 Transformers的 `AutoClass`，并加载 ONNX 格式的检查点。

例如，如果您要在问答任务上进行推理，加载 [optimum/roberta-base-squad2](https://huggingface.co/optimum/roberta-base-squad2) checkpoint，其中包含一个 `model.onnx` 文件：

```py
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

model = ORTModelForQuestionAnswering.from_pretrained("optimum/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

onnx_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

question = "What's my name?"
context = "My name is Philipp and I live in Nuremberg."
pred = onnx_qa(question, context)
```

如果你使用英特尔 CPU，可以查看 🤗 [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)，支持各种压缩技术（量化、剪枝、知识蒸馏）和将模型转换为 [OpenVINO](https://huggingface.co/docs/optimum/intel/inference) 格式以提高推理性能的工具。
