<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GPU推理

GPU 一直是机器学习的标准硬件选择，不同于 CPU，它们优化了内存带宽和并行性。为了跟上现代模型的更大尺寸或在现有和老旧硬件上运行这些大型模型，有几种优化方法可以加速 GPU 推理。在本指南中，您将学习如何使用 FlashAttention-2（更节省内存的注意力机制）、BetterTransformer（PyTorch 本地快速路径执行）和 bitsandbytes 将您的模型量化到更低的精度。最后，学习如何使用 🤗 Optimum 通过 ONNX Runtime 在 Nvidia GPU 上加速推理。

<Tip>

这里描述的大多数优化方法同样适用于多 GPU 设置！

</Tip>

## FlashAttention-2

<Tip>

FlashAttention-2 是实验性的，未来版本可能会发生相当大的更改。

</Tip>

FlashAttention-2 是标准注意力机制的更快、更高效的实现，可以通过以下方式显著加速推理过程：

1. 在序列长度上进行额外的注意力计算并行化
2. 在 GPU 线程之间分配工作，以减少它们之间的通信和共享内存读/写

FlashAttention-2 支持 Llama、Mistral 和 Falcon 模型的推理。您可以通过在 GitHub 上打开 Issue 或 Pull Request 来请求添加对另一个模型的 FlashAttention-2 支持。

在开始之前，请确保已安装 FlashAttention-2（有关更多先决条件的详细信息，请参阅[安装指南](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)）。


```bash
pip install flash-attn --no-build-isolation
```

要启用 FlashAttention-2，请向 [`~AutoModelForCausalLM.from_pretrained`] 添加 `use_flash_attention_2` 参数：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    use_flash_attention_2=True,
)
```

<Tip>

FlashAttention-2 只能在模型的 dtype 为 `fp16` 或 `bf16` 时使用，并且仅可在 Nvidia GPU 上运行。在使用 FlashAttention-2 之前，请确保将您的模型转换为相应的 dtype 并加载到支持的设备上。

</Tip>

FlashAttention-2 可以与其他优化技术（例如量化）结合，以进一步加速推断。例如，您可以将 FlashAttention-2 与 8 位或 4 位量化结合使用：

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load in 8bit
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    load_in_8bit=True,
    use_flash_attention_2=True,
)

# load in 4bit
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    load_in_4bit=True,
    use_flash_attention_2=True,
)
```

### 预期加速

你可以从推理速度的显著提升中受益，尤其是在输入具有长序列的情况下。然而，由于FlashAttention-2不支持使用`padding tokens`计算注意力权重，因此在序列包含`padding tokens`时，必须手动对批量推理的注意力权重进行填充/解填充。这会导致使用`padding tokens`进行批量生成的速度明显变慢。

为了克服这个问题，在训练过程中应该避免在序列中使用`padding tokens`（通过打包数据集或[连接序列](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py#L516)直到达到最大序列长度）。

对于在[tiiuae/falcon-7b](https://hf.co/tiiuae/falcon-7b)上进行的单个前向传递，序列长度为4096且不使用`padding tokens`的各种批量大小下，预期的加速效果是：

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/falcon-7b-inference-large-seqlen.png">
</div>

对于在 [meta-llama/Llama-7b-hf](https://hf.co/meta-llama/Llama-7b-hf) 模型上的单次前向传递，带有长度为 4096 的序列和各种不带`padding tokens`的批量大小，预期的加速效果为：

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-7b-inference-large-seqlen.png">
</div>

对于包含`padding tokens`的序列（使用`padding tokens`进行生成），您需要取消/填充输入序列，以正确计算注意力权重。对于相对较小的序列长度，单次前向传递会产生一些额外开销，导致轻微的加速效果（在下面的示例中，输入的 30% 填充有`padding tokens`）。

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-small-seqlen-padding.png">
</div>

但对于更大的序列长度，您可以期待更多的加速优势：

<Tip>

FlashAttention更节省内存，这意味着您可以在更大的序列长度上进行训练，而无需担心内存不足的问题。对于更大的序列长度，您可能将内存使用降低多达20倍。查看[flash-attention](https://github.com/Dao-AILab/flash-attention)仓库获取更多详情。

</Tip>

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-large-seqlen-padding.png">
</div>

## BetterTransformer

<Tip>

请查阅我们在[PyTorch 2.0中🤗解码器模型的开箱即用加速和内存节省](https://pytorch.org/blog/out-of-the-box-acceleration/)中的BetterTransformer和scaled dot product attention的基准测试，以及在[BetterTransformer](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2)博文中了解更多有关快速路径执行的信息。

</Tip>

BetterTransformer加速推理过程，采用快速路径执行（Transformer函数的原生PyTorch专用实现）。快速路径执行中的两项优化措施包括：

1. 融合：将多个顺序操作合并成单个“内核”，减少计算步骤次数。
2. 跳过内在的`padding tokens`稀疏性，以避免和`nested tensors`中进行不必要的计算。

BetterTransformer还将所有注意力操作转换为更节约内存的[scaled dot product attention（SDPA）](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention)，并在内部调用优化的内核，例如[FlashAttention](https://huggingface.co/papers/2205.14135)。

在开始之前，请确保已[安装🤗 Optimum](https://huggingface.co/docs/optimum/installation)。

然后，您可以使用 [`PreTrainedModel.to_bettertransformer`] 方法启用BetterTransformer：

```python
model = model.to_bettertransformer()
```

您可以使用 [`~PreTrainedModel.reverse_bettertransformer`] 方法还原为原始的Transformers模型。在保存模型以使用传统的Transformers建模之前，应该使用这种方法：

```py
model = model.reverse_bettertransformer()
model.save_pretrained("saved_model")
```

### FlashAttention

SDPA底层也可以调用FlashAttention内核。FlashAttention仅适用于使用 `fp16` 或 `bf16` dtype 的模型，因此在使用之前，请确保将模型转换为适当的dtype。

要启用FlashAttention，或者检查它在给定环境（硬件、问题规模）中是否可用，请使用[`torch.backends.cuda.sdp_kernel`](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel) 作为上下文管理器：

```diff
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16).to("cuda")
# convert the model to BetterTransformer
model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

+ with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

如果您遇到以下的bug并且错误追踪信息，请尝试使用PyTorch的nightly版本，它可能对FlashAttention有更广泛的覆盖。

```bash
RuntimeError: No available kernel. Aborting execution.

# install PyTorch nightly
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

## bitsandbytes

bitsandbytes 是一个量化库，支持 4 位和 8 位的量化。量化相对于完整精度版本，可以减小模型大小，使大模型更容易适配于内存有限的 GPU。

确保您已安装 bitsnbytes 和 🤗 Accelerate。

```bash
# these versions support 8-bit and 4-bit
pip install bitsandbytes>=0.39.0 accelerate>=0.20.0

# install Transformers
pip install transformers
```

### 4-bit

要加载一个 4 位的模型进行推理，使用 `load_in_4bit` 参数。`device_map` 参数是可选的，但我们建议将其设置为 `"auto"`，以允许 🤗 Accelerate 根据环境中的可用资源自动高效地分配模型。

```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
```

加载一个 4 位模型以在多个 GPU 上进行推理，您可以控制为每个 GPU 分配多少内存。例如，要为第一个 GPU 分配 600MB 的内存，为第二个 GPU 分配 1GB 的内存：

```py
max_memory_mapping = {0: "600MB", 1: "1GB"}
model_name = "bigscience/bloom-3b"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_4bit=True, max_memory=max_memory_mapping
)
```

### 8-bit

<Tip>

如果您对 8 位量化的概念感兴趣，可以阅读[使用Hugging Face Transformers、Accelerate和bitsandbytes进行大规模transformer的8位矩阵乘法的简介](https://huggingface.co/blog/hf-bitsandbytes-integration)的博客文章。

</Tip>

要加载一个 8 位推理模型，请使用 `load_in_8bit` 参数。`device_map` 参数是可选的，但我们建议将其设置为 `"auto"`，以便 🤗 Accelerate 根据环境中可用的资源自动且高效地分配模型：

```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
```

如果要使用 8 位模型进行文本生成，应使用 [`~transformers.GenerationMixin.generate`] 方法，而不是 [`Pipeline`] 函数，因为后者对 8 位模型未进行优化，速度会慢一些。某些采样策略，比如 nucleus 采样，对 8 位模型也不受 [`Pipeline`] 支持。同时，输入应放在与模型相同的设备上：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

加载多个 GPU 中的 4 位推理模型时，可以控制要分配给每个 GPU 的显存量。例如，分配 1GB 显存给第一个 GPU，分配 2GB 显存给第二个 GPU：

```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping
)
```

<Tip>

请随意在 Google Colab 的免费 GPU 上尝试运行一个拥有 110 亿参数的 [T5 模型](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing) 或者 30 亿参数的 [BLOOM 模型](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing) 进行推理！

</Tip>

## 🤗 Optimum

<Tip>

学习有关在 [NVIDIA GPU 上的加速推理](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#accelerated-inference-on-nvidia-gpus) 的有关使用 ORT 与 🤗 Optimum 的详细信息。此部分仅提供简单的例子。

</Tip>

ONNX Runtime（ORT）是一个模型加速器，支持在 Nvidia GPU 上进行加速推理。ORT使用优化技术，例如将常见操作融合为单个节点和常数折叠，以减少执行的计算数量，从而加快推理速度。此外，ORT会将计算量最大的操作放在 GPU 上，将其他操作放在 CPU 上，智能地在两个设备之间分配工作负载。

ORT受到 🤗 Optimum 的支持，可以在 🤗 Transformers 中使用。您需要使用 [`~optimum.onnxruntime.ORTModel`] 处理您的任务，并指定 `provider` 参数，可以设置为 [`CUDAExecutionProvider`](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#cudaexecutionprovider) 或 [`TensorrtExecutionProvider`](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider)。如果要加载尚未导出为 ONNX 格式的模型，可以将 `export=True` 设置为将模型即时转换为 ONNX 格式：


```py
from optimum.onnxruntime import ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
  "distilbert-base-uncased-finetuned-sst-2-english",
  export=True,
  provider="CUDAExecutionProvider",
)
```

现在您可以使用该模型进行推理：

```py
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

pipeline = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer, device="cuda:0")
result = pipeline("Both the music and visual were astounding, not to mention the actors performance.")
```

## 组合优化

通常可以将上述几种优化技术组合起来，以获得模型的最佳推理性能。例如，您可以以4位加载模型，然后使用FlashAttention启用BetterTransformer :

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# load model in 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=quantization_config)

# enable BetterTransformer
model = model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# enable FlashAttention
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
