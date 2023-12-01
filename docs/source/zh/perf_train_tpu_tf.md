<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 使用 TensorFlow 在 TPU 上进行训练

<Tip>

如果您不需要长篇解释，只是想要 TPU 的代码示例来开始学习，可以查看[我们的 TPU 示例notebook！](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)

</Tip>

### 什么是TPU？

TPU 指的是  **张量处理单元**。它们是由谷歌设计的硬件，用于极大地加速神经网络中的张量计算，类似于 GPU。它们可用于网络训练和推理。TPU通常可以通过谷歌的云服务访问，但少量TPU 也可以通过 Google Colab 和 Kaggle Kernels 直接免费访问。

因为[🤗 Transformers 中的所有 TensorFlow 模型都是 Keras 模型](https://huggingface.co/blog/tensorflow-philosophy)，因此本文档中的大部分方法通常适用于任何 Keras 模型的 TPU 训练！然而，在 HuggingFace 生态系统（hug-o-system?）中的 Transformers 和 Datasets 方面，也有一些特定点，我们在接触到这些点时会特别指出。

### 有哪些可用的TPU？

新用户通常对TPU的种类和不同的访问方式感到困惑。首先需要理解的关键区别是**TPU节点**和**TPU虚拟机**之间的区别。

当您使用一个**TPU节点**时，实际上是间接地访问了一个远程的TPU。您需要一个单独的虚拟机，它将初始化你的网络和数据管道，然后将它们转发到远程节点。当您在Google Colab上使用TPU时，你正在以**TPU节点（TPU Nodes）**的方式访问它。

对于不习惯使用TPU节点的人来说，使用TPU节点可能会产生一些意外的情况！特别是因为TPU位于与您运行Python代码的机器物理上不同的系统上，您的数据不能在本地存储 - 任何从您机器的内部存储器加载数据的pipeline都会完全失败！相反，数据必须存储在Google Cloud Storage中，这样您的数据管道仍然可以访问它，即使pipeline正在远程的TPU节点上运行。

<Tip>

如果您可以将所有数据都存储在内存中的`np.ndarray`或`tf.Tensor`中，那么即使在使用Colab或TPU节点时，您也可以在不将其上传到Google Cloud Storage的情况下对其进行`fit()`操作。

</Tip>

<Tip>

**🤗特定的Hugging Face技巧🤗：** 在我们的TF代码示例中，您会看到`Dataset.to_tf_dataset()`方法和其更高级别的包装器`model.prepare_tf_dataset()`。这两个方法在TPU节点上都会失败。原因是尽管它们创建了一个`tf.data.Dataset`，但它不是一个“纯粹”的`tf.data`管道，而是使用`tf.numpy_function`或`Dataset.from_generator()`从底层的HuggingFace `Dataset`中流式传输数据。这个HuggingFace `Dataset`是由本地磁盘上的数据支持的，而远程TPU节点无法读取这些数据。

</Tip>

第二种访问TPU的方式是通过**TPU虚拟机**。在使用TPU虚拟机时，您可以直接连接到TPU所在的计算机，就像在GPU虚拟机上训练一样。TPU虚拟机通常更容易使用，特别是在处理数据管道方面。以上所有警告都不适用于TPU虚拟机！

这是一个有观点的文件，所以我们的观点是：**如果可能的话，尽量避免使用TPU节点。**它比TPU虚拟机更令人困惑，更难调试。它也可能在将来不受支持 - Google的最新TPU，TPUv4，只能作为TPU虚拟机访问，这表明TPU节点将逐渐成为一种“遗留且不更新”的访问方法。然而，我们理解唯一的免费TPU访问是在Colab和Kaggle内核中使用TPU节点，因此我们将尝试解释如何在您必须使用时处理它！请查看[TPU示例笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)以获取更多关于此的代码示例。

### 有哪些大小的TPU可用？

单个TPU（v2-8/v3-8/v4-8）运行8个副本。TPU存在于可以同时运行数百或数千个副本的**pods**中。当您使用超过一个单独的TPU但少于整个pod时（例如，v3-32），您的TPU群集被称为**pod切片**。

当您通过Colab访问免费的TPU时，通常可以获得一个单独的v2-8 TPU。

### 我不断听到关于XLA的事情。什么是XLA，它与TPU有什么关系？

XLA是一种优化编译器，被TensorFlow和JAX使用。在JAX中，它是唯一的编译器，而在TensorFlow中是可选的（但在TPU上是必须的）。训练Keras模型时启用它的最简单方法是将参数`jit_compile=True`传递给`model.compile()`。如果您没有收到任何错误并且性能良好，那么这是一个伟大的迹象，表明您已经准备好转移到TPU！

在TPU上调试通常比在CPU/GPU上更难，因此我们建议首先在具有XLA的CPU/GPU上运行您的代码，然后再尝试在TPU上运行。当然，您不必长时间训练 - 只需进行几个步骤以确保您的模型和数据管道按照预期工作即可。

<Tip>

XLA编译的代码通常更快 - 因此，即使您不打算在TPU上运行，添加`jit_compile=True`也可以提高您的性能。但是，请务必注意下面有关XLA兼容性的注意事项！

</Tip>

<Tip warning={true}>

**来自痛苦经验的提示：** 虽然使用`jit_compile=True`是提高速度和测试CPU/GPU代码是否与XLA兼容的好方法，但实际上如果在TPU上实际训练时将其保留，它可能会引起很多问题。在TPU上，XLA编译将隐式发生，因此请记住在实际在TPU上运行代码之前删除该行！

</Tip>

###如何使我的模型与XLA兼容？

在许多情况下，您的代码可能已经与XLA兼容！但是，有一些在通用的TensorFlow中工作的东西在XLA中不起作用。我们将它们提炼成以下三个核心规则：

<Tip>

**🤗特定的HuggingFace技巧🤗：**   我们投入了很多精力来重写我们的TensorFlow模型和损失函数，使其与XLA兼容。我们的模型和损失函数默认遵守规则#1和#2，因此如果您使用`transformers`模型，可以跳过它们。但是，在编写自己的模型和损失函数时，不要忘记这些规则！

</Tip>

#### XLA 规则#1: 您的代码不能有“数据相关条件语句”

这意味着任何`if`语句都不能依赖于`tf.Tensor`内部的值。例如，这段代码块不能使用XLA编译！

```python
if tf.reduce_sum(tensor) > 10:
    tensor = tensor / 2.0
```

这在一开始可能看起来非常受限，但大多数神经网络代码并不需要这样做。您通常可以通过使用`tf.cond`（参见[此处](https://www.tensorflow.org/api_docs/python/tf/cond)的文档）或通过删除条件并找到一个巧妙的带有指示变量的数学技巧来绕过这个限制，如下所示：

```python
sum_over_10 = tf.cast(tf.reduce_sum(tensor) > 10, tf.float32)
tensor = tensor / (1.0 + sum_over_10)
```

这段代码与上面的代码具有完全相同的效果，但通过避免条件语句，我们确保它可以在XLA中编译而没有任何问题！

#### XLA 规则 #2:  您的代码不能有“依赖于数据的形状”"

这意味着您的代码中的所有`tf.Tensor`对象的形状都不能依赖于它们的值。例如，函数`tf.unique`不能与XLA一起编译，因为它返回一个包含输入中每个唯一值的一个实例的`tensor`。输出的形状显然会根据输入`Tensor`的重复程度而有所不同，因此XLA拒绝处理它！

总的来说，大多数神经网络代码默认遵循规则#2。然而，有几种常见的情况会导致问题。一个非常常见的例子是使用**标签掩码**，将标签设置为负值以指示在计算损失时应忽略这些位置。如果您查看支持标签掩码的NumPy或PyTorch损失函数，您通常会看到像这样使用[布尔索引](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing)的代码：

```python
label_mask = labels >= 0
masked_outputs = outputs[label_mask]
masked_labels = labels[label_mask]
loss = compute_loss(masked_outputs, masked_labels)
mean_loss = torch.mean(loss)
```

这段代码在 NumPy 或 PyTorch 中完全没问题，但在 XLA 中会出问题！为什么？因为 `masked_outputs` 和 `masked_labels` 的形状取决于屏蔽了多少位置 - 这造成了**数据相关的形状**。然而，就像规则 #1 一样，我们通常可以重写这段代码，得到完全相同的输出，而不需要任何数据相关的形状。

```python
label_mask = tf.cast(labels >= 0, tf.float32)
loss = compute_loss(outputs, labels)
loss = loss * label_mask  # Set negative label positions to 0
mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(label_mask)
```

在这里，我们通过针对每个位置计算损失，但在计算均值时将被屏蔽的位置在分子和分母中清零，从而避免了数据相关的形状，这与第一个块产生了完全相同的结果，并保持了与 XLA 的兼容性。请注意，我们使用了与规则 #1 相同的技巧 - 将 `tf.bool` 转换为 `tf.float32` 并将其用作指示变量。这是一个非常有用的技巧，所以如果您需要将自己的代码转换为 XLA，请记住它！

#### XLA 规则 #3: XLA 需要为其看到的每个不同输入形状重新编译您的模型

这是一个大问题。这意味着如果您的输入形状非常多样化，XLA 将不断重新编译您的模型，这将带来巨大的性能问题。这种情况常见于 NLP 模型，其中经过标记化后，输入文本长度是变化的。在其他模态中，静态形状更为常见，因此这个规则就不是那么大的问题。

如何绕过规则 #3？关键是**填充** - 如果您将所有输入填充到相同的长度，然后使用一个 `attention_mask`，您可以获得与可变形状相同的结果，但不会出现任何 XLA 问题。然而，过度填充也会导致严重的减速 - 如果您将所有样本都填充到整个数据集中的最大长度，您可能会得到由无尽填充标记组成的批次，这将浪费大量计算资源和内存！

对于这个问题并没有完美的解决方案。但您可以尝试一些技巧。其中一个非常有用的技巧是**将批量样本填充到 32 或 64 个标记的倍数**。这通常只会略微增加标记数量，但会大幅减少唯一输入形状的数量，因为现在每个输入形状必须是 32 或 64 的倍数。较少的唯一输入形状意味着更少的 XLA 编译！


<Tip>

**🤗 HuggingFace 的专属提示 🤗：** 我们的标记器和数据集合器有一些方法可以帮助您。当调用标记器时，您可以使用 `padding="max_length"` 或 `padding="longest"` 来让它们输出填充的数据。我们的标记器和数据集合器还有一个 `pad_to_multiple_of` 参数，可以帮助您减少看到的唯一输入形状的数量！

</Tip>

### 我如何在TPU上训练模型

当您的训练兼容 XLA并且您的数据集已经被适当地准备好（如果您正在使用 TPU 节点/Colab），那么在 TPU 上运行将会非常简单！您实际上需要在您的代码中做的改变就是添加几行来初始化你的 TPU，并确保您的模型和数据集都被创建在 `TPUStrategy` 范围内。查看 [我们的 TPU 示例notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb) 以看到这个过程！

### 总结

让我们用一个简单的清单来总结一下，当您想要准备好您的模型进行 TPU 训练时可以遵循以下步骤：

- 确保您的代码遵循 XLA 的三个规则
- 在 CPU/GPU 上使用 `jit_compile=True` 编译您的模型，并确认您可以使用 XLA 进行训练
- 将数据集加载到内存中，或者使用兼容 TPU 的数据集加载方法（参见 [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)）
- 将您的代码迁移到 Colab（加速器设置为“TPU”）或 Google Cloud 上的 TPU VM
- 添加 TPU 初始化代码（参见 [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)）
- 创建您的 `TPUStrategy` 并确保数据集加载和模型创建在 `strategy.scope()` 内（参见 [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)）
- 当您切换到 TPU 时，不要忘记再次将 `jit_compile=True` 移除！
- 🙏🙏🙏🥺🥺🥺
- 调用 `model.fit()`
- 您成功了！