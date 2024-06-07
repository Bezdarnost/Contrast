Dear Zheng Chen,

Thank you for your response and for sharing your preference for written communication. My English skills are quite similar, so I completely understand. I am open to meeting in the future if you prefer. Additionally, I can learn Mandarin in my free time if it would help us to communicate.

I would like to share my vision for our collaboration:

In my previous work, I encountered a few significant challenges: a lack of time to implement all my ideas and insufficient computational resources.

I was hoping to contribute to your next research paper. I noticed that you frequently collaborate with Yulun Zhang and Jinjin Gu. I would be honored to become a co-author, assisting with visualizations, LaTeX writing, and implementing your ideas in Python (PyTorch) or Triton. Essentially, I could serve as a research engineer and LaTeX writer.

Alternatively, I need strong papers as a first author for my future PhD application. If this aligns with your goals, we could focus on developing my ideas, with me handling the majority of the work. Here are a few of my ideas:

In my previous work, I began developing a "Contrast" model (Convolutional Transformer State Space Model), but due to time and resource constraints, I consider it version 0. For version 1, I would like to implement the following

1) Integrate Mamba blocks from VmambaV3 and Transformer blocks with Flash Attention v2. The Xiaomi team demonstrated the potential of Mamba blocks in super-resolution tasks, alongside effective ensembling techniques.

2) Replace softmax attention with clipped softmax from this work: https://arxiv.org/abs/2306.12929. This adjustment addresses the issue of activation outliers in attention mechanisms, which can exceed machine precision. Sometimes the model attempts to produce a zero output from softmax in certain patches, necessitating extremely large (in absolute value) values in others.

3) Rewrite the convolution from PyTorch to Triton, enhancing its performance on CUDA.

4) Address the issue of image edges, as convolutions tend to introduce synthetic information at the edges, which can cause artifacts. My idea is to add an extra neuron to each filter specifically for edge correction(It will not go all the way around the image, but specifically around the edges). For instance, in the case of a 3x3 convolution with 1 input channel and 1 output channel, instead of having 9 weights and 1 bias per kernel, we would have an additional weight for edge correction. This approach should not significantly impact performance but could theoretically solve the edge problem and speed up training. In my tests, models that did not trim the edges in the output (in the training process) took slightly longer to train as they attempted to correct the edges themselves.

5) I am not convinced that the current structure of transformers for computer vision tasks is optimal. Vision transformers were largely adapted from NLP, where local information is less significant. However, in our field, local information is crucial. I propose modifying the structure from:
-> LN -> Attention -+> LN -> MLP -+>
to:
-> LN -> Attention -+> LN -> Conv -+>

6) This is highly experimental, but we could consider using this: https://github.com/HazyResearch/ThunderKittens. It might be even faster than the classical FA2

I am excited to hear your thoughts on these ideas and discuss how we can move forward with our collaboration.

Best regards, Aman Urumbekov
