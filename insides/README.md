<details>
<summary>Plans and hypothesis</summary>

# Ideas, that haven't tested yet
- Normalization sandwich
- Give information from all 6 mambas directly to transformer
- Use mamba-2 insted of mamba-1 blocks

# Tested ideas, that don't worth additional work now
- I need to add registers in model. New thought: Its very hard to implement in vmamba blocks registers, because they use 2D input, not 1D as a transformers or VIM
- Replace Softmax to Clipped Softmax as in this [paper](https://arxiv.org/abs/2306.12929). New thought: It makes model slower by 20%, not worth it
- Try to use Windows in some Mamba blocks to better manage diagonal information on images. New thought: Its decrease the model capability, overlap cross attention blocks from Contrast have already managed well with problem with diagonal information.
- Probably use Multi-Query Attention(MQA). New thought: basically don't worth it, because window attention is already very powerful

</details>

<details>
<summary>Thesis Findings on Hybrid Transformer and Mamba Architecture</summary>

1. **Pure Mamba Performance**:
   - **Finding**: Pure Mamba architecture performs poorly, as expected, due to its limited ability to accurately copy information. Its architecture relies on compressed and hidden representations, which only allow for approximate copying.
   - **Explanation**: The inherent design of Mamba, with its focus on compressed representations, restricts its precision in copying data, leading to suboptimal performance.

2. **Positional Embeddings in Hybrids**:
   - **Finding**: The hybrid architecture can perform equally well without positional embeddings (PE) as with them. Mamba blocks can learn to encode pixel position information for transformer blocks, but this requires time. Initially, models with PE perform better, but eventually, the performance equalizes, with a slight advantage for models without PE.
   - **Explanation**: Omitting PE accelerates the model and slightly enhances its accuracy. This finding suggests that Mamba blocks can compensate for the lack of positional information, though this learning process impacts early training phases.

3. **Channel Attention Block (CAB) Removal**:
   - **Finding**: Removing the Channel Attention Block improves performance. The parameters freed up by removing CAB can be reallocated to other parts of the model, resulting in higher accuracy.
   - **Explanation**: The reallocation of parameters previously used by CAB to other model components enhances overall performance, making the architecture more efficient.

4. **Attention Blocks Enhancing Mamba**:
   - **Finding**: Adding attention blocks after several Mamba layers (6 in this case) significantly boosts performance. The attention blocks mitigate the weaknesses of Mamba, resulting in a powerful combination that achieves high performance and accuracy.
   - **Explanation**: Attention blocks compensate for the limitations of Mamba, creating a hybrid that trains faster by approximately 25% on Set5 benchmarks and achieves similar PSNR. Further tests on other benchmarks and inference speed comparisons are planned, expecting an increased performance gap in inference.

</details>

Baseline is a HAT with (S)W-MSA replaced by SS2D blocks from VMambaV2 

Its the number of parameters of this models:
| Model | Parameters |
|----------|----------|
| baseline | 745,572 |
| baseline_no_pe | 718,194 |
| baseline_no_pe_no_cab | 764,202 |
| mamba(pure) | 759,108 |
| hat_light | 771,099 |


# full plot
<p align="center">
  <img src="../images/baseline_urban100.png" width="75%">
</p>
