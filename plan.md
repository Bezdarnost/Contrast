Whats insights I have:
- HATs window attention layers have a limited recieptive field. Mamba blocks also have limitations with diagonal information. In hybrid form they can help each other.
- Removing the normalization layer at the end helps a lot to stabilize numbers at the last blocks. I feel, that its becoming harder to train wider model.
- Its better to remove Channel Attention Block(CAB) and instead add more layers and make model wider to compensate the number of parameters.
- Its better not to delete Position Embeddings(PE) from Overlap Cross Attention Block(OCAB). With PE model "see" wider and reconstruct more details, but its slightly weaker in metrics than the model without PE. I think its because with PE light model couldn't reconstuct some colors. But I think this problem will disappear with bigger size model.

What I need to check:
- Possibly, its worth to remove LeakyReLU from the end, because previously model needs it after the normalization layer, but now we don't have it.
- Possibly, its worth to add PE into the mamba blocks. Theoretically, it will help to get information about a new rows and columns in image.

- We can include into the FFN a DWConv to help it to manage a local information(its better to use reflect mode, like "reflect", because with zeros its create some artifacts at the borders).
- Possibly we need to add the GEGLU instead of FFN to make model better (model training for test it).
