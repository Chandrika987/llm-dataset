import numpy as np
from transformer_block import TransformerBlock

np.random.seed(42)

batch_size = 1
seq_len = 5
embed_dim = 16

x = np.random.randn(batch_size, seq_len, embed_dim)

print("Input Embeddings:\n", x)

model = TransformerBlock(
    embed_dim=embed_dim,
    num_heads=4,
    ff_hidden_dim=64
)

output = model.forward(x)

print("\nTransformed Embeddings:\n", output)
