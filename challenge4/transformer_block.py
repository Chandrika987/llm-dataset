import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((1, 1, dim))
        self.beta = np.zeros((1, 1, dim))
        self.eps = eps

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class MultiHeadSelfAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Wq = np.random.randn(embed_dim, embed_dim)
        self.Wk = np.random.randn(embed_dim, embed_dim)
        self.Wv = np.random.randn(embed_dim, embed_dim)
        self.Wo = np.random.randn(embed_dim, embed_dim)

    def split_heads(self, x):
        B, T, C = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def forward(self, x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        scores = Q @ K.transpose(0, 1, 3, 2)
        scores /= np.sqrt(self.head_dim)

        attention = softmax(scores)
        out = attention @ V

        out = out.transpose(0, 2, 1, 3)
        B, T, _, _ = out.shape
        out = out.reshape(B, T, self.embed_dim)

        return out @ self.Wo


class FeedForward:
    def __init__(self, embed_dim, hidden_dim):
        self.W1 = np.random.randn(embed_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim)
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        x = np.maximum(0, x @ self.W1 + self.b1)
        return x @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_hidden_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn.forward(x)
        x = self.norm1(x)

        x = x + self.ffn.forward(x)
        x = self.norm2(x)

        return x
