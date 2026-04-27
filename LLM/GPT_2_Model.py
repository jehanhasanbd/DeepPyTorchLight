import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        # shape

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dims = d_out // num_heads

        # Wq , Wk, Wv
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        # Linear layer for combine head outputs
        self.output_proj = nn.Linear(d_out, d_out)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        no_of_batch, no_of_token, embedding_dim = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(
            no_of_batch, no_of_token, self.num_heads, self.head_dims
        )
        keys = keys.view(no_of_batch, no_of_token, self.num_heads, self.head_dims)
        values = values.view(no_of_batch, no_of_token, self.num_heads, self.head_dims)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # attention = Q @ (K)T
        attention_score = queries @ keys.transpose(2, 3)

        # masking
        attention_score = attention_score.masked_fill(
            self.mask.bool()[:no_of_token, :no_of_token], -torch.inf
        )

        # normalize
        d_k = keys.shape[-1]
        attention_weight = torch.softmax(
            attention_score / d_k ** 0.5, dim=-1
        )

        # dropout
        attention_weight = self.dropout(attention_weight)

        context_vector = (attention_weight @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(
            no_of_batch, no_of_token, self.d_out
        )
        context_vector = self.output_proj(context_vector)
        return context_vector


class Feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['embedding_dim'], 4 * config['embedding_dim']),
            nn.GELU(),
            nn.Linear(4 * config['embedding_dim'], config['embedding_dim'])
        )

    def forward(self, x):
        return self.layers(x)


class LayerNormalize(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt_(var + self.eps)
        return (self.scale * x_normalized) + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=config['embedding_dim'],
            d_out=config['embedding_dim'],
            context_length=config['context_length'],
            num_heads=config['num_of_heads'],
            dropout=config['drop_rate'],
            qkv_bias=config['qkv_bias']

        )

        self.ff = Feedforward(config)

        self.normalization1 = LayerNormalize(config['embedding_dim'])
        self.normalization2 = LayerNormalize(config['embedding_dim'])

        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.normalization1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.normalization2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.positional_embedding = nn.Embedding(config['context_length'], config['embedding_dim'])
        self.dropout_embedding = nn.Dropout(config['drop_rate'])

        self.transformer_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['num_of_layers'])]
        )
        self.final_normalization = LayerNormalize(config['embedding_dim'])
        self.output_head = nn.Linear(
            config['embedding_dim'], config['vocab_size'], bias=False
        )

    def forward(self, inputs):
        batch_size, no_of_token = inputs.shape
        token_embedding = self.token_embedding(inputs)
        positional_embedding = self.positional_embedding(
            torch.arange(no_of_token, device=inputs.device)
        )

        x = token_embedding + positional_embedding
        x = self.dropout_embedding(x)
        x = self.transformer_block(x)
        x = self.final_normalization(x)
        logits = self.output_head(x)
        return logits


def no_of_params_LLM(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params -= sum(p.numel() for p in model.output_head.parameters())

    size_in_byte = total_params * 4
    size_in_mb = size_in_byte / (1024 * 1024)

    return (f"Total no of parameters:  {total_params:,}",
            f"Size of parameters:  {size_in_mb:.2} MB")


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 768,
    "num_of_heads": 12,
    "num_of_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}