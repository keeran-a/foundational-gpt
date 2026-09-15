"""Decoder-only transformer, pretrained weight import, and text generation."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True
    position_embedding: str = "learned"
    rope_theta: float = 10000.0
    activation: str = "gelu"

    def __post_init__(self):
        dimensions = (self.block_size, self.vocab_size, self.n_layer, self.n_head, self.n_embd)
        if any(value <= 0 for value in dimensions):
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("Embedding width must be divisible by the number of heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        if self.position_embedding not in {"learned", "rotary"}:
            raise ValueError("position_embedding must be learned or rotary")
        if self.position_embedding == "rotary" and (self.n_embd // self.n_head) % 2:
            raise ValueError("Rotary embeddings require an even width per attention head")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive and finite")
        if self.activation not in {"gelu", "gelu_tanh"}:
            raise ValueError("activation must be gelu or gelu_tanh")


def apply_rotary(features, cosine, sine):
    dtype = features.dtype
    even, odd = features.float()[..., 0::2], features.float()[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2).to(dtype)


class LayerNorm(nn.Module):
    def __init__(self, width, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, features):
        return F.layer_norm(features, self.weight.shape, self.weight, self.bias, eps=1e-5)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary = config.position_embedding == "rotary"
        if self.rotary:
            head_width = config.n_embd // config.n_head
            frequency = config.rope_theta ** (-torch.arange(0, head_width, 2, dtype=torch.float32) / head_width)
            angles = torch.outer(torch.arange(config.block_size, dtype=torch.float32), frequency)
            self.register_buffer("rope_cosine", angles.cos(), persistent=False)
            self.register_buffer("rope_sine", angles.sin(), persistent=False)

    def forward(self, features):
        batch, length, width = features.shape
        projected = self.c_attn(features).reshape(batch, length, 3, self.n_head, width // self.n_head)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        if self.rotary:
            cosine, sine = self.rope_cosine[:length], self.rope_sine[:length]
            query = apply_rotary(query, cosine, sine)
            key = apply_rotary(key, cosine, sine)
        context = F.scaled_dot_product_attention(
            query, key, value, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self.resid_dropout(self.c_proj(context))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.approximation = "tanh" if config.activation == "gelu_tanh" else "none"

    def forward(self, features):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(features), approximate=self.approximation)))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = FeedForward(config)

    def forward(self, features):
        features = features + self.attn(self.ln_1(features))
        return features + self.mlp(self.ln_2(features))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Parameter paths retain compatibility with existing GPT-2 checkpoints.
        layers = {
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList(DecoderLayer(config) for _ in range(config.n_layer)),
            "ln_f": LayerNorm(config.n_embd, config.bias),
        }
        if config.position_embedding == "learned":
            layers = {"wte": layers.pop("wte"), "wpe": nn.Embedding(config.block_size, config.n_embd), **layers}
        self.transformer = nn.ModuleDict(layers)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._initialize)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=residual_std)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def get_num_params(self, non_embedding=True):
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            total -= self.transformer.wpe.weight.numel()
        return total

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.block_size:
            raise ValueError(f"Expected [batch, length] tokens with length 1..{self.config.block_size}")
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError("Targets must have the same shape as input tokens")
        features = self.transformer.wte(tokens)
        if "wpe" in self.transformer:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            features = features + self.transformer.wpe(positions)
        features = self.transformer.drop(features)
        for layer in self.transformer.h:
            features = layer(features)
        features = self.transformer.ln_f(features)
        if targets is None:
            return self.lm_head(features[:, -1:, :]), None
        logits = self.lm_head(features)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def crop_block_size(self, block_size):
        if not 0 < block_size <= self.config.block_size:
            raise ValueError("Context can only be reduced to a positive length")
        if "wpe" in self.transformer:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size].detach().clone())
            self.transformer.wpe.num_embeddings = block_size
        for layer in self.transformer.h:
            if layer.attn.rotary:
                layer.attn.rope_cosine = layer.attn.rope_cosine[:block_size]
                layer.attn.rope_sine = layer.attn.rope_sine[:block_size]
        self.config.block_size = block_size

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        from transformers import GPT2LMHeadModel

        sizes = {
            "gpt2": (12, 12, 768), "gpt2-medium": (24, 16, 1024),
            "gpt2-large": (36, 20, 1280), "gpt2-xl": (48, 25, 1600),
        }
        overrides = override_args or {}
        if model_type not in sizes or set(overrides) - {"dropout"}:
            raise ValueError("Use a GPT-2 model name and an optional dropout override")
        layers, heads, width = sizes[model_type]
        model = cls(GPTConfig(
            block_size=1024, vocab_size=50257, n_layer=layers, n_head=heads,
            n_embd=width, bias=True, activation="gelu_tanh", **overrides,
        ))
        source = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
        cls.import_weights(model.state_dict(), source)
        return model

    @staticmethod
    @torch.no_grad()
    def import_weights(destination, source):
        source = {key: value for key, value in source.items()
                  if not key.endswith((".attn.bias", ".attn.masked_bias"))}
        if source.keys() != destination.keys():
            raise ValueError("Pretrained parameter names do not match the model")
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        for name, target in destination.items():
            value = source[name].t() if name.endswith(transposed) else source[name]
            if target.shape != value.shape:
                raise ValueError(f"Pretrained parameter shape mismatch: {name}")
            target.copy_(value)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        groups = [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=device_type == "cuda")

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("Generation requires a nonempty [batch, length] prompt")
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Token count must be nonnegative and temperature must be positive and finite")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        for _ in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.config.block_size:])
            scores = logits[:, -1] / temperature
            if top_k is not None:
                cutoff = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
                scores = scores.masked_fill(scores < cutoff, -torch.inf)
            next_token = torch.multinomial(scores.softmax(dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
