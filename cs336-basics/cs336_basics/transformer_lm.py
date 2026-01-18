import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from cs336_basics.bpe.tokenizer import Tokenizer
from cs336_basics.nn import Embedding, Linear, RMSNorm, RoPEConfig, TransformerBlock, softmax


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.device = device

        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, device=device) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def forward(
        self,
        token_ids: Int[Tensor, "bs seq_len"],
    ) -> Float[Tensor, "bs seq_len d_model"]:
        x = self.token_embeddings(token_ids)  # (bs, seq_len, d_model)

        rope_config = RoPEConfig(theta=self.theta, d_k=self.d_model // self.num_heads, max_seq_len=self.context_length)
        for layer in self.layers:
            x = layer(x, rope_config=rope_config)

        x = self.lm_head(self.ln_final(x))  # (bs, seq_len, vocab_size)

        return x

    @torch.no_grad()
    def generate(
        self,
        tokens: list[int],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
    ):
        assert temperature > 0.0, "Temperature must be positive"
        assert top_p > 0.0 and top_p <= 1.0, "Top-p must be in (0, 1]"
        assert max_tokens > 0, "Max tokens must be positive"

        tokens = tokens[:]

        for _ in range(max_tokens):
            token_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            logits = self.forward(token_ids)
            last_logits = logits[:, -1, :]

            probs = softmax(last_logits / (temperature + 1e-9))
            probs = probs.squeeze(0)  # (vocab_size,)

            if top_p < 1.0:
                prob_sorted, prob_ids = probs.sort(descending=True)
                prob_sorted_cumsum = torch.cumsum(prob_sorted, dim=0)
                # Keep the smallest set of highest-prob tokens whose cumsum >= top_p
                # Keep token i if cumsum before adding it is < top_p
                mask = prob_sorted_cumsum - prob_sorted < top_p
                prob_sorted[~mask] = 0.0
                prob_sorted = prob_sorted / torch.sum(prob_sorted)

                # Unsort probabilities back to original order
                probs = torch.zeros_like(probs)
                probs[prob_ids] = prob_sorted

            token_id = int(torch.multinomial(probs, num_samples=1).item())
            tokens.append(token_id)
            yield token_id

    def completion(
        self,
        text: str,
        tokenizer: Tokenizer,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 500,
    ):
        tokens = tokenizer.encode(text)
        for token_id in self.generate(tokens, temperature=temperature, top_p=top_p, max_tokens=max_tokens):
            text = tokenizer.decode([token_id])

            if text in tokenizer.special_tokens:
                break

            yield text
