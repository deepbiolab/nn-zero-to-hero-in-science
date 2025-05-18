from tqdm import tqdm
import torch
import numpy as np

from esm.tokenization import EsmSequenceTokenizer
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
)


def embed_sequence(model: ESM3InferenceClient, sequence: str) -> LogitsOutput:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    config = LogitsConfig(sequence=True)
    output = model.logits(protein_tensor, config)
    return output


def calc_perplexity(client, seqs=[]) -> float:
    """
    Calculate the average perplexity for a group of sequences

    Args:
        model: ESM3InferenceClient instance
        seqs: List of generated CDRH3 sequences

    Returns:
        float: Average perplexity
    """
    # Configure logits output
    ntokens = 0
    nlls = []
    tokenizer = EsmSequenceTokenizer()
    pad_token_id = tokenizer.pad_token_id
    for seq in tqdm(seqs):
        output = embed_sequence(client, seq)  # Get logits: [1, seq_len, vocab_size]
        logits = output.logits.sequence.squeeze(0)  # [seq_len, vocab_size]
        # Get token ids
        input_ids = tokenizer(seq, return_tensors="pt")["input_ids"].squeeze(
            0
        )  # [seq_len]
        # Predict next token
        target = input_ids[1:]  # Actual next token
        pred_logits = logits[:-1]  # Logits for predicting next token

        # Calculate cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            pred_logits, target, reduction="sum", ignore_index=pad_token_id
        )
        n_token = (target != pad_token_id).sum().item()
        nlls.append(loss.item())
        ntokens += n_token

    avg_nll = sum(nlls) / ntokens
    perplexity = np.exp(avg_nll)
    return perplexity


# Usage example
if __name__ == "__main__":
    from esm.models.esm3 import ESM3
    client = ESM3.from_pretrained("esm3_sm_open_v1")
    ppl = calc_perplexity(client, ["AAA", "CCC", "ACC"])
    print(f"Average Perplexity: {ppl:.2f}")
