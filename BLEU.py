import torch
import math
from collections import Counter
from typing import List, Tuple, Dict, Any


def bleu_stats(hypothesis: List[int], reference: List[int]) -> torch.Tensor:
    """
    Compute statistics for BLEU using PyTorch.
    
    Args:
        hypothesis: List of token indices for the hypothesis text
        reference: List of token indices for the reference text
    
    Returns:
        torch.Tensor: Statistics vector with 10 elements for BLEU calculation
    """
    stats = torch.zeros(10, dtype=torch.float32)
    
    # Length of hypothesis and reference
    stats[0] = len(hypothesis)
    stats[1] = len(reference)
    
    # Compute n-gram matches for n from 1 to 4
    for n in range(1, 5):
        # Create n-grams for hypothesis
        s_ngrams = Counter(
            tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)
        )
        
        # Create n-grams for reference
        r_ngrams = Counter(
            tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)
        )
        
        # Count matched n-grams
        matched = sum((s_ngrams & r_ngrams).values())
        stats[2 * n] = max(matched, 0)
        stats[2 * n + 1] = max(len(hypothesis) + 1 - n, 0)
    
    return stats


def bleu(stats: torch.Tensor) -> torch.Tensor:
    """
    Compute BLEU score given statistics.
    
    Args:
        stats: Tensor of statistics from bleu_stats
    
    Returns:
        torch.Tensor: BLEU score
    """
    # Return 0 if any of the matched counts are 0
    if torch.any(stats == 0):
        return torch.tensor(0.0)
    
    c, r = stats[0], stats[1]
    
    # Calculate precision for each n-gram level
    precisions = stats[2::2] / stats[3::2]
    log_precisions = torch.log(precisions)
    
    # Average the log precision values
    log_bleu_prec = torch.mean(log_precisions)
    
    # Apply brevity penalty
    brevity_penalty = torch.min(torch.tensor(0.0), 1 - r / c)
    
    # Calculate final BLEU score
    bleu_score = torch.exp(brevity_penalty + log_bleu_prec)
    
    return bleu_score


def get_bleu(hypotheses: List[List[int]], references: List[List[int]]) -> torch.Tensor:
    """
    Get BLEU score for a batch of hypotheses and references.
    
    Args:
        hypotheses: List of token index lists for each hypothesis
        references: List of token index lists for each reference
    
    Returns:
        torch.Tensor: BLEU score scaled to 0-100
    """
    stats = torch.zeros(10, dtype=torch.float32)
    
    # Accumulate stats from all hypothesis-reference pairs
    for hyp, ref in zip(hypotheses, references):
        stats += bleu_stats(hyp, ref)
    
    # Calculate BLEU score and scale to 0-100
    return 100 * bleu(stats)


def idx_to_word(x: List[int], vocab: Any) -> str:
    """
    Convert token indices to a string of words, filtering out special tokens.
    
    Args:
        x: List of token indices
        vocab: Vocabulary object with an itos (index-to-string) attribute
    
    Returns:
        str: Space-joined string of words
    """
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
    
    return " ".join(words)


def corpus_bleu(model_outputs: torch.Tensor, targets: torch.Tensor, vocab: Any) -> torch.Tensor:
    """
    Calculate corpus-level BLEU score for model outputs against reference targets.
    
    Args:
        model_outputs: Tensor of shape [batch_size, seq_len] with predicted token indices
        targets: Tensor of shape [batch_size, seq_len] with reference token indices
        vocab: Vocabulary object with itos attribute
    
    Returns:
        torch.Tensor: Corpus BLEU score
    """
    # Convert tensors to lists
    hypotheses = []
    references = []
    
    # Process each sequence in the batch
    for i in range(model_outputs.size(0)):
        # Get non-padding tokens
        hypothesis = [token.item() for token in model_outputs[i] if token != vocab.stoi['<pad>']]
        reference = [token.item() for token in targets[i] if token != vocab.stoi['<pad>']]
        
        hypotheses.append(hypothesis)
        references.append(reference)
    
    # Calculate BLEU score
    return get_bleu(hypotheses, references)


def sentence_bleu(hypothesis: torch.Tensor, reference: torch.Tensor, vocab: Any) -> torch.Tensor:
    """
    Calculate BLEU score for a single sentence.
    
    Args:
        hypothesis: Tensor of shape [seq_len] with predicted token indices
        reference: Tensor of shape [seq_len] with reference token indices
        vocab: Vocabulary object with itos attribute
    
    Returns:
        torch.Tensor: BLEU score for the sentence
    """
    # Convert tensors to lists, removing padding tokens
    hyp = [token.item() for token in hypothesis if token != vocab.stoi['<pad>']]
    ref = [token.item() for token in reference if token != vocab.stoi['<pad>']]
    
    # Calculate stats and BLEU score
    stats = bleu_stats(hyp, ref)
    return 100 * bleu(stats)
