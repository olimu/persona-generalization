"""
Pretty Debug Print for Tokenized Inputs

Visualizes input_ids, labels, and attention_mask with:
  • Underlined tokens → within attention_mask (mask=1)
  • Blue tokens → included in labels (label ≠ -100)
  • Black tokens → not in labels (label = -100)
"""

from __future__ import annotations
from typing import Optional, Union, List, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class TokenColors:
    """ANSI escape codes for terminal formatting."""
    RESET = "\033[0m"
    BLUE = "\033[94m"
    BLACK = "\033[30m"
    UNDERLINE = "\033[4m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Background hints for better visibility
    BG_LIGHT = "\033[48;5;255m"
    

def debug_print_tokens(
    input_ids: Union[Any, List[int]],
    labels: Union[Any, List[int]],
    attention_mask: Union[Any, List[int]],
    tokenizer,
    title: str = "Token Debug View",
    show_ids: bool = False,
    max_width: int = 120,
) -> None:
    """
    Pretty-print tokenized inputs with visual formatting.
    
    Args:
        input_ids: Token IDs (1D tensor or list)
        labels: Label IDs, typically -100 for ignored tokens (1D tensor or list)
        attention_mask: Attention mask, 1 for attended tokens (1D tensor or list)
        tokenizer: HuggingFace tokenizer for decoding
        title: Header title for the output
        show_ids: If True, also print token IDs below the tokens
        max_width: Maximum line width before wrapping
    
    Formatting:
        • Underlined → token is attended (attention_mask = 1)
        • Blue       → token is in labels (label ≠ -100)
        • Black/Dim  → token is masked in labels (label = -100)
    """
    C = TokenColors
    
    # Convert to lists if tensors
    if HAS_TORCH and isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.squeeze().tolist()
    if HAS_TORCH and isinstance(labels, torch.Tensor):
        labels = labels.squeeze().tolist()
    if HAS_TORCH and isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.squeeze().tolist()
    
    # Handle single integers
    if isinstance(input_ids, int):
        input_ids = [input_ids]
    if isinstance(labels, int):
        labels = [labels]
    if isinstance(attention_mask, int):
        attention_mask = [attention_mask]
    
    # Decode each token individually, preserving special tokens
    tokens = []
    for token_id in input_ids:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        # Make whitespace visible
        decoded = decoded.replace('\n', '↵').replace('\t', '→').replace(' ', '·')
        if decoded == '':
            decoded = '∅'
        tokens.append(decoded)
    
    # Build header
    header_line = f"{'─' * 60}"
    print(f"\n{C.BOLD}┌{header_line}┐{C.RESET}")
    print(f"{C.BOLD}│ {title:^58} │{C.RESET}")
    print(f"{C.BOLD}├{header_line}┤{C.RESET}")
    
    # Legend
    print(f"{C.BOLD}│{C.RESET} Legend: {C.UNDERLINE}underline{C.RESET}=attended  ", end="")
    print(f"{C.BLUE}blue{C.RESET}=in labels  ", end="")
    print(f"{C.DIM}dim{C.RESET}=masked", end="")
    print(f"          {C.BOLD}│{C.RESET}")
    print(f"{C.BOLD}└{header_line}┘{C.RESET}\n")
    
    # Build formatted output
    formatted_parts = []
    current_line_len = 0
    lines = []
    current_line = []
    
    for i, (token, label, mask) in enumerate(zip(tokens, labels, attention_mask)):
        # Determine styling
        styles = []
        
        # Underline if in attention mask
        if mask == 1:
            styles.append(C.UNDERLINE)
        
        # Color based on labels
        if label != -100:
            styles.append(C.BLUE)
        else:
            styles.append(C.DIM)
        
        # Build the styled token
        style_prefix = ''.join(styles)
        styled_token = f"{style_prefix}{token}{C.RESET}"
        
        # Calculate display length (without ANSI codes)
        display_len = len(token) + 1  # +1 for separator
        
        # Check if we need to wrap
        if current_line_len + display_len > max_width and current_line:
            lines.append(current_line)
            current_line = []
            current_line_len = 0
        
        current_line.append(styled_token)
        current_line_len += display_len
    
    # Don't forget the last line
    if current_line:
        lines.append(current_line)
    
    # Print the tokens
    separator = f"{C.DIM}│{C.RESET}"
    for line in lines:
        print(f" {separator.join(line)}")
    
    # Optionally show token IDs
    if show_ids:
        print(f"\n{C.BOLD}Token IDs:{C.RESET}")
        id_lines = []
        current_id_line = []
        current_len = 0
        
        for i, (token, token_id) in enumerate(zip(tokens, input_ids)):
            id_str = f"{token_id:>5}"
            if current_len + 6 > max_width and current_id_line:
                id_lines.append(current_id_line)
                current_id_line = []
                current_len = 0
            current_id_line.append(id_str)
            current_len += 6
        
        if current_id_line:
            id_lines.append(current_id_line)
        
        for line in id_lines:
            print(f" {' '.join(line)}")
    
    # Summary statistics
    total = len(input_ids)
    attended = sum(attention_mask)
    labeled = sum(1 for l in labels if l != -100)
    
    print(f"\n{C.DIM}─────────────────────────────────────{C.RESET}")
    print(f"{C.BOLD}Summary:{C.RESET} {total} tokens │ {attended} attended │ {labeled} in labels")
    print()


def debug_print_batch(
    input_ids: Any,
    labels: Any,
    attention_mask: Any,
    tokenizer,
    max_samples: int = 4,
    **kwargs
) -> None:
    """
    Debug print for batched inputs.
    
    Args:
        input_ids: Batched token IDs (2D tensor)
        labels: Batched labels (2D tensor)
        attention_mask: Batched attention mask (2D tensor)
        tokenizer: HuggingFace tokenizer
        max_samples: Maximum number of samples to print
        **kwargs: Additional arguments passed to debug_print_tokens
    """
    batch_size = input_ids.shape[0]
    num_to_print = min(batch_size, max_samples)
    
    print(f"\n{'═' * 70}")
    print(f" BATCH DEBUG VIEW ({num_to_print}/{batch_size} samples)")
    print(f"{'═' * 70}")
    
    for i in range(num_to_print):
        debug_print_tokens(
            input_ids[i],
            labels[i],
            attention_mask[i],
            tokenizer,
            title=f"Sample {i + 1}/{batch_size}",
            **kwargs
        )

