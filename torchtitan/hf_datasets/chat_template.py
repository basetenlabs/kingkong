# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chat template module for handling conversation formatting and token masking.

This module provides utilities for:
1. Formatting conversations using HuggingFace chat templates
2. Creating token masks to only train on assistant responses
3. Supporting various chat formats (OpenAI-style messages, etc.)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.tools.logging import logger

# Enable debug tracing with environment variable: TORCHTITAN_DEBUG_TOKENS=1
DEBUG_TOKENS = os.environ.get("TORCHTITAN_DEBUG_TOKENS", "0") == "1"


@dataclass
class TokensAndMask:
    """Container for tokens and their corresponding training mask.

    The mask indicates which tokens should be trained on (True) vs ignored (False).
    Typically, we mask out user messages and only train on assistant responses.

    Note: tokens should have bos and eos tokens in them, otherwise attention masking
    won't work properly later.
    """

    tokens: list[int]
    mask: list[bool]  # True for trainable tokens, False otherwise

    def __len__(self) -> int:
        return len(self.tokens)

    def split(self, first_len: int) -> tuple["TokensAndMask", "TokensAndMask"]:
        """Split at index, returning (prefix, suffix)."""
        tokens_1, tokens_2 = self.tokens[:first_len], self.tokens[first_len:]
        mask_1, mask_2 = self.mask[:first_len], self.mask[first_len:]
        return TokensAndMask(tokens_1, mask_1), TokensAndMask(tokens_2, mask_2)


@dataclass
class MessageRow:
    """Represents a single conversation with messages.

    This is the expected format for chat/instruction datasets:
    - messages: List of message dicts with 'role' and 'content' keys
    - tools: Optional list of tool definitions for function calling
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "MessageRow":
        """Create MessageRow from a dataset sample dict."""
        messages = item.get("messages", item.get("conversations", []))
        if not messages:
            raise ValueError(
                f"Expected 'messages' or 'conversations' key in data. "
                f"Available keys: {list(item.keys())}"
            )
        # Normalize message format if needed
        normalized = []
        for msg in messages:
            # Handle different key names (role/from, content/value)
            role = msg.get("role", msg.get("from", ""))
            content = msg.get("content", msg.get("value", ""))
            normalized.append({"role": role, "content": content})
        return cls(messages=normalized, tools=item.get("tools"))


@dataclass
class ChatTemplate:
    """Handles chat template formatting and mask generation.

    This class wraps a HuggingFace tokenizer's chat template and provides
    utilities for:
    1. Formatting conversations into a single string
    2. Identifying which parts of the formatted text are assistant responses
    3. Creating masks for training only on assistant content
    """

    renderer: Callable[[MessageRow], str]
    start_of_generation: str
    end_of_generation: str

    def format(self, message_row: MessageRow) -> str:
        """Format a MessageRow into a string using the chat template."""
        return self.renderer(message_row)

    @classmethod
    def render_with_tools(cls, tokenizer: Any, message_row: MessageRow) -> str:
        """Render messages using HF tokenizer's chat template, with optional tools."""
        result = tokenizer.apply_chat_template(
            message_row.messages,
            tools=message_row.tools,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        # Clean up empty think blocks that some models produce
        return result.replace("<think>\n\n</think>\n\n", "")

    @classmethod
    def from_hf_tokenizer(
        cls,
        tokenizer: Any,
        start_of_generation_sequence: str,
        end_of_generation_sequence: str,
    ) -> "ChatTemplate":
        """Create a ChatTemplate from a HuggingFace tokenizer.

        Args:
            tokenizer: A HuggingFace tokenizer with a chat_template
            start_of_generation_sequence: Marks start of assistant content
            end_of_generation_sequence: Marks end of assistant content

        Returns:
            ChatTemplate instance
        """
        return ChatTemplate(
            renderer=lambda message_row: cls.render_with_tools(tokenizer, message_row),
            start_of_generation=start_of_generation_sequence,
            end_of_generation=end_of_generation_sequence,
        )


def calc_generation_mask(
    tokens: list[int],
    start_of_generation_seq: list[int],
    end_of_generation_seq: list[int],
) -> list[bool]:
    """Calculate mask for trainable tokens based on generation boundaries.

    Takes in a token list and returns a mask representing where the generation
    starts and stops. The mask is True for tokens that should be trained on
    (assistant responses) and False otherwise.

    For example, let's say the tokens are:
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    And the start sequence = [2, 3], end sequence = [7, 8]

    Then the mask will be:
    [0, 0, 0, 1, 1, 1, 1, 1, 0]  # Tokens 4-8 are trainable (including end seq)

    There may be multiple start and end sequences present in the token list.
    In that case, the starts and ends are assumed to be alternating.

    Args:
        tokens: The full token sequence
        start_of_generation_seq: Token sequence marking start of assistant content
        end_of_generation_seq: Token sequence marking end of assistant content

    Returns:
        List of booleans, True for trainable tokens
    """
    if start_of_generation_seq == end_of_generation_seq:
        raise ValueError(
            f"start_of_generation_seq may not equal end_of_generation_seq "
            f"(={start_of_generation_seq})"
        )

    n = len(tokens)
    start_len = len(start_of_generation_seq)
    end_len = len(end_of_generation_seq)

    end_of_starts = []  # Positions right after start sequences
    start_of_ends = []  # Positions where end sequences begin

    looking_for_start = True
    for i in range(n):
        if (
            looking_for_start
            and i + start_len <= n
            and tokens[i : i + start_len] == start_of_generation_seq
        ):
            end_of_starts.append(i + start_len)
            looking_for_start = False
        if (
            not looking_for_start
            and i + end_len <= n
            and tokens[i : i + end_len] == end_of_generation_seq
        ):
            start_of_ends.append(i)
            looking_for_start = True

    if len(end_of_starts) != len(start_of_ends):
        if DEBUG_TOKENS:
            logger.warning(
                "[DEBUG MASK] MISMATCH! Found %d starts but %d ends",
                len(end_of_starts),
                len(start_of_ends),
            )
            logger.warning("  end_of_starts positions: %s", end_of_starts)
            logger.warning("  start_of_ends positions: %s", start_of_ends)
        raise ValueError("number of start and end generation sequences is not equal")

    if DEBUG_TOKENS:
        logger.warning(
            "[DEBUG MASK] calc_generation_mask found %d trainable regions:",
            len(end_of_starts),
        )
        for idx, (start, end) in enumerate(zip(end_of_starts, start_of_ends)):
            region_len = end - start + end_len
            logger.warning(
                "  Region %d: positions %d-%d (%d tokens)",
                idx,
                start,
                end - 1 + end_len,
                region_len,
            )

    # Build the mask - trainable region includes tokens from after start seq
    # up to and including the end sequence
    mask = [False] * n
    for end_of_start, start_of_end in zip(end_of_starts, start_of_ends):
        for i in range(end_of_start, start_of_end + end_len):
            mask[i] = True

    return mask


def tokenize_and_mask(
    tokenizer: "HfTokenizerWithPadToken",
    chat_template: ChatTemplate,
) -> Callable[[str], TokensAndMask]:
    """Create a function that tokenizes text and creates a training mask.

    The mask will be True for tokens corresponding to assistant responses,
    and False for everything else (system prompts, user messages, etc.).

    Args:
        tokenizer: Tokenizer with encode method and pad_id
        chat_template: ChatTemplate for identifying assistant spans

    Returns:
        Function that takes formatted text and returns TokensAndMask
    """

    def _tokenize_and_mask(text: str) -> TokensAndMask:
        # Tokenize with EOS and append pad token to signal end of conversation
        tokens = tokenizer.encode(text, add_eos=True)
        tokens.append(tokenizer.pad_id)

        # Get the token sequences for start/end markers
        start_seq = tokenizer.encode(chat_template.start_of_generation)
        end_seq = tokenizer.encode(chat_template.end_of_generation)

        # Calculate mask based on generation boundaries
        mask = calc_generation_mask(tokens, start_seq, end_seq)

        if DEBUG_TOKENS:
            # Debug logging
            eos_id = tokenizer.eos_id
            pad_id = tokenizer.pad_id

            # Try to decode end marker
            im_end_tokens = tokenizer.encode(chat_template.end_of_generation)
            im_end_id = im_end_tokens[0] if im_end_tokens else None

            logger.warning("=" * 80)
            logger.warning("[DEBUG TOKENS] tokenize_and_mask called")
            logger.warning("  Text length: %d chars", len(text))
            logger.warning("  Token count: %d", len(tokens))
            logger.warning(
                "  Start sequence: %s (tokens: %s)",
                chat_template.start_of_generation,
                start_seq,
            )
            logger.warning(
                "  End sequence: %s (tokens: %s)",
                chat_template.end_of_generation,
                end_seq,
            )
            logger.warning(
                "  EOS token ID: %s, PAD token ID: %s, end_marker ID: %s",
                eos_id,
                pad_id,
                im_end_id,
            )

            # Count trainable tokens
            trainable_count = sum(mask)
            logger.warning(
                "  Trainable tokens: %d / %d (%.1f%%)",
                trainable_count,
                len(mask),
                100 * trainable_count / len(mask) if mask else 0,
            )

            # Show last N tokens with their mask status
            last_n = 20
            logger.warning("  Last %d tokens:", last_n)
            for i in range(max(0, len(tokens) - last_n), len(tokens)):
                try:
                    token_str = tokenizer.decode([tokens[i]])
                except Exception:
                    token_str = "<decode_error>"
                logger.warning(
                    "    [%d] token=%d (%r) trainable=%s",
                    i,
                    tokens[i],
                    token_str,
                    mask[i],
                )
            logger.warning("=" * 80)

        return TokensAndMask(tokens, mask)

    return _tokenize_and_mask


def tokenize_no_mask(tokenizer: HuggingFaceTokenizer) -> Callable[[str], TokensAndMask]:
    """Create a function that tokenizes text without masking (all tokens trainable).

    This is used for pretraining-style datasets where all tokens should be trained on.

    Args:
        tokenizer: Tokenizer with encode method

    Returns:
        Function that takes text and returns TokensAndMask with all True mask
    """

    def _tokenize_no_mask(text: str) -> TokensAndMask:
        tokens = tokenizer.encode(text)
        return TokensAndMask(tokens, mask=[True] * len(tokens))

    return _tokenize_no_mask


class HfTokenizerWithPadToken(HuggingFaceTokenizer):
    """HuggingFace tokenizer that ensures pad token is available.

    This extends HuggingFaceTokenizer to add pad_token and pad_id attributes
    which are required for finetuning datasets that need padding.

    If pad_token is not defined in the tokenizer config, falls back to:
    1. EOS token (most common for decoder-only models)
    2. UNK token (if EOS is also not available)
    """

    def __init__(self, tokenizer_path: str):
        super().__init__(tokenizer_path)

        # Try to get pad_token from config
        self.pad_token: str | None = self._get_token_from_config(
            self.config, "pad_token"
        )

        # Fall back to EOS token if pad_token is not defined
        if self.pad_token is None:
            self.pad_token = self._get_token_from_config(self.config, "eos_token")
            if self.pad_token is not None:
                logger.info(
                    f"pad_token not found in tokenizer config, using eos_token "
                    f"'{self.pad_token}' as pad_token"
                )

        # Fall back to UNK token if EOS is also not defined
        if self.pad_token is None:
            self.pad_token = self._get_token_from_config(self.config, "unk_token")
            if self.pad_token is not None:
                logger.warning(
                    f"Neither pad_token nor eos_token found, using unk_token "
                    f"'{self.pad_token}' as pad_token"
                )

        if self.pad_token is None:
            raise ValueError(
                "Could not find pad_token, eos_token, or unk_token in tokenizer config. "
                "Please ensure your tokenizer has at least one of these tokens defined."
            )

        self.pad_id: int = self.token_to_id(self.pad_token)
        if self.pad_id is None:
            raise ValueError(
                f"Could not find token ID for pad_token '{self.pad_token}'. "
                f"The token may not be in the vocabulary."
            )

        assert isinstance(
            self.pad_id, int
        ), f"pad_id must be int, got {type(self.pad_id)}"
        assert isinstance(
            self.pad_token, str
        ), f"pad_token must be str, got {type(self.pad_token)}"
