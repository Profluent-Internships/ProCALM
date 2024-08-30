import os
from typing import List, Optional

from tokenizers import AddedToken, Tokenizer

END_OF_SPAN_TOKEN = "<eos_span>"  # nosec
PAD_TOKEN_ID = 0


def get_tokenizer(model=None) -> Tokenizer:
    fname = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    tokenizer: Tokenizer = Tokenizer.from_file(fname)

    return tokenizer
