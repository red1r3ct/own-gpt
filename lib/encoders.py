import tiktoken

base_encoding = tiktoken.get_encoding('cl100k_base')

pad_token = base_encoding.n_vocab + 1
encoding = tiktoken.Encoding(
    name="cl100k",
    pat_str=base_encoding._pat_str,
    mergeable_ranks=base_encoding._mergeable_ranks,
    special_tokens={
        **base_encoding._special_tokens,
        "<|pad|>": pad_token,
    }
)

allowed_special = set(encoding._special_tokens.keys())

vocab_size = encoding.n_vocab

def byte_pair_encoder(batch):
    return encoding.encode_batch(batch, allowed_special=allowed_special)

def byte_pair_decoder(batch):
    return encoding.decode_batch(batch)

def noop_encoder(batch):
    return batch

def noop_decoder(batch):
    return batch

