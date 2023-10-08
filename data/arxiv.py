import os
from datasets import load_dataset


def make_arxiv_abstracts_dataset(base_path: str, encoder, min_tokens=50, test_size=0.2):
    dataset = load_dataset("gfissore/arxiv-abstracts-2021")['train']
    
    dataset = dataset.map(
        lambda x: { 'text': encoder(build_batch(x)) },
        batched=True,
    )
    dataset = dataset.select_columns(['text'])
    dataset = dataset.filter(lambda x: len(x['text']) > min_tokens)

    dataset = dataset.train_test_split(test_size=test_size, shuffle=True)

    os.makedirs(base_path, exist_ok=True)
    dataset.save_to_disk(base_path, num_proc=4)

    return base_path


def build_batch(batch):
    out = []
    for i in range(len(batch['title'])):
        out.append(f'TITLE: {batch["title"][i]}\nCATEGORIES: {", ".join(batch["categories"][i])}\nABSTRACT: {batch["abstract"][i]} <|endoftext|>')
        
    return out
