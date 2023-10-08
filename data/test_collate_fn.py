import torch
from datasets import load_from_disk

from data.collate_fn import make_collate_fn
from data.arxiv import make_arxiv_abstracts_dataset
from lib.encoders import byte_pair_encoder, byte_pair_decoder

def test_make_collate_fn(tmp_path):
    fn = make_collate_fn(100)
    batch_size = 12
    
    dataset_path = make_arxiv_abstracts_dataset(tmp_path, encoder=byte_pair_encoder)

    all_data = load_from_disk(dataset_path).with_format('torch')
    loader = torch.utils.data.DataLoader(all_data['test'], batch_size=batch_size, collate_fn=fn)
    
    x, y = next(iter(loader))
    assert x.shape[0] == 12
    assert x.shape[1] == 100
    assert y.shape[0] == 12
    assert y.shape[1] == 100
