import pytest
import os

from data.arxiv import make_arxiv_abstracts_dataset
from lib.encoders import noop_encoder


def test_make_arxiv_abstracts_dataset(tmp_path):
    make_arxiv_abstracts_dataset(tmp_path, encoder=noop_encoder)
    
    assert os.path.exists(str(tmp_path) + '/dataset_dict.json')
