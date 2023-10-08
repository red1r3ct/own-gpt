import sys
import os
import json
import numpy as np

if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    if not os.path.exists(dataset_dir):
        raise Exception(f"Dataset {dataset_dir} does not exist")
    print(f"Decoding {dataset_dir}\n")

    dataset_type = input("Choose train of val dataset: ")
    assert dataset_type in ("train", "val")

    with np.load(f"{dataset_dir}/train_val_split.npz", allow_pickle=True) as data:
        embeddings = data[dataset_type]

    with open(os.path.join(dataset_dir, "embeddings.json"), "r") as file:
        embeddings_index = json.load(file)
    inverted_embeddings_index = {}
    for char, index in embeddings_index.items():
        inverted_embeddings_index[index] = char

    while True:
        entry_index = int(input("Choose entry index: "))
        assert entry_index < len(embeddings)

        print("Encoded doc:")
        for index in embeddings[entry_index]:
            print(index, end="")
        print()
        print("Decoded doc:")
        for index in embeddings[entry_index]:
            print(inverted_embeddings_index[index], end="")
        print()
