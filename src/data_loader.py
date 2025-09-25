from typing import Dict, Any, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

class _StreamingDataset(IterableDataset):
    """
    An iterable dataset that streams data from a Hugging Face dataset.
    """

    def __init__(self, config: Dict[str, Any], tokenizer: AutoTokenizer, split: str = "train"):
        self.config = config
        self.tokenizer = tokenizer
        
        self.dataset = load_dataset(
            config["data"]["train_path"],
            split=config["data"]["dataset_split"],
            streaming=config["data"]["streaming"]
        )
        
        if split == "train":
            self.max_samples = config["data"].get("max_samples")
            self.skip_samples = 0
        else: # eval
            self.max_samples = config["data"].get("num_eval_samples")
            self.skip_samples = config["data"].get("max_samples", 0)

        if self.skip_samples > 0:
            self.dataset = self.dataset.skip(self.skip_samples)

        self.max_length = config["training"]["max_length"]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        count = 0
        for sample in self.dataset:
            if self.max_samples and count >= self.max_samples:
                break
            
            # Heuristic to find the text field
            text = ""
            if "text" in sample:
                text = sample["text"]
            elif "content" in sample:
                text = sample["content"]
            elif "instruction" in sample and "output" in sample:
                text = sample["instruction"] + "\n" + sample["output"]
            else:
                # If no common field is found, try to concatenate all string fields
                text = " ".join(str(v) for v in sample.values() if isinstance(v, str))

            if text:
                tokenized_sample = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                yield {
                    "input_ids": tokenized_sample["input_ids"].squeeze(0),
                    "attention_mask": tokenized_sample["attention_mask"].squeeze(0)
                }
                count += 1

def get_data_loader(config: Dict[str, Any], tokenizer: AutoTokenizer, split: str = "train") -> DataLoader:
    """
    Creates a DataLoader for training the SAE.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        tokenizer (AutoTokenizer): The tokenizer to use.
        split (str): The dataset split to use ("train" or "eval").

    Returns:
        DataLoader: The PyTorch DataLoader.
    """
    dataset = _StreamingDataset(config, tokenizer, split=split)
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=8, # Adjust based on your system
        pin_memory=True
    )
