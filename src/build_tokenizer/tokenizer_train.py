import os
from typing import Optional

import fire

from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def load_training_corpus(
        data_dir: str,
        batch_size: int = 1024):
    print("Loading dataset...")
    dataset = load_dataset("parquet",
                           data_dir=data_dir)

    dataset = dataset['train']
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]


def train(
        data_dir: str,
        save_to: str,
        model_id: Optional[str] = "answerdotai/ModernBERT-base",
        batch_size: Optional[int] = 1024):
    old_tokenizer = AutoTokenizer.from_pretrained(model_id)

    training_corpus = load_training_corpus(data_dir, batch_size)

    unused_tokens = [f"<|unused_{i}|>".upper() for i in range(16)]

    unused_tokens.extend(["<|HISTORY|>",
                          "<|PHONE_NUMBER|>",
                          "<|EMAIL|>",
                          "<|IP_ADDRESS|>",
                          "<|URL|>",
                          "<|DATE|>",
                          "<|TIME|>"
                          ])

    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus,
                                                      vocab_size=len(old_tokenizer),
                                                      special_tokens_map=old_tokenizer.special_tokens_map,
                                                      new_special_tokens=unused_tokens,
                                                      num_processes=4,
                                                      show_progress=True)

    tokenizer.save_pretrained(save_to)
    print(f"Tokenizer saved to{save_to}")


if __name__ == '__main__':
    fire.Fire(train)
