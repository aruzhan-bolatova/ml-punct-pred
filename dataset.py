"""
Load kz-transformers/multidomain-kazakh-dataset from Hugging Face.

The dataset has a single "default" config with ~80M rows. Full dataset: ~25GB.
"""

from typing import Optional

from datasets import load_dataset

DATASET_ID = "kz-transformers/multidomain-kazakh-dataset"


def load_multidomain_kazakh(
    streaming: bool = False,
    subset_size: Optional[int] = None,
    language_filter: Optional[str] = "kaz",
):
    """
    Load the multidomain Kazakh dataset from Hugging Face.

    Args:
        streaming: If True, use streaming to avoid loading full dataset into memory.
        subset_size: If set, load only the first N rows (useful for testing).
        language_filter: Filter by predicted_language ("kaz", "rus", or None for all).

    Returns:
        Hugging Face Dataset (or IterableDataset if streaming=True)
    """
    load_kwargs = {
        "path": DATASET_ID,
        "split": "train",
        "streaming": streaming,
    }

    if subset_size and not streaming:
        load_kwargs["split"] = f"train[:{subset_size}]"

    dataset = load_dataset(**load_kwargs)

    if language_filter:
        dataset = dataset.filter(
            lambda x: x["predicted_language"] == language_filter
        )

    if subset_size and streaming:
        dataset = dataset.take(subset_size)

    return dataset


if __name__ == "__main__":
    # Quick test: load a small subset
    print("Loading dataset (first 1000 rows)...")
    ds = load_multidomain_kazakh(subset_size=1000)
    print(f"Loaded {len(ds)} examples")
    print(f"Columns: {ds.column_names}")
    print("\nSample:")
    print(ds[0])
