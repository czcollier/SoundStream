from torch.utils.data import Dataset, IterableDataset
import torchaudio
from datasets import load_dataset

class AmiDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.source_data = load_dataset(
            "edinburghcstr/ami",
            'ihm', split='train', streaming=True)

    def __iter__(self):
      return iter(self.source_data)
