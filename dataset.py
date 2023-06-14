from torch.utils.data import Dataset, IterableDataset
import torchaudio

import glob

class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""
    
    def __init__(self, audio_dir):
        super().__init__()
        
        self.filenames = glob.glob(audio_dir+"/*.wav")
        _, self.sr = torchaudio.load(self.filenames[0])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        return torchaudio.load(self.filenames[index])[0]
        
class AmiDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.source_data = load_dataset(
            "edinburghcstr/ami",
            'ihm', split='train', streaming=True)
        
    def __iter__(self):
      return iter(self.source_data)
