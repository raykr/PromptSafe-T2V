import pandas as pd
from torch.utils.data import Dataset

class PairDataset(Dataset):
    # items: (malicious, rewritten, benign)
    def __init__(self, csv_path):
        # prompt, rewritten_prompt, benign_prompt -> malicious, rewritten, benign
        self.data = pd.read_csv(csv_path)
        # 直接使用DataFrame的列，不需要转置
        self.malicious = self.data['prompt'].tolist()
        self.rewritten = self.data['rewritten_prompt'].tolist()
        self.benign = self.data['benign_prompt'].tolist()
        print(f"Loaded {len(self.malicious)} samples")

    def __len__(self):
        return len(self.malicious)

    def __getitem__(self, i):
        return {
            "malicious": self.malicious[i], 
            "rewritten": self.rewritten[i], 
            "benign": self.benign[i]
        }

