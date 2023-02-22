import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, random_split
import os

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# data path
questionare_data_path=r"C:\Users\kalik\PycharmProjects\cs224n\datasets\CANS\questionare"
vignette_data_path = r"C:\Users\kalik\PycharmProjects\cs224n\datasets\CANS\vignette"

# Data model
class CANSDataModel:
    def __init__(self, description='', ratings='', category=''):
        self.description = description
        self.ratings = ratings
        self.category = category


# Define the dataset class
class CANSDataset(Dataset):
    def __init__(self, questionare_data_path, vignette_data_path):
        self.load_questionare(questionare_data_path)
        self.load_vignette(vignette_data_path)

        for q in self.cans_questionare:
            for l in q.ratings:
                self.labels = int(l.strip())
            for w in q.description:
                self.texts = w.strip()

        for v in self.cans_vignette:
            for w in v.split():
                self.texts = w.strip()

        print(f"we loaded {len(self.cans_questionare)} questionare and {len(self.cans_vignette)} vignette")


    def load_questionare(self, data_path):
        splitter = "Ratings and Descriptions"
        cans_data_objects = []
        # Read in the text file
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".txt"):
                    filepath = os.path.join(subdir, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read().split(splitter)
                        obj = CANSDataModel()
                        obj.category = files
                        obj.description = text[0:1]
                        obj.ratings = text[1:]
                        cans_data_objects.append(obj)

        # append data to main object
        self.cans_questionare = cans_data_objects


    def load_vignette(self, data_path):
        # Read in the text file
        vignette = []
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".txt"):
                    filepath = os.path.join(subdir, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        vignette.append(f.read())

        # append data to main object
        self.cans_vignette = vignette


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the input text
        inputs = tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        # Convert the label to a tensor
        label = torch.tensor(self.labels[idx])
        return inputs, label


if __name__ == '__main__':
    # Create the dataset object
    dataset = CANSDataset(questionare_data_path, vignette_data_path)

    # Split the dataset into train and eval sets
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    print(train_dataset, eval_dataset)