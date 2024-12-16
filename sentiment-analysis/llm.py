import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import warnings
from typing import List, Tuple, Dict
from models.simple_bert import SimpleBert
from models.simple_gpt2 import SimpleGPT2

warnings.simplefilter(action='ignore', category=FutureWarning)


class TextDataset(Dataset):
    """
    Dataset for processing text using a tokenizer.

    Args:
        text (List[str]): List of texts to process.
        tokenizer: Tokenizer to convert text into tokens.
    """
    def __init__(self, text: List[str], tokenizer):
        self.texts = [tokenizer(t, padding='max_length', max_length=128, truncation=True, return_tensors="pt") for t in text]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.texts[idx]

def get_model(model_name: str, model_path : str) -> nn.Module:
    """
    Loads a model based on the provided name.

    Args:
        model_name (str): Model name ('BERT' or 'GPT2').
        model_path (str): Model path 'modelstore/...'
    Returns:
        nn.Module: Loaded model.
    """
    if model_name == 'BERT':
        model = SimpleBert(hidden_size=768, num_classes=3, max_seq_len=128, bert_model_name="bert-base-cased")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    elif model_name == 'GPT2':
        model = SimpleGPT2(hidden_size=768, num_classes=3, max_seq_len=128, gpt_model_name="gpt2")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        raise ValueError("Unsupported model name. Choose 'BERT' or 'GPT2'.")

    return model

def data_preprocess(text: List[str], model: nn.Module) -> DataLoader:
    """
    Converts texts into a DataLoader format.

    Args:
        text (List[str]): Texts to process.
        model (nn.Module): Model to determine the tokenizer.

    Returns:
        DataLoader: Prepared DataLoader with texts.
    """
    tokenizer = model.get_tokenizer()
    text_dataset = TextDataset(text, tokenizer)
    text_dataloader = DataLoader(text_dataset, batch_size=2)
    return text_dataloader

def predict(model: nn.Module, dataloader: DataLoader) -> Tuple[int, float]:
    """
    Performs predictions based on the model and DataLoader.

    Args:
        model (nn.Module): Loaded model.
        dataloader (DataLoader): Dataset for predictions.

    Returns:
        Tuple[int, float]: Predicted class and confidence.
    """
    device = torch.device("cpu")
    model.eval()  # Переведення моделі в режим оцінки
    with torch.no_grad():
        for test_input in dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            probabilities = F.softmax(output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy().flatten().tolist()
            confidence = torch.max(probabilities, dim=1)[0].cpu().numpy().flatten().tolist()

            return predicted_labels[0], confidence[0]