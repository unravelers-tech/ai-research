import torch
from transformers import BertModel, BertTokenizer
from torch import nn


class SimpleBert(nn.Module):
    """
    Class for the BERT model with an additional linear layer.

    Args:
        hidden_size (int): The hidden layer size of BERT.
        num_classes (int): Number of output classes.
        max_seq_len (int): Maximum sequence length.
        bert_model_name (str): The name of the pre-trained BERT model to load.
    """
    def __init__(self, hidden_size: int, num_classes: int, max_seq_len: int, bert_model_name: str):
        super(SimpleBert, self).__init__()
        self.bertmodel = BertModel.from_pretrained(bert_model_name)
        self.fc1 = nn.Linear(hidden_size * max_seq_len, num_classes)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def get_tokenizer(self) -> BertTokenizer:
        """
        Returns the Bert tokenizer.

        Returns:
            BertTokenizer: The tokenizer for Bert.
        """
        return self.tokenizer

    def forward(self, input_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_id (torch.Tensor): Token IDs of input texts.
            mask (torch.Tensor): Attention mask for the input texts.

        Returns:
            torch.Tensor: The output after the linear layer.
        """
        bert_out, _ = self.bertmodel(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = bert_out.shape[0]
        linear_output = self.fc1(bert_out.view(batch_size, -1))
        return linear_output
