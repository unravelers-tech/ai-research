import torch
from transformers import GPT2Model, GPT2Tokenizer
from torch import nn


class SimpleGPT2(nn.Module):
    """
    Class for the GPT-2 model with an additional linear layer.

    Args:
        hidden_size (int): The hidden layer size of GPT-2.
        num_classes (int): Number of output classes.
        max_seq_len (int): Maximum sequence length.
        gpt_model_name (str): The name of the pre-trained GPT-2 model to load.
    """
    def __init__(self, hidden_size: int, num_classes: int, max_seq_len: int, gpt_model_name: str):
        super(SimpleGPT2, self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size * max_seq_len, num_classes)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_tokenizer(self) -> GPT2Tokenizer:
        """
        Returns the GPT-2 tokenizer.

        Returns:
            GPT2Tokenizer: The tokenizer for GPT-2.
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
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size, -1))
        return linear_output