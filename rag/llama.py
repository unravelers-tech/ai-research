from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
import torch
from time import time
from typing import Dict, Optional, Union

class LlamaModel:
    def __init__(self, base_model: str, system_message: str = "You are an AI assistant designed to answer questions accurately and concisely.") -> None:
        """
        Initializes the LlamaModel class with a tokenizer, model, and text generator pipeline.
        
        Args:
            base_model (str): The name or path to the pre-trained model.
            system_message (str, optional): Default system message for the model. Defaults to an AI assistant message.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.system_message = system_message

        # Load the pre-trained model with optimized settings for low CPU usage
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Initialize the text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    def create_prompt(self, 
                      user_message: str, 
                      message_format: str = "Question: {user_message} Answer:", 
                      system_message: Optional[str] = None, 
                      **kwargs) -> Dict[str, Union[str, bool]]:
        """
        Creates a formatted prompt based on the user message and the system message.

        Args:
            user_message (str): The message from the user.
            message_format (str, optional): The format template for the message. Defaults to a question-answer format.
            system_message (str, optional): Optionally provide a different system message for this prompt. Defaults to None.
            **kwargs: Additional arguments to format the message.

        Returns:
            dict: A dictionary with the generated prompt and a flag indicating it's a generated template.
        """
        # Update system message if a new one is provided
        if system_message:
            self.system_message = system_message

        # Format the user message according to the provided template
        formatted_user_message = message_format.format(user_message=user_message, **kwargs)

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": formatted_user_message},
        ]
        
        # Create prompt template with a generation flag
        prompt = {
            'generated': True,
            'text': self.generator.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        }
        
        return prompt

    def create(self, prompt: Union[str, Dict[str, Union[str, bool]]], temperature: float = 0.7, max_length: int = 1024) -> Dict[str, Union[str, float]]:
        """
        Generates text based on the provided prompt or template.

        Args:
            prompt (Union[str, dict]): The prompt for text generation, either as a string or a generated prompt template.
            temperature (float, optional): Sampling temperature for controlling randomness. Defaults to 0.7.
            max_length (int, optional): Maximum length of the generated text. Defaults to 1024.

        Returns:
            dict: A dictionary containing the generated answer and the generation time.
        """
        start_time = time()

        # Check if the prompt is generated using create_prompt
        if isinstance(prompt, dict) and prompt.get('generated', False):
            prompt_text = prompt['text']
        else:
            prompt_text = self.create_prompt(prompt)['text']

        terminators = [
            self.generator.tokenizer.eos_token_id,
            self.generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate response
        sequences = self.generator(
            prompt_text,
            do_sample=True,
            top_p=0.9,
            temperature=temperature,
            num_return_sequences=1,
            eos_token_id=terminators,
            max_new_tokens=max_length,
            return_full_text=False,
            pad_token_id=terminators[0],
        )

        answer = sequences[0]['generated_text']
        end_time = time()
        total_time = round(end_time - start_time, 2)
        
        # Return the answer and time taken
        return {
            'answer': answer,
            'time': f'{total_time}s'
        }
