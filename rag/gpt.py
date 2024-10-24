from openai import OpenAI
from time import time
from typing import Dict, Optional, Union

class GPTModel:
    def __init__(self, api_key: str, model_name: str = "gpt-4o", system_message: str = "You are an AI assistant designed to answer questions accurately and concisely.") -> None:
        """
        Initializes the OpenAIModel class with an API key and model name for interacting with OpenAI's GPT models.

        Args:
            api_key (str): The OpenAI API key for authentication.
            model_name (str, optional): The model name to use, e.g., 'gpt-4'. Defaults to 'gpt-4'.
            system_message (str, optional): Default system message for the model. Defaults to an AI assistant message.
        """

        self.model = OpenAI(api_key= api_key)
        self.model_name = model_name
        self.system_message = system_message

    def create_prompt(self, 
                      user_message: str, 
                      message_format: str = "Question: {user_message} Answer:", 
                      system_message: Optional[str] = None, 
                      **kwargs) -> Dict[str, Union[str, bool]]:
        """
        Creates a formatted prompt for OpenAI's chat-based models.

        Args:
            user_message (str): The message from the user.
            message_format (str, optional): The format template for the message. Defaults to a question-answer format.
            system_message (str, optional): Optionally provide a different system message for this prompt. Defaults to None.
            **kwargs: Additional arguments to format the message.

        Returns:
            dict: A dictionary with the system and user messages.
        """
        # Update system message if a new one is provided
        if system_message:
            self.system_message = system_message

        # Format the user message according to the provided template
        formatted_user_message = message_format.format(user_message=user_message, **kwargs)

        # Create the prompt structure for the OpenAI API
        prompt = {
            'generated': True,
            'messages': [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": formatted_user_message},
            ]
        }

        return prompt

    def create(self, prompt: Union[str, Dict[str, Union[str, bool]]], temperature: float = 1, max_length: int = 1024) -> Dict[str, Union[str, float]]:
        """
        Generates text based on the provided prompt or template using OpenAI's API.

        Args:
            prompt (Union[str, dict]): The prompt for text generation, either as a string or a generated prompt template.
            temperature (float, optional): Sampling temperature for controlling randomness. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.

        Returns:
            dict: A dictionary containing the generated answer and the generation time.
        """
        start_time = time()

        # Check if the prompt is generated
        if isinstance(prompt, dict) and prompt.get('generated', False):
            messages = prompt['messages']
        else:
            # If a string is passed, we create a simple prompt
            messages = self.create_prompt(prompt)['messages']

        # Make the API call to OpenAI
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_length,
            n=1,
            stop=None,
            top_p=1.0,
            presence_penalty=0,
            frequency_penalty=0
        )

        answer = response.choices[0].message.content
        end_time = time()
        total_time = round(end_time - start_time, 2)

        # Return the answer and time taken
        return {
            'answer': answer,
            'time': f'{total_time}s'
        }
