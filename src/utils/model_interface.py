from abc import ABC, abstractmethod
import logging
import torch

class ModelInterface(ABC):
    @abstractmethod
    def generate(self, inputs, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_device(self) -> torch.device:
        pass

class UnslothModelWrapper(ModelInterface):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate(self, inputs, **kwargs) -> torch.Tensor:
        inputs = inputs.to(self.model.device)
        return self.model.generate(inputs, **kwargs)
    
    def get_device(self) -> torch.device:
        return self.model.device
    
    def get_max_length(self) -> int:
        return self.model.config.max_position_embeddings

class TGIModelWrapper(ModelInterface):
    def __init__(self, client):
        self.client = client
        self._device = torch.device("cpu")  # TGI runs on server side

        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        
    def generate(self, inputs, **kwargs) -> str:
        # Only include parameters that were explicitly provided
        tgi_params = {}
        param_mapping = {
            'temperature': 'temperature',
            'top_p': 'top_p',
            'max_new_tokens': 'max_tokens'
        }
        
        for src_param, tgi_param in param_mapping.items():
            if src_param in kwargs:
                tgi_params[tgi_param] = kwargs[src_param]
        
        # Generate completion
        try:
            completion = self.client.chat.completions.create(
                model="tgi",  # not used by TGI
                messages=inputs,
                stream=False,
                **tgi_params
                )
        except Exception as e:
            print(f"Error generating completion: {e}")
            return None
        
        # Return response text directly
        return completion.choices[0].message.content
    
    def get_device(self) -> torch.device:
        return self._device 
    
    def get_max_length(self):
        return None