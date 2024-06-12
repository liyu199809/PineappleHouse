from transformers import AutoTokenizer, AutoModel


class LLM:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()
        print("Model Initialized")

    def model_predict(self, prompt):
        # arguments to send the API
        # TODO: support override kwargs
        kwargs = {
            "temperature": 0.0,
            "max_tokens": 2000,
            "stream": False,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["[SQL]:"],
        }
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response