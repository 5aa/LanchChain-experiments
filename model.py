import requests
import sseclient
import json
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.history = []

    @abstractmethod
    def generate(self, prompt, stream=True):
        pass

    def clear_history(self):
        self.history = []


class TextGenWebUI(BaseModel):
    def __init__(self, url="http://127.0.0.1:5000/v1/chat/completions", api_key=None):
        super().__init__(api_key)
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def generate(self, prompt, stream=True):
        self.history.append({"role": "user", "content": prompt})
        data = {"mode": "instruct", "stream": stream, "messages": self.history}

        response = requests.post(
            self.url, headers=self.headers, json=data, verify=False, stream=stream
        )

        if stream:
            client = sseclient.SSEClient(response)
            assistant_message = ""
            for event in client.events():
                payload = json.loads(event.data)
                chunk = payload["choices"][0]["delta"]["content"]
                assistant_message += chunk
                yield chunk
            self.history.append({"role": "assistant", "content": assistant_message})
        else:
            response_json = response.json()
            assistant_message = response_json["choices"][0]["message"]["content"]
            self.history.append({"role": "assistant", "content": assistant_message})
            yield assistant_message


class OpenAIModel(BaseModel):
    def __init__(self, api_key, model="gpt-4"):
        super().__init__(api_key)
        self.url = "https://api.openai.com/v1/chat/completions"
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def generate(self, prompt, stream=True):
        self.history.append({"role": "user", "content": prompt})
        data = {"model": self.model, "messages": self.history, "stream": stream}

        response = requests.post(
            self.url, headers=self.headers, json=data, stream=stream
        )

        if stream:
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                        if line.strip() != "[DONE]":
                            chunk = json.loads(line)["choices"][0]["delta"].get(
                                "content", ""
                            )
                            if chunk:
                                yield chunk
        else:
            response_json = response.json()
            assistant_message = response_json["choices"][0]["message"]["content"]
            self.history.append({"role": "assistant", "content": assistant_message})
            yield assistant_message


def get_model(model_type, **kwargs):
    if model_type.lower() == "textgenwebui":
        return TextGenWebUI(**kwargs)
    elif model_type.lower() == "openai":
        return OpenAIModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
