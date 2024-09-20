from custom_memory import ChatMLFormatMemory

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.textgen import TextGen


model_url = "http://localhost:5000"

# Mistral prompt
# template = "<s>[INST] {question} [/INST]"

# ChatML prompt
template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
{chat_history}<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = PromptTemplate(
    template=template, input_variables=["system_message", "chat_history", "question"]
)

memory = ChatMLFormatMemory(memory_key="chat_history", return_messages=False)

llm = TextGen(
    model_url=model_url,
    max_new_tokens=1000,
    temperature=0.7,
    top_p=0.9,
    top_k=20,
    repetition_penalty=1.15,
)

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False, memory=memory)

variables = {
    "system_message": "You are a helpful assistant.",
    "question": "Explain LLMs to a fifth grader.",
}

response = llm_chain.predict(question=variables["question"]).lstrip(": ")

print("--- You ---\n" + variables["question"])
print("--- Assistant ---\n" + response)

while True:
    user_input = input("--- You ---\n")  # Get user input
    response = llm_chain.predict(question=user_input).lstrip(": ")

    print("--- Assistant ---\n" + response)
