import tiktoken


# OpenAI estimate, for accurate llama 3 count use their tokenizer
def estimate_tokens(text: str, encoding_name: str = "gpt-4") -> int:
    # encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)


if __name__ == "__main__":
    # Example usage
    text = "This is a test string to estimate the number of tokens."
    num_tokens = estimate_tokens(text)
    print(f"Text: {text}")
    print(f"Number of tokens: {num_tokens}")
