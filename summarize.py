import re
from model import get_model
from tokenizer import estimate_tokens


# basic sentence-level chunking with overlap
def chunk_text(text, max_chunk_size=4000, overlap=200):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence + " ")

        if current_chunk_tokens + sentence_tokens > max_chunk_size:
            chunks.append(current_chunk.strip())
            overlap_text = " ".join(current_chunk.split()[-overlap:])
            current_chunk = overlap_text + " " + sentence
            current_chunk_tokens = estimate_tokens(current_chunk)
        else:
            current_chunk += sentence + " "
            current_chunk_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def summarize(text, model, max_chunk_size=4000, stream=False):
    chunks = chunk_text(text, max_chunk_size)
    summaries = []

    print(f"Text broken into {len(chunks)} chunk(s):")
    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = estimate_tokens(chunk)
        print(f"Chunk {i}: {chunk_tokens} tokens")
        prompt = f"Please summarize the following text concisely (respond with only the summary):\n\n{chunk}"

        print(f"Summarizing chunk {i}:")
        summary = ""
        for chunk in model.generate(prompt, stream=stream):
            summary += chunk
            print(chunk, end="", flush=True)
        print()  # New line after summary

        summaries.append(summary)

    if len(summaries) > 1:
        # TODO: combined summaries could still exceed max_chunk_size
        combined_summaries = " ".join(summaries)
        final_prompt = f"Please provide a concise overall summary of the following summaries (respond with only the summary):\n\n{combined_summaries}"

        print("Generating final summary:")
        final_summary = ""
        for chunk in model.generate(final_prompt, stream=stream):
            final_summary += chunk
            print(chunk, end="", flush=True)
        print()  # New line after final summary

        return final_summary
    else:
        return summaries[0]


def summarize_file(file_path, model, max_chunk_size=4000, stream=False):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        print(f"File: {file_path}")
        print(f"Total tokens in file: {estimate_tokens(text)}")
        return summarize(text, model, max_chunk_size, stream)
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except IOError:
        return f"Error: Unable to read file at {file_path}"


if __name__ == "__main__":
    """ 
    For OpenAI
    model = get_model("openai", api_key="your-api-key-here", model="gpt-4")

    Example usage with text
    text = "Your long text here..."
    print("Summarizing text:")
    print(f"Total tokens in text: {estimate_tokens(text)}")
    summary = summarize(text, model, stream=False)
    print("\nSummary of text:")
    print(summary)
    """

    # For TextGen WebUI
    model = get_model("textgenwebui")

    # Example usage with file
    # file_path = "alice-in-wonderland-ch1.txt"
    file_path = "reddit-post.txt"
    print("\nSummarizing file:")
    file_summary = summarize_file(
        file_path,
        model,
        max_chunk_size=4000,
        stream=True,
    )
    print("\nSummary of file:")
    print(file_summary)
