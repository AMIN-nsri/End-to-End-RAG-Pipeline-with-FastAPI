# Generation logic will go here 

import os
from typing import List, Optional
from together import Together


def format_prompt(context_chunks: List[str], question: str, system_prompt: Optional[str] = None) -> str:
    """
    Format the prompt for the LLM: system instructions, context, and user question.
    """
    context = '\n\n'.join(context_chunks)
    sys_prompt = system_prompt or (
        "You are a helpful and accurate assistant. Use the provided context to answer the question. "
        "If the answer is not in the context, say 'I don't know based on the provided information.'"
    )
    prompt = f"<|system|> {sys_prompt}\n<|context|> {context}\n<|user|> {question}\n<|assistant|>"
    return prompt


def generate_answer(
    prompt: str,
    model_name: str = 'meta-llama/Meta-Llama-3-70B-Instruct-Turbo',
    temperature: float = 0.2,
    max_tokens: int = 512,
    top_p: float = 0.95,
) -> str:
    """
    Call Together API to generate an answer from the prompt.
    """
    client = Together()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    # The Together API returns an iterator for streaming; get the first chunk
    if hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content
    # If streaming, concatenate all chunks
    elif hasattr(response, '__iter__'):
        return ''.join(chunk.choices[0].delta.content for chunk in response if chunk.choices and chunk.choices[0].delta.content)
    else:
        return "[Error: No response from LLM]"


def call_llm_with_context(
    context_chunks: List[str],
    question: str,
    model_name: str = 'meta-llama/Meta-Llama-3-70B-Instruct-Turbo',
    temperature: float = 0.2,
    max_tokens: int = 512,
    top_p: float = 0.95,
    system_prompt: Optional[str] = None,
) -> str:
    """
    High-level wrapper: formats prompt and calls the LLM.
    """
    prompt = format_prompt(context_chunks, question, system_prompt)
    return generate_answer(
        prompt,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )


if __name__ == '__main__':
    # Example usage for testing the generation pipeline
    import os
    context_chunks = [
        "The Seine is the most important river in northern France. It flows through Paris and empties into the English Channel.",
        "The Loire is the longest river in France, flowing through the center of the country and into the Atlantic Ocean."
    ]
    question = "What are the main rivers in France?"
    print('Formatting prompt...')
    prompt = format_prompt(context_chunks, question)
    print(f'Prompt:\n{prompt}\n')
    print('Calling Together API to generate answer...')
    # Make sure TOGETHER_API_KEY is set in your environment or .env file
    answer = generate_answer(
        prompt,
        model_name='meta-llama/Meta-Llama-3-70B-Instruct-Turbo',
        temperature=0.2,
        max_tokens=256,
        top_p=0.95
    )
    print(f'LLM Answer:\n{answer}') 