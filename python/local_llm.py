# python/local_llm.py
import sys
from transformers import pipeline

def main():
    # Read the prompt text from standard input.
    prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.")
        return

    # Create a text generation pipeline using a local model (GPT-2 in this example).
    generator = pipeline('text-generation', model='gpt2')
    
    # Generate text based on the prompt.
    # max_length determines the maximum length of the generated output.
    output = generator(prompt, max_length=200, num_return_sequences=1)
    answer = output[0]['generated_text']
    
    # Print the generated answer.
    print(answer)

if __name__ == '__main__':
    main()