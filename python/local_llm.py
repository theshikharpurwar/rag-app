# python/local_llm.py
import sys
from transformers import pipeline

def main():
    prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.")
        sys.stdout.flush()
        return

    try:
        generator = pipeline('text-generation', model='gpt2')
        output = generator(prompt, max_length=200, num_return_sequences=1)
        answer = output[0]['generated_text']
        print(answer)
        sys.stdout.flush()
    except Exception as e:
        print("Error: " + str(e))
        sys.stdout.flush()

if __name__ == '__main__':
    main()