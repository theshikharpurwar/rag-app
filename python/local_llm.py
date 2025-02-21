# python/local_llm.py
import sys
from transformers import pipeline

def main():
    prompt = sys.stdin.read().strip()
    generator = pipeline('text-generation', model='qwen2.5-7b-instruct')
    output = generator(prompt, max_length=200, num_return_sequences=1)
    answer = output[0]['generated_text']
    print(answer)

if __name__ == '__main__':
    main()