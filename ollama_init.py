import time
import os

time.sleep(15)
os.system('ollama pull mistral')
os.system('ollama pull gemma')
os.system('ollama pull qwen2')
os.system('ollama pull llama3')
os.system('ollama pull deepseek-llm')
os.system('ollama pull orca-mini')
os.system('ollama pull phi')