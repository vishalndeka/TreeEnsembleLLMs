import heapq
import random

# @dataclass(order=True)
class LLMNode():
    def __init__(self, model_name:str, score:float = 0.0) -> None:
        self.model_name = model_name
        self.score = score

    def __lt__(self, other) -> bool:
        return self.score > other.score
    
    def __repr__(self) -> str:
        return f"{self.model_name} with score {self.score}"

if __name__ == "__main__":
    llama = LLMNode("llama", random.random())
    gemma = LLMNode("gemma", random.random())
    mistral = LLMNode("mistral", random.random())
    deepseek = LLMNode("deepseek", random.random())
    phi = LLMNode("phi", random.random())

    llmList = [llama, gemma, mistral, deepseek, phi]
    heapq.heapify(llmList)
    for llm in llmList:
        print(llm)

    