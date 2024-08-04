import heapq
import random
import requests
import json
import os

# @dataclass(order=True)
class LLMNode():
    def __init__(self, model_name:str, score:float = 0.0) -> None:
        self.model_name = model_name
        self.score = score

    def __lt__(self, other) -> bool:
        return self.score > other.score
    
    def __repr__(self) -> str:
        return f"{self.model_name} with score {self.score}"
    
def generate(llm: str, prompt:str) -> str:
    url = 'http://127.0.0.1:11434/api/generate'
    f = open('input.json', 'r')
    data = json.load(f)
    data['model'] = llm
    data['prompt'] = prompt
    response = requests.post(url, json=data)
    assert response.status_code == 200
    dictionary = json.loads(response.text)
    f.close()
    return dictionary['response']

def init_atlas_dataset():
    dataset = []
    directory = 'atlas-converse'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if f.endswith('json'):
            with open(f, 'r') as file:
                dataset.extend(json.load(f))
        f.close()

    return dataset

def init_tree(llmList:list) -> list:
    li = []
    for _ in llmList:
        li.append(LLMNode(_, random.random()))
    heapq.heapify(li)
    return li

if __name__ == "__main__":
    llmNodes = init_tree(['llama', 'gemma', 'mistral', 'deepseek', 'phi'])
    
    print("LLMs being used in the tree:")
    for _ in llmNodes:
        print(_.model_name, end=' | ')

    
    atlas_dataset = init_atlas_dataset()
    

    score_phi = 1
    for _ in llmNodes:
        if _.model_name == 'phi':
            _.score = score_phi
            break
    
    heapq.heapify(llmNodes)
    print('After updating score: ')
    for _ in llmNodes:
        print(_)

    