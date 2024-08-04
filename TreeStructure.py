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
        return f"{self.model_name} with score {self.score:.3f}"
    
def generate(llm: str, prompt: str, context: str) -> str:
    url = 'http://127.0.0.1:11434/api/generate'
    f = open('input.json', 'r')
    data = json.load(f)
    data['model'] = llm
    data['prompt'] = "Answer the question: " + prompt + ", given the context: " + context
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

def nodes_at_depth(heap, depth):
    start_index = 2**depth - 1
    end_index = 2**(depth + 1) - 1
    return heap[start_index:end_index]

if __name__ == "__main__":
    llmNodes = init_tree(['llama3', 'gemma', 'mistral', 'phi', 'deepseek-llm', 'qwen2', 'orca-mini'])
    
    print("LLMs initialized in the tree:")
    for _ in llmNodes:
        print(_, end=' | ')
    print('####')
    
    # atlas_dataset = init_atlas_dataset()
    q = "What is applied anthropology?"
    a = "Applied anthropology refers to the practical application of anthropological theories and methods to solve contemporary social problems. It involves working with communities to understand and address issues such as poverty, health, education, and inequality."
    
    d_max = 2
    c = [""]*len(nodes_at_depth(llmNodes, d_max))
    for i in range(d_max, -1, -1):
        li = nodes_at_depth(llmNodes, i)
        if len(c) < len(li):
            c.extend([""]*(len(li)-len(c)))
        for j in range(0, len(li), 2):
            response1 = generate(li[j].model_name, q, c.pop(0))
            context = response1
            if j+1<len(li):
                response2 = generate(li[j+1].model_name, q[j+1], c.pop(0))
                context += response2
            c.append(context)

    print("THE FINAL OUTPUT IS: " + c[0])

    # score_phi = 1
    # for _ in llmNodes:
    #     if _.model_name == 'phi':
    #         _.score = score_phi
    #         break
    
    # heapq.heapify(llmNodes)
    # print('After updating score: ')
    # for _ in llmNodes:
    #     print(_)

    