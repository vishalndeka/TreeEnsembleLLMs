import heapq
import random
import requests
import json
import os
import math

# @dataclass(order=True)
class LLMNode():
    def __init__(self, model_name:str, score:float = 0.0) -> None:
        self.model_name = model_name
        self.score = score

    def __lt__(self, other) -> bool:
        return self.score > other.score
    
    def __repr__(self) -> str:
        return f"{self.model_name} with score {self.score:.3f}"

class TreeEnsemble():
    def __init__(self, tree_list: list) -> None:
        self.tree_structure = self.init_tree(tree_list)
        self.max_depth = math.floor(math.log(len(tree_list))) + 1

    def init_tree(self, llmList: list) -> list:
        li = []
        for _ in llmList:
            li.append(LLMNode(_, random.random()))
        heapq.heapify(li)
        return li
    
    def nodes_at_depth(self, depth: int) -> list:
        start_index = 2**depth - 1
        end_index = 2**(depth + 1) - 1
        return self.tree_structure[start_index:end_index]
    
    def restructure_tree(self, scores: list) -> None:
        for i in range(len(self.tree_structure)):
            self.tree_structure[i].score = scores[i]
        heapq.heapify(self.tree_structure)



# auxiliary methods
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

def query_tree(tree: TreeEnsemble, query: str) -> str:
    context = [""]*len(tree.nodes_at_depth(tree.max_depth))
    for i in range(tree.max_depth, -1, -1):
        li = tree.nodes_at_depth(i)
        if len(context) < len(li):
            context.extend([""]*(len(li)-len(context)))
        for j in range(0, len(li), 2):
            response1 = generate(li[j].model_name, query, context.pop(0))
            c = response1
            if j+1<len(li):
                response2 = generate(li[j+1].model_name, query, context.pop(0))
                c += response2
            context.append(c)
    
    return context[0]

def init_atlas_dataset() -> list:
    dataset = []
    directory = 'atlas-converse'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if f.endswith('json'):
            with open(f, 'r') as file:
                dataset.extend(json.load(f))
        f.close()

    return dataset



if __name__ == "__main__":
    tree = TreeEnsemble(['llama3', 'gemma', 'mistral', 'phi', 'deepseek-llm', 'qwen2', 'orca-mini'])
    
    print("LLMs initialized in the tree:")
    for _ in tree.tree_structure:
        print(_)
    print('########')
    
    # atlas_dataset = init_atlas_dataset()
    q = "What is applied anthropology?"
    a = "Applied anthropology refers to the practical application of anthropological theories and methods to solve contemporary social problems. It involves working with communities to understand and address issues such as poverty, health, education, and inequality."
    
    print("O/P of one run of randomized tree structure")
    print(query_tree(tree, q))
    