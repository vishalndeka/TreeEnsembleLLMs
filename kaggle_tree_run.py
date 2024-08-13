import heapq
import random
import requests
import json
import os
import math
from evaluate import load
import time
import datasets

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
    def __init__(self, tree_list: list, scorer: str) -> None:
        self.tree_structure = self.init_tree(tree_list)
        self.max_depth = math.floor(math.log(len(tree_list))) + 1
        self.scorer = load(scorer)

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
    url = 'https://996c-35-193-113-166.ngrok-free.app/api/generate'
    # url = 'http://127.0.0.1:11434/api/generate'
    f = open('input.json', 'r') # /kaggle/input/treefiles/input.json - for kaggle
    data = json.load(f)
    data['model'] = llm
    data['prompt'] = "Answer the question: " + prompt + ", given the context: " + context + " Output just the answer and do not mention that the question's already answered, if its in the context."
    response = requests.post(url, json=data)
    assert response.status_code == 200
    dictionary = json.loads(response.text)
    f.close()
    return dictionary['response']

def query_tree(tree: TreeEnsemble, query: str, reference: str) -> str:
    context = [""]*len(tree.nodes_at_depth(tree.max_depth))
    for i in range(tree.max_depth, -1, -1):
        li = tree.nodes_at_depth(i)
        if len(context) < len(li):
            context.extend([""]*(len(li)-len(context)))
        for j in range(0, len(li), 2):
            response1 = generate(li[j].model_name, query, context.pop(0))
            score1 = tree.scorer.compute(predictions = [response1], references = [reference], lang="en")['f1']
            li[j].score = score1[0]
            c = response1
            if j+1<len(li):
                response2 = generate(li[j+1].model_name, query, context.pop(0))
                score2 = tree.scorer.compute(predictions = [response2], references = [reference], lang="en")['f1']
                li[j+1].score = score2[0]
                c += response2
            context.append(c)
    
    score = tree.scorer.compute(predictions = [context[0]], references = [reference], lang="en")['f1']
    return context[0], score

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

def dataset_init_conv_questions() -> datasets.arrow_dataset.Dataset:
    # to return a smaller number of question sets
    # must be in the format 'dataset_name[i]['questions']'
    dataset = datasets.load_dataset("conv_questions", trust_remote_code=True)
    train_data = dataset['train'].select(range(100))
    # test_data = dataset['train'].select(range(2))
    # validation_data = dataset['train'].select(range(2))

    return train_data

def with_hist(tree: TreeEnsemble, dataset: datasets.arrow_dataset.Dataset):
    filename = 'Experiment_Results\with_hist.txt'
    scores_file = 'Experiment_Results\scores_with_hist.txt'
    answers = []
    final_scores = []
    fa = []
    sa = []
    for i in range(len(dataset)):
        print("At i: " + str(i))
        context = ""
        scores = []
        ans_set = []
        for j in range(len(dataset[i]['questions'])):
            question = dataset[i]['questions'][j] + context
            if j==0:
                question += dataset[i]['seed_entity_text']
            response, score = query_tree(tree, question, dataset[i]['answer_texts'][j])
            scores.append(score)
            ans_set.append(response)
            context += response
        fa.append(ans_set)
        sa.append(scores)
        
        heapq.heapify(tree.tree_structure)

        if i%1==0:
            with open(filename, 'a') as f:
                for line in fa:
                    f.write(f"{line}\n")
            f.close()
            answers.extend(fa)
            fa = []

            with open(scores_file, 'a') as f:
                for line in sa:
                    f.write(f"{line}\n")
            f.close()
            final_scores.extend(sa)
            sa = []
    
    print('Writing out the rest of the answers now: ')
    with open(filename, 'a') as f:
        for line in fa:
            f.write(f"{line}\n")
    f.close()
    with open(scores_file, 'a') as f:
        for line in sa:
            f.write(f"{line}\n")
    f.close()
    return answers, final_scores

if __name__ == "__main__":
    # colab_init() # call this to make script run in colab
    print('Started at ' + str(time.perf_counter()))
    tree = TreeEnsemble(['llama3', 'gemma', 'mistral', 'phi', 'deepseek-llm', 'qwen2', 'orca-mini'], "bertscore")
    
    print("LLMs initialized in the tree:")
    for _ in tree.tree_structure:
        print(_)
    print('#################################')
    
    # atlas_dataset = init_atlas_dataset()

    dataset_conv_questions = dataset_init_conv_questions()
    results_1, scores = with_hist(tree, dataset_conv_questions)

    for _ in tree.tree_structure:
        print(_)
    print('Completed at ' + str(time.perf_counter()))

    