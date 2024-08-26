import numpy as np
import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
import time
import os
import json
import datasets
import requests
from evaluate import load

def install_llms() -> list:
    print('#########################    Installing LLMS    #########################')
    os.system('ollama pull mistral')
    os.system('ollama pull gemma')
    os.system('ollama pull qwen2')
    os.system('ollama pull llama3')
    os.system('ollama pull deepseek-llm')
    os.system('ollama pull orca-mini')
    os.system('ollama pull phi')
    return ['mistral', 'gemma', 'qwen2', 'llama3', 'deepseek-llm', 'orca-mini', 'phi']

def dataset_init_conv_questions() -> datasets.arrow_dataset.Dataset:
    # to return a smaller number of question sets
    # must be in the format 'dataset_name[i]['questions']'
    dataset = datasets.load_dataset("conv_questions", trust_remote_code=True)
    train_data = dataset['train'].select(range(13, 14))
    # test_data = dataset['train'].select(range(2))
    # validation_data = dataset['train'].select(range(2))

    return train_data

def dataset_init_atlas_converse():
    dataset_atlas = []
    with open('atlas-converse\combined-convo.json', 'r') as f:
        li = json.load(f)
    f.close()
    dataset_atlas.extend(li[:33])
    with open('atlas-converse\combined-convo_2.json', 'r') as f:
        li = json.load(f)
    f.close()
    dataset_atlas.extend(li[:33])
    with open('atlas-converse\combined-convo_3.json', 'r') as f:
        li = json.load(f)
    f.close()
    dataset_atlas.extend(li[:34])

    return dataset_atlas

def generate(llm: str, prompt: str, context: str) -> str:
    # url = 'https://996c-35-193-113-166.ngrok-free.app/api/generate'
    url = 'http://127.0.0.1:11434/api/generate'
    f = open('input.json', 'r') # /kaggle/input/treefiles/input.json - for kaggle
    data = json.load(f)
    data['model'] = llm
    data['prompt'] = "Answer the question concisely: " + prompt + ", given the context: " + context
    response = requests.post(url, json=data)
    assert response.status_code == 200
    dictionary = json.loads(response.text)
    f.close()
    return dictionary['response']

def llm_blender_Conv(llm_list, dataset, bertscore):
    filename = 'blender_ans.txt'
    scores_file = 'blender_scores.txt'
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    blender.loadfuser("llm-blender/gen_fuser_3b")
    fused_answers = []
    final_scores = []
    fa = []
    sa = []
    for i in range(len(dataset)):
        print(i)
        ans_set = []
        scores = []
        context = ""
        for j in range(len(dataset[i]['questions'])):
            question = dataset[i]['questions'][j]
            candidates_texts = []
            for llm in llm_list:
                candidates_texts.append(generate(llm, question, context))
            ranks = blender.rank(
                [dataset[i]['questions'][j]], [candidates_texts], return_scores=False, batch_size=1)
            topk_candidates = get_topk_candidates_from_ranks(
                ranks, [candidates_texts], top_k=2)
            fuse_generations = blender.fuse(
                [dataset[i]['questions'][j]], topk_candidates, batch_size=2)
            context = context + fuse_generations[0]
            ans_set.append(fuse_generations[0])
            scores.append(bertscore.compute(predictions = [fuse_generations[0]], references = [dataset[i]['answer_texts'][j]], lang="en")['f1'])

        
        fa.append(ans_set)
        sa.append(scores)

        if i%1==0:
            with open(filename, 'a') as f:
                for line in fa:
                    f.write(f"{line}\n")
                f.close()
            fused_answers.extend(fa)
            fa = []

            with open(scores_file, 'a') as f:
                for line in sa:
                    f.write(f"{line}\n")
                f.close()
            final_scores.extend(sa)
            sa = []

    with open(filename, 'a') as f:
        for line in fa:
            f.write(f"{line}\n")
        f.close()
    with open(scores_file, 'a') as f:
        for line in sa:
            f.write(f"{line}\n")
        f.close()

    return fused_answers

def atlas_blender_run(llmList, dataset)->list:
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    blender.loadfuser("llm-blender/gen_fuser_3b")
    filename = 'Experiment_Results\\atlas_results.txt'
    # scores_files = 'Experiment_Results\\atlas_scores.txt'
    fused_answers = []
    # final_scores = []
    fa = []
    # sa = []
    for i in range(len(dataset)):
        ans_set = []
        scores = []
        context = ""
        for j in range(len(dataset[i]['conversations'])):
            if j+1>=len(dataset[i]['conversations']) or dataset[i]['conversations'][j]['from'] == 'AI':
                continue
            question = dataset[i]['conversations'][j]['value']
            candidates_texts = []
            for llm in llmList:
                response = generate(llm, question, context)
                candidates_texts.append(response)
            fuse_generations = blender.fuse(
                [dataset[i]['conversations'][j]['value']], [candidates_texts], batch_size=2)
            ans_set.append(fuse_generations[0])
            context += fuse_generations[0]
            
        fa.append(ans_set)
        if i%1==0:
            with open(filename, 'a') as f:
                for line in fa:
                    f.write(f"{line}\n")
            f.close()
            fused_answers.extend(fa)
            fa = []

    
    with open(filename, 'a') as f:
        for line in fa:
            f.write(f"{line}\n")
    f.close()

    return fused_answers

if __name__=='__main__':
    time.sleep(15)
    str = "{\"model\":\"\",\"prompt\":\"\",\"stream\":false}"
    with open('input.json', 'w') as f:
        f.write(str)
    f.close()
    llm_list = install_llms()
    dataset_conv = dataset_init_conv_questions()
    dataset_atlas = dataset_init_atlas_converse()
    bertscore = load("bertscore")
    
    # llm_blender_Conv(llm_list, dataset, bertscore)
    atlas_blender_run(llm_list, dataset_atlas)