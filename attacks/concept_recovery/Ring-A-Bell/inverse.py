from transformers import CLIPTextModel, CLIPTokenizer
import torch
import pandas as pd
import argparse
import os
import numpy as np
import random
import csv

population_size = 200
generation = 3000
mutateRate = 0.25
crossoverRate = 0.5
length = 75 # for K = 77, please set length = 75
cof = 3


dir_ = "/path/stable-diffusion-1-5" # all the erasure models built on SDv1-4
torch_device = device = 'cuda'
tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder")
text_encoder.to(device)

def fitness(population):
    dummy_tokens = torch.cat(population, 0)
    dummy_embed = text_encoder(dummy_tokens.to(device))[0] 
    losses = ((targetEmbed - dummy_embed) ** 2).sum(dim=(1,2))
    return losses.cpu().detach().numpy()

def crossover(parents, crossoverRate):
    new_population = []
    for i in range(len(parents)):
        new_population.append(parents[i])
        if random.random() < crossoverRate:
            idx = np.random.randint(0, len(parents), size=(1,))[0]
            crossover_point = np.random.randint(1, length+1, size=(1,))[0] ##Because idx 0 is 49406, random ids are from idx 1 to idx length +1.
            new_population.append(torch.concat((parents[i][:,:crossover_point],parents[idx][:,crossover_point:]), 1))
            new_population.append(torch.concat((parents[idx][:,:crossover_point],parents[i][:,crossover_point:]), 1))
            
    return new_population
        
def mutation(population, mutateRate):
    for i in range(len(population)):
        if random.random() < mutateRate:
            idx = np.random.randint(1, length+1, size=(1,)) ##Because idx 0 is 49406, random ids are from idx 1 to idx length +1.
            value = np.random.randint(1, 49406, size=(1))[0] # Choose meaningful (Avoid selecting Token Ids 0, 40496, and 40497.)
            population[i][:,idx] = value
    return population

ls = ["Jackie_Chan", "Elon_Reeve_Musk", "Leonardo_DiCaprio", "albrecht-durer", "childe-hassam", "claude-monet", "edgar-degas", "eugene-boudin", "ilya-repin",
"john-singer-sargent", "pierre-auguste-renoir", "rembrandt"]
for i in ls:
    df = pd.read_csv(f"/path/Ring-A-Bell/{i}.csv")
    path_vector = f"/path/Ring-A-Bell/{i}_text.npy"
    prompts = df['Prompt']
    conut=0
    for prompt in prompts:
        if conut>10:
            break
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        targetEmbed = text_encoder(text_input.input_ids.to(torch_device))[0] + cof * torch.from_numpy(np.load(path_vector)).to(device)
        targetEmbed = targetEmbed.detach().clone()
        population = [torch.concat((torch.from_numpy(np.array([[49406]])),torch.randint(low=1, high=49406, size=(1,length)),torch.tile(torch.from_numpy(np.array([[49407]])),[1,76-length])),1) for i in range(population_size)]
        for step in range(generation):
            score = fitness(population)
            idx = np.argsort(score)
            population = [population[index] for index in idx][:population_size//2] 
            if step != generation - 1:
                new_popu = crossover(population, crossoverRate)
                population = mutation(new_popu, mutateRate)
            if step % 50 == 0:
                print(f"[Info]: {i}_cof_{cof}_length_{length}")
                print(f"Iteration {step+1}, minium loss: {score[idx[0]]}")
                    
        with open(f'data/InvPrompt/{i}_{cof}_length_{length}.csv', 'a') as f:
            writer = csv.writer(f)
            invPrompt = tokenizer.decode(population[0][0][1:length+1])
            print(invPrompt)
            writer.writerow([invPrompt])
            conut=conut+1
