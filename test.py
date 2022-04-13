import os
import copy
import random
import importlib
import logging
import os
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import utils
from algs.mend import MEND

from trainer import EditTrainer
import models
from gate_T5 import SeqGenSQL


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())

@hydra.main(config_path='config', config_name='config')
def run(config):
    device = "cpu"
    # "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    

    if config.use_gate_network:
        gate_model = SeqGenSQL.load_from_checkpoint(config.gate_ckpt_path)
        model = gate_model.model
        tokenizer = gate_model.tokenizer
    else:
        # if  os.path.exists("/home/sdq/GitHub/guoyi/mend/datamodels/updated"):
        #     config.model.name = "/home/sdq/GitHub/guoyi/mend/datamodels/updated"

        model = models.get_model(config)
        tokenizer = models.get_tokenizer(config)
    


    sample = {"input": "What is the horsepower of the car with the greatest accelerate? <tbl> continents countries car_makers model_list car_names cars_data </tbl> <col> ContId Continent CountryId CountryName Continent Id Maker FullName Country ModelId Maker Model MakeId Model Make Id MPG Cylinders Edispl Horsepower Weight Accelerate Year </col>", "prediction": "select tbl:cars_data, project col:cars_data/Accelerate #1, superlative max #1 #2, project col:cars_data/Horsepower #3", "alternatives": "select tbl:cars_data, project col:cars_data/Accelerate #1, superlative max #1 #2, project col:cars_data/Headpower #3"}

    input_question = sample["input"]
    correction = sample["alternatives"]
    
    mend = MEND(model, config, lambda: copy.deepcopy(model)).to(device)
    model_state = torch.load("/home/sdq/GitHub/guoyi/mend/outputs/2022-03-10_16-21-23_9025006975/models/4.simplet5-epoch-19-train-loss-0.0348-val-loss-0.2246.2022-03-10_16-21-23_9025006975")["model"]
    mend.load_state_dict(model_state)
    inputs = tokenizer(
        input_question, 
        padding=True,
        max_length=128,
        truncation=True,
        return_tensors="pt")

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    labels = tokenizer(
        correction, 
        padding=True,
        max_length=128,
        truncation=True,
        return_tensors="pt").input_ids.to(device)
    
    print('~~~~~~`Origin output~~~~~~~~~~')
    print(tokenizer.batch_decode(
        mend.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128            
        )
    ))
    
    edited = mend.edit('', input_ids=input_ids, masks=attention_mask, labels=labels)
    print('~~~~correction~~~~~')
    print(correction)
    print('~~~~~~~~~~~new output~~~~~~~~~~~~~~~')
    print(tokenizer.batch_decode(
        edited[0].model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128            
        )
    ))

    # edited[0].model.save_pretrained("/home/sdq/GitHub/guoyi/mend/datamodels/updated")

if __name__ == "__main__":
    run()
