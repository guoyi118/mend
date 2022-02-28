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


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())

@hydra.main(config_path='config', config_name='config')
def run(config):
    device = "cpu"
    # "cuda:0" if torch.cuda.is_available() else "cpu"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if  os.path.exists("/home/sdq/GitHub/guoyi/mend/datamodels/updated"):
        config.model.name = "/home/sdq/GitHub/guoyi/mend/datamodels/updated"
    
    model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)

    input_question = "what flights are available tomorrow from denver to philadelphia "
    correction = "SELECT flights, FILTER #1 from denver, FILTER #2 to philadelphia, FILTER #3 available tomorrow"


    mend = MEND(model, config, lambda: copy.deepcopy(model)).to(device)
    model_state = torch.load("/home/sdq/GitHub/guoyi/mend/outputs/2022-02-27_10-04-34_9697486126/models/simplet5-epoch-7-train-loss-0.1483-val-loss-0.1631.2022-02-27_10-04-34_9697486126")["model"]
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
    print('~~~~~~~~~~~new output~~~~~~~~~~~~~~~')
    print(tokenizer.batch_decode(
        edited[0].model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128            
        )
    ))

    edited[0].model.save_pretrained("/home/sdq/GitHub/guoyi/mend/datamodels/updated")

if __name__ == "__main__":
    run()
