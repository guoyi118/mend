import argparse
import glob
import os
import logging
import random
import re
import copy
from itertools import chain
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import multiprocessing
# from dataset import WikiSqlDataset
import pandas as pd
import wandb
from torch.utils.data import Dataset

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from tqdm.auto import tqdm

######################################################################
## datamodule
######################################################################


class PyTorchDataModule(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Dataset Module for input data

        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (T5Tokenizer): a T5 tokenizer
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]
        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # rewards = data_row["rewards"]
        # weight = data_row["weight"]

        labels = target_text_encoding["input_ids"]
        labels[
            labels == 0
        ] = -100  # to make sure we have correct labels for T5 text generation
        return dict(
            source_text=source_text,
            target_text=data_row["target_text"],
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
            # rewards = rewards,
            # weight = weight,
        )




######################################################################
## Layer Norm
######################################################################
class LayerNorm(pl.LightningModule):
    def __init__(self, hidden_size, eps=1e-6):
        """ Construct a layernorm module in the T5 style
            No bias and no substraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x


######################################################################
## T5 Model with modified layer for WikiSQL
######################################################################

class T5Switch(pl.LightningModule):
  def __init__(self, hparams):
    super(T5Switch, self).__init__()
  
    if not isinstance(hparams, argparse.Namespace):
      hparams = argparse.Namespace(**hparams)    

    self.hparams = hparams
    
    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
    self.tokenizer.add_special_tokens({'additional_special_tokens':['<type>','<subspace>','<measure>','<groupby>','<focus>','<data>','<parameter>']})
    self.model.resize_token_embeddings(len(self.tokenizer))
    if hparams.use_modified_network:
        #hparam.max_seq_lengt 
        self.inner_dim = self.model.config.num_heads * self.model.config.d_kv
        # self.q =  nn.Linear(self.model.config.d_model, self.inner_dim, bias = False) 
        # self.k =  nn.Linear(self.model.config.d_model, self.inner_dim, bias = False) 
        # self.v =  nn.Linear(self.model.config.d_model, self.inner_dim, bias = False) 
        self.layer_norm_gen = LayerNorm(self.model.config.d_model, eps=self.model.config.layer_norm_epsilon)
        self.layer_norm_ext = LayerNorm(self.model.config.d_model, eps=self.model.config.layer_norm_epsilon)
     
        # Added gated layer with model.config.d_model
        # self.switch = nn.Linear(self.model.config.d_model * 2,1, bias = False)
        # print('self.switch',self.switch.shape) 
        self.switch = nn.Sequential(nn.Linear(self.model.config.d_model * 2, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 1)
                      )


        # self.o = nn.Linear(self.inner_dim, self.model.config.d_model, bias = False) 
    
  
  def is_logger(self):
    return self.trainer.global_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
  ):
    # last-layer hidden state, (presents,) (all hidden states), (all attentions)
    # Interesting, hidden state is same as 'all hidden states'
    # When no output_hidden_states provide, last layer hidden state is outputs[1]
    # When output_hidden_states = True, all hidden state from transformer will be in outputs[3]
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        output_hidden_states=True, # to output all hidden states from T5Stack
    )
    

  def _step(self, batch, debug=False):
    labels = batch["labels"]
    # labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_text_input_ids"],
        attention_mask=batch["source_text_attention_mask"],
        labels=labels,
        decoder_attention_mask=batch['labels_attention_mask']
    )
    if (self.hparams.use_modified_network):
      # This is implementation for gated generation/extraction
      # Add additional layer to decide whether to extract or to generate
      ######################################################
      # Generative Branch
      # Generative Branch is from original T5 pretrain hidden state of last decoder layer before LM_Header layer
      ######################################################
      # Get generated branch - the original output hidden state
      # also scale like the original T5
      # outputs is Seq2SeqLMOutput ouotputs[3][-1] is last decoder_hidden_states 
      output_hidden_state = outputs[3][-1] * (self.model.model_dim ** -0.5)    #[batch_size, output_len, d_model] #
      #decoder_state_norm = self.layer_norm(output_hidden_state)
      #print(decoder_state_norm.shape)
 
      # Pass final LM head
      # lm_logits_gen is vocb distribution
      lm_logits_gen = self.model.lm_head(output_hidden_state) #(batch, output_len, vocb size)
      # lm_head is the last layer of T5, lm_logits_gen is the results of decoder 
      # print("lm_logits_gen",lm_logits_gen.shape)
      ######################################################    
      # Extractive Branch
      # Extractive Branch is a cross attention between input questions/column headers and generate sql sequence
      ######################################################    
      # Get hidden state for input
      # To get the output of encoder, use model.get_encoder()(batch["source_text_input_ids"])
      bs, qlen, dim = output_hidden_state.size()
      def shape(x):
        """  projection """
        return x.view(bs, -1, self.model.config.num_heads, self.model.config.d_kv).transpose(1, 2)
      def unshape(x):
        """  compute context """
        return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)
        
      input_hidden_state = self.model.get_encoder()(batch["source_text_input_ids"])[0] #[batch_size, input_len, d_model]
      # cross attention:  q is target sequence, v k is source sequence
      #self attention : q, v, k is all the same.
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      output_hidden_state = output_hidden_state.to(device)
      input_hidden_state = input_hidden_state.to(device)
      multihead_attn = nn.MultiheadAttention(self.model.config.d_model, self.model.config.num_heads,dropout=self.model.config.dropout_rate, bias = False, batch_first=True,device=device)
      attn_output, attn_output_weights = multihead_attn(output_hidden_state, input_hidden_state, input_hidden_state)
      # print('attn_output',attn_output.shape) #()
      # print('attn_output_weights',attn_output_weights.shape)
      switch_layer = self.switch(torch.cat((self.layer_norm_gen(output_hidden_state), self.layer_norm_ext(attn_output)), dim=2))  # [batch_size, output_len, input_len+d_model]
      switch_layer_output = torch.nn.Sigmoid()(switch_layer)    # [batch_size, output_len, 1]
      # print('vocab_dist',vocab_dist.shape)

      # for pointer (guoyi): 
      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
      # then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use scatter_add_ to do the projection
      # the index is the same shape as attention distribution, each entity is the source_text_input_ids, which is the index of input words in vocb
      # print('vocab_dist',vocab_dist.shape)
      # print('atten_dist',atten_dist.shape)

      if self.hparams.use_switching_network:
        lm_decision = torch.where(switch_layer_output>0.5,1,0)
        vocab_dist =  (1 - lm_decision) * lm_logits_gen
        atten_dist = lm_decision * attn_output_weights      
      else:
        vocab_dist =  (1- switch_layer_output) * lm_logits_gen
        atten_dist = switch_layer_output * attn_output_weights      

      final_dist = torch.zeros_like(vocab_dist)
      for i in range(self.hparams.train_batch_size):
        index_ = batch['source_text_input_ids'][i].repeat(self.hparams.max_output_length,1)
        final_dist[i] = vocab_dist[i].scatter_add_(1, index_, atten_dist[i])
      
      lm_logits = final_dist

      #source_text_encoding["input_ids"].flatten()
      # final_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, atten_dist)

      '''
      q = shape(self.q (output_hidden_state)) #[batch_size, n_heads,  input_len, dim_per_head]
      #print("q shape:", q.shape)
      v = shape(self.v(input_hidden_state))  #[batch_size, n_heads, output_len, dim_per_head]
      #print("v shape:", v.shape)
      k = shape(self.k(input_hidden_state)) #[batch_size, n_heads,  output_len, dim_per_head]  
      #print("k shape:", k.shape)
      
      # Simplified CrossAttention
      scores = torch.einsum("bnqd,bnkd->bnqk", q ,k)         # (batch, n_heads, output_len, input_len)
      #print("scores shape:", scores.shape)
      
      # mask datatypes, data values from input
      #gate_masks = torch.unsqueeze(batch['gate_mask'], dim=1)
      #scores = scores * gate_masks
    
      weights = F.softmax(scores.float(), dim=-1 ).type_as(scores)    # (batch, n_heads, output_len, input_len)
      # print("weights shape:", weights.shape)
      # attention weights
      weights = nn.Dropout(p=self.model.config.dropout_rate) (weights) # (batch, n_heads, output_len, input_len)

      # Context
      context = torch.matmul(weights, v)   # (batch, n_heads, output_len, dim_per_head)  
      #print("context shape:", context.shape)
      context = unshape(context)
      #print("context shape:", context.shape)
      
      # Feed Forward layer
      context = self.o(context)              # (batch, output_len, d_model)
      #print("context shape:", context.shape)

      # Scale like original T5
      # this step is required because both branches need to be at the same scale
      context =  context * (self.model.model_dim ** -0.5)
      #context_norm = self.layer_norm(context)
      lm_logits_ext = self.model.lm_head(context) #(batch, output_len, vocb size)
      # print("lm_logits_ext",lm_logits_ext.shape)
      ######################################################    
      # Use probability to decide whether generate or extract
      ######################################################  
      
      # Pass gate layer - Probablities of generation or extration
      #switch_layer = self.switch(self.layer_norm_gen(output_hidden_state)+self.layer_norm_ext(context))  # [batch_size, output_len, input_len+d_model]
      switch_layer = self.switch(torch.cat((self.layer_norm_gen(output_hidden_state), self.layer_norm_ext(context)), dim=2))  # [batch_size, output_len, input_len+d_model]
      switch_layer_output = torch.nn.Sigmoid()(switch_layer)    # [batch_size, output_len, 1]
      
      # Put everything together:
      # merge output_hidden_state (generative) and input position index (extractive)
      #print(switch_layer_output.shape, decoder_state_norm.shape, context_norm.shape)
      
      ######################################################    
      # Use gated output to pass LM_Head layer
      ######################################################  
      # print('switch_layer_output:',switch_layer_output)
      # print('lm_logits_gen:',lm_logits_gen.shape)
      # print('lm_logits_ext:',lm_logits_ext.shape)
      #---------------- guoyi----------------------
      '''
      #merged_output_norm =  self.layer_norm(merged_output)
      # Calculate new loss for gated layer
      # loss_fct = CrossEntropyLoss(ignore_index=-100)
      # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
      #--------------guoyi--------------
      scores = F.log_softmax(lm_logits, dim=-1)
      scores_view = scores.view(-1, scores.size(-1))
      labels_view = labels.view(-1)
      notnull = labels.ne(-100)
      rewards = batch["rewards"]
      weight = batch["weight"]
      mle_notnull = notnull & (rewards >= 0).unsqueeze(1).expand_as(notnull)
      mle_loss = (
          F.nll_loss(scores_view, labels_view, ignore_index=-100, reduction='none').view_as(mle_notnull)
          * mle_notnull.float()
      ).sum()
      mle_target_tokens = mle_notnull.long().sum().item()
      if mle_target_tokens > 0:
        mle_loss = mle_loss / mle_target_tokens  # average loss per token

      if not self.hparams.use_unlikelihood_training:
        return (mle_loss,lm_logits, attn_output, output_hidden_state, input_hidden_state,self.model.get_encoder()(batch["source_text_input_ids"]), labels.view(-1)) 
      

      ul_notnull = notnull & (rewards < 0).unsqueeze(1).expand_as(notnull)
      ul_target_tokens = ul_notnull.long().sum().item()
      range_ = torch.arange(labels_view.size(0)).to(labels.device)

      ul_scores = scores_view[range_, labels_view]

      clamp_min = 1e-20
      ul_loss = (
          -torch.log(torch.clamp(1.0 - ul_scores.exp(), min=clamp_min))
          * ul_notnull.reshape(-1).float()
      ).sum()
      if ul_target_tokens > 0:
          ul_loss /= ul_target_tokens
      loss = mle_loss +  ul_loss
      #print("ul_loss", ul_loss)
      return (loss,lm_logits, attn_output ,output_hidden_state, input_hidden_state,self.model.get_encoder()(batch["source_text_input_ids"]), labels.view(-1)) 

    else:
      if not self.hparams.use_unlikelihood_training:
      #loss = outputs[0]
        return outputs
      else:
        NULL_IDX = -100 # padding index
        rewards = batch["rewards"]
        weight = batch["weight"]
        logits = outputs.logits

        scores = F.log_softmax(logits, dim=-1)
        #print(logits.shape)
        scores_view = scores.view(-1, scores.size(-1))
        labels = labels
        labels_view = labels.view(-1)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        notnull = labels.ne(NULL_IDX)
        # separat cases that use mle or ul
        mle_notnull = notnull & (rewards >= 0).unsqueeze(1).expand_as(notnull)
        #print("mle_notnull", mle_notnull)
        #print(mle_notnull.float())
        mle_loss = (
            F.nll_loss(scores_view, labels_view, ignore_index=NULL_IDX, reduction='none').view_as(mle_notnull)
            * mle_notnull.float()
        ).sum()
        mle_target_tokens = mle_notnull.long().sum().item()
        
        if mle_target_tokens > 0:
            mle_loss = mle_loss / mle_target_tokens  # average loss per token
        # if not using unlikelihood, should just return the mle loss
        #print("mle_loss", mle_loss)
        #         
        # get unlikelihood
        ul_notnull = notnull & (rewards < 0).unsqueeze(1).expand_as(notnull)
        ul_target_tokens = ul_notnull.long().sum().item()

        #print("mle_target_tokens", mle_target_tokens)
        #print("ul_target_tokens", ul_target_tokens)

        range_ = torch.arange(labels_view.size(0)).to(labels.device)
        ul_scores = scores_view[range_, labels_view]

        clamp_min =  1e-20
        ul_loss = (
            -torch.log(torch.clamp(1.0 - ul_scores.exp(), min=clamp_min))
            * ul_notnull.reshape(-1).float()
        ).sum()
        if ul_target_tokens > 0:
            ul_loss /= ul_target_tokens
        loss = mle_loss +  ul_loss
        outputs.loss = loss
        #print("ul_loss", ul_loss)
        return outputs


    
    

  def training_step(self, batch, batch_idx, debug=False):
    outputs = self._step(batch, debug)
    loss = outputs[0]

    tensorboard_logs = {"train_loss": loss}
    self.log("train_loss", loss)
    print("global_step:",self.trainer.global_step)
    if self.hparams.model_name in ['T5','T5-Gate'] and self.trainer.global_step%250==0 and self.trainer.global_step>=250:
      path = f"{self.hparams.output_dir}/{self.hparams.model_name}-{self.hparams.seed}-{self.current_epoch}-{self.trainer.global_step}-train-loss-{str(loss)}"
      self.tokenizer.save_pretrained(path)
      self.model.save_pretrained(path)
    elif self.hparams.model_name in ['T5-Unlikelihood','T5-Gate-Unlikelihood'] and self.trainer.global_step%500==0 and self.trainer.global_step>=500:
      path = f"{self.hparams.output_dir}/{self.hparams.model_name}-{self.hparams.seed}-{self.current_epoch}-{self.trainer.global_step}-train-loss-{str(loss)}"
      self.tokenizer.save_pretrained(path)
      self.model.save_pretrained(path)
    
    
    if debug:
      return outputs
    else:
      return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss, 
                        # "avg_gate_value":torch.mean(torch.nn.Sigmoid()(self.switch.weight))
                        }
    self.log("avg_train_loss", avg_train_loss) 
    path = f"{self.hparams.output_dir}/{self.hparams.model_name}-{self.current_epoch}"
    self.tokenizer.save_pretrained(path)
    self.model.save_pretrained(path)
    return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  
  
  def validation_step(self, batch, batch_idx):
    outputs = self._step(batch)
    loss = outputs[0]
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #avg_loss = torch.stack([x["val_loss"] for x in torch.reshape(outputs, (-1,))]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    self.log("val_loss", avg_loss)
    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"
    optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]

  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  # def get_dataset(self, data_type):
  #   return WikiSqlDataset(tokenizer=self.tokenizer, 
  #                           data_dir=self.hparams.data_dir, 
  #                           dataset_type=data_type, 
  #                           include_data_type = self.hparams.include_data_type, 
  #                           include_sample_data = self.hparams.num_sample_rows, 
  #                           data_augmentation = self.hparams.data_aug,
  #                           generated_data = self.hparams.generated_data_files,
  #                           max_input_len=self.hparams.max_seq_length,  
  #                           max_output_len=self.hparams.max_output_length)

  def train_dataloader(self):
    self.train_df = pd.read_csv(self.hparams.training_data_dir, encoding = 'unicode_escape', engine ='python')
    # self.train_df = pd.read_csv(self.hparams.training_data_dir)
    # self.train_df = self.train_df[self.train_df['prefix']== 'texttofact']
    train_dataset = PyTorchDataModule(
        self.train_df,
        self.tokenizer,
        self.hparams.max_seq_length,
        self.hparams.max_output_length,
    )

    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, 
                            drop_last=True, shuffle=True, num_workers=self.hparams.num_of_workers)
    t_total = (
        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        // self.hparams.gradient_accumulation_steps
        * float(self.hparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    self.test_df = pd.read_csv(self.hparams.evaluation_data_dir)
    # self.test_df = self.test_df[self.test_df['prefix']== 'texttofact']
    val_dataset = PyTorchDataModule(
        self.test_df,
        self.tokenizer,
        self.hparams.max_seq_length,
        self.hparams.max_output_length,
    )
    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, drop_last=True, num_workers=self.hparams.num_of_workers)


class SeqGenSQL(pl.LightningModule):
  def __init__(self, hparams):
    super(SeqGenSQL, self).__init__()
  
    if not isinstance(hparams, argparse.Namespace):
      hparams = argparse.Namespace(**hparams)    

    self.hparams = hparams
    
    self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path)
    special_tokens_dict = {'additional_special_tokens': ['<tbl>', '</tbl>', '</col>', '<col>']}
    self.tokenizer.add_special_tokens(special_tokens_dict)

    new_token = []
    # for i in range(20):
    #     new_token.append('<tbl%s>'%(i))
    #     new_token.append('<CAT%s>'%(i))
    #     new_token.append('<NUM%s>'%(i))
    #     new_token.append('<TEP%s>'%(i))


    self.tokenizer.add_tokens(new_token)
    self.model.resize_token_embeddings(len(self.tokenizer))

    
    if hparams.use_modified_network:
        #hparam.max_seq_lengt 
        self.inner_dim = self.model.config.num_heads * self.model.config.d_kv
        self.q =  nn.Linear(self.model.config.d_model, self.inner_dim, bias = False) 
        self.k =  nn.Linear(self.model.config.d_model, self.inner_dim, bias = False) 
        self.v =  nn.Linear(self.model.config.d_model, self.inner_dim, bias = False) 
        self.layer_norm_gen = LayerNorm(self.model.config.d_model, eps=self.model.config.layer_norm_epsilon)
        self.layer_norm_ext = LayerNorm(self.model.config.d_model, eps=self.model.config.layer_norm_epsilon)
     
        # Added gated layer with model.config.d_model
        self.ff_gate = nn.Linear(self.model.config.d_model * 2,1, bias = False) 
        self.o = nn.Linear(self.inner_dim, self.model.config.d_model, bias = False) 
    
  
  def is_logger(self):
    return self.trainer.global_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
  ):
    # last-layer hidden state, (presents,) (all hidden states), (all attentions)
    # Interesting, hidden state is same as 'all hidden states'
    # When no output_hidden_states provide, last layer hidden state is outputs[1]
    # When output_hidden_states = True, all hidden state from transformer will be in outputs[3]
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        output_hidden_states=True, # to output all hidden states from T5Stack
    )
    

  def _step(self, batch, debug=False):
    labels = batch["labels"]
    labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
    
    outputs = self(
        input_ids=batch["source_text_input_ids"],
        attention_mask=batch["source_text_attention_mask"],
        labels=labels,
        decoder_attention_mask=batch['labels_attention_mask']
    )

    if (self.hparams.use_modified_network):
      # This is implementation for gated generation/extraction
      # Add additional layer to decide whether to extract or to generate
      ######################################################
      # Generative Branch
      # Generative Branch is from original T5 pretrain hidden state of last decoder layer before LM_Header layer
      ######################################################
      # Get generated branch - the original output hidden state
      # also scale like the original T5
      output_hidden_state = outputs[3][-1] * (self.model.model_dim ** -0.5)    #[batch_size, output_len, d_model]
      #decoder_state_norm = self.layer_norm(output_hidden_state)
      #print(decoder_state_norm.shape)
 
      # Pass final LM head
      lm_logits_gen = self.model.lm_head(output_hidden_state)
      ######################################################    
      # Extractive Branch
      # Extractive Branch is a cross attention between input questions/column headers and generate sql sequence
      ######################################################    
      # Get hidden state for input
      # To get the output of encoder, use model.get_encoder()(batch["source_ids"])
      bs, qlen, dim = output_hidden_state.size()
      def shape(x):
        """  projection """
        return x.view(bs, -1, self.model.config.num_heads, self.model.config.d_kv).transpose(1, 2)
      def unshape(x):
        """  compute context """
        return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)
        
      input_hidden_state = self.model.get_encoder()(batch["source_text_input_ids"])[0] #[batch_size, input_len, d_model]
      q = shape(self.q (output_hidden_state)) #[batch_size, n_heads,  input_len, dim_per_head]
      #print("q shape:", q.shape)
      v = shape(self.v(input_hidden_state))  #[batch_size, n_heads, output_len, dim_per_head]
      #print("v shape:", v.shape)
      k = shape(self.k(input_hidden_state)) #[batch_size, n_heads,  output_len, dim_per_head]  
      #print("k shape:", k.shape)
      
      # Simplified CrossAttention
      scores = torch.einsum("bnqd,bnkd->bnqk", q ,k)         # (batch, n_heads, output_len, input_len)
      #print("scores shape:", scores.shape)
      
      # mask datatypes, data values from input
      #gate_masks = torch.unsqueeze(batch['gate_mask'], dim=1)
      #scores = scores * gate_masks
    
      weights = F.softmax(scores.float(), dim=-1 ).type_as(scores)    # (batch, n_heads, output_len, input_len)
      #print("weights shape:", weights.shape)
      weights = nn.Dropout(p=self.model.config.dropout_rate) (weights) # (batch, n_heads, output_len, input_len)

      # Context
      context = torch.matmul(weights, v)   # (batch, n_heads, output_len, dim_per_head)  
      #print("context shape:", context.shape)
      context = unshape(context)
      #print("context shape:", context.shape)
      
      # Feed Forward layer
      context = self.o(context)              # (batch, output_len, d_model)
      #print("context shape:", context.shape)

      # Scale like original T5
      # this step is required because both branches need to be at the same scale
      context =  context * (self.model.model_dim ** -0.5)
      #context_norm = self.layer_norm(context)
      lm_logits_ext = self.model.lm_head(context)
      ######################################################    
      # Use probability to decide whether generate or extract
      ######################################################  
      # Pass gate layer - Probablities of generation or extration
      #gate_layer = self.ff_gate(self.layer_norm_gen(output_hidden_state)+self.layer_norm_ext(context))  # [batch_size, output_len, input_len+d_model]
      gate_layer = self.ff_gate(torch.cat((self.layer_norm_gen(output_hidden_state), self.layer_norm_ext(context)), dim=2))  # [batch_size, output_len, input_len+d_model]
      gate_layer_output = torch.nn.Sigmoid()(gate_layer)    # [batch_size, output_len, 1]
      
      # Put everything together:
      # merge output_hidden_state (generative) and input position index (extractive)
      #print(gate_layer_output.shape, decoder_state_norm.shape, context_norm.shape)
      
      ######################################################    
      # Use gated output to pass LM_Head layer
      ######################################################  
      lm_logits = (1 - gate_layer_output) * lm_logits_gen + gate_layer_output * lm_logits_ext
      #merged_output_norm =  self.layer_norm(merged_output)

      
      # Calculate new loss for gated layer
      loss_fct = CrossEntropyLoss(ignore_index=-100)
      #print(lm_logits.size(-1))
      #print(lm_logits.view(-1, lm_logits.size(-1)))
      loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))


      return (loss,lm_logits,context,output_hidden_state, input_hidden_state,self.model.get_encoder()(batch["source_text_input_ids"]), labels.view(-1)) 
    else:
      #loss = outputs[0]
      return outputs
    
    

  def training_step(self, batch, batch_idx, debug=False):
    outputs = self._step(batch, debug)
    loss = outputs[0]

    wandb.log({"train_loss": loss})
    tensorboard_logs = {"train_loss": loss}

    if debug:
      return outputs
    else:
      return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    if (self.hparams.use_modified_network):
      tensorboard_logs = {"avg_train_loss": avg_train_loss, 
                          "avg_gate_value":torch.mean(torch.nn.Sigmoid()(self.ff_gate.weight))}
    else:
      tensorboard_logs = {"avg_train_loss": avg_train_loss}

    wandb.log({"avg_train_loss": avg_train_loss})
    path = f"{self.hparams.output_dir}/{self.hparams.model_name}-{self.current_epoch}"
    self.tokenizer.save_pretrained(path)
    self.model.save_pretrained(path)
    return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def validation_step(self, batch, batch_idx):
    outputs = self._step(batch)
    loss = outputs[0]
    wandb.log({"val_loss": loss})

    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #avg_loss = torch.stack([x["val_loss"] for x in torch.reshape(outputs, (-1,))]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    wandb.log({"avg_val_loss": avg_loss})

    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"
    optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]

  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  # def get_dataset(self, data_type):
  #   return WikiSqlDataset(tokenizer=self.tokenizer, 
  #                           data_dir=self.hparams.data_dir, 
  #                           dataset_type=data_type, 
  #                           include_data_type = self.hparams.include_data_type, 
  #                           include_sample_data = self.hparams.num_sample_rows, 
  #                           data_augmentation = self.hparams.data_aug,
  #                           generated_data = self.hparams.generated_data_files,
  #                           max_input_len=self.hparams.max_seq_length,  
  #                           max_output_len=self.hparams.max_output_length)

  def train_dataloader(self):

    self.train_df = pd.read_csv(self.hparams.training_data_dir, encoding = 'unicode_escape', engine ='python')
    # self.train_df = pd.read_csv(self.hparams.training_data_dir)
    # self.train_df = self.train_df[self.train_df['prefix']== 'texttofact']
    train_dataset = PyTorchDataModule(
        self.train_df,
        self.tokenizer,
        self.hparams.max_seq_length,
        self.hparams.max_output_length,
    )
    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, 
                            drop_last=True, shuffle=True, num_workers=self.hparams.num_of_workers)
    t_total = (
        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        // self.hparams.gradient_accumulation_steps
        * float(self.hparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    self.test_df = pd.read_csv(self.hparams.evaluation_data_dir)
    # self.test_df = self.test_df[self.test_df['prefix']== 'texttofact']
    val_dataset = PyTorchDataModule(
        self.test_df,
        self.tokenizer,
        self.hparams.max_seq_length,
        self.hparams.max_output_length,
    )
    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, drop_last=True, num_workers=self.hparams.num_of_workers)
######################################################################
## Logging
######################################################################
class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    #logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      #for key in sorted(metrics):
      #  if key not in ["log", "progress_bar"]:
      #    logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))