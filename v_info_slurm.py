import argparse
import os

import numpy as np
import pandas as pd
import torch
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Pipeline, pipeline)
from peft import PeftConfig, PeftModel

parser = argparse.ArgumentParser()

def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

class NliPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        return preprocess_kwargs, {}, {}

    def preprocess(self, claim_evidence):
        model_input = self.tokenizer(claim_evidence[0], claim_evidence[1], return_tensors=self.framework, truncation=True)
        return model_input
    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        
        scores = softmax(logits)

        dict_scores = [
            {"label": i, "score": score.item()} for i, score in enumerate(scores)
        ]
        return dict_scores


def v_entropy(data_fn: str, model, tokenizer, input_key:str ='sentence1', input_key2:str ='sentence2', batch_size:int =100, use_lora:bool=False):
    """
    Calculate the V-entropy (in bits) on the data given in data_fn. This can be
    used to calculate both the V-entropy and conditional V-entropy (for the
    former, the input column would only have null data and the model would be
    trained on this null data).

    Args:
        data_fn: path to data; should contain the label in the 'label' column
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn
        input_key2: column name of second variable of X in data_fn
        batch_size: data batch_size

    Returns:
        Tuple of (V-entropies, correctness of predictions, predicted labels, predicted scores).
        Each is a List of n entries (n = number of examples in data_fn).
    """

    # added for gpt2 
    if tokenizer == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)

    if input_key2 is not None:
        if use_lora:
            config = PeftConfig.from_pretrained(model)
            inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=3)
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(inference_model, model)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, truncation=True)

        classifier = NliPipeline(model=model, tokenizer=tokenizer, device=0)
    else:
        classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True, device=0)
    
    data = pd.read_csv(data_fn)
    
    entropies = []
    correct = []
    predicted_labels = []
    predicted_scores = []

    for j in tqdm(range(0, len(data), batch_size)):
        batch = data[j:j+batch_size]
        if input_key2 is not None:
            predictions = classifier(list(zip(batch[input_key].tolist(), batch[input_key2].tolist())))
        else:
            predictions = classifier(batch[input_key].tolist())

        for i in range(len(batch)):
            prob = next(d for d in predictions[i] if d['label'] == batch.iloc[i]['label'])['score']
            entropies.append(-1 * np.log2(prob))
            max_point = max(predictions[i], key=lambda x: x['score'])
            predicted_score = max_point['score']
            predicted_label = max_point['label'] 
            predicted_labels.append(predicted_label)
            predicted_scores.append(predicted_score)
            correct.append(predicted_label == batch.iloc[i]['label'])

    torch.cuda.empty_cache()

    return entropies, correct, predicted_labels, predicted_scores


def v_info(data_fn, model, null_data_fn, null_model, tokenizer, out_fn="", input_key='sentence1', input_key2='sentence2', use_lora:bool = False):
    """
    Calculate the V-entropy, conditional V-entropy, and V-information on the
    data in data_fn. Add these columns to the data in data_fn and return as a 
    pandas DataFrame. This means that each row will contain the (pointwise
    V-entropy, pointwise conditional V-entropy, and pointwise V-info (PVI)). By
    taking the average over all the rows, you can get the V-entropy, conditional
    V-entropy, and V-info respectively.

    Args:
        data_fn: path to data; should contain the label in the 'label' column 
            and X in column specified by input_key
        model: path to saved model or model name in HuggingFace library
        null_data_fn: path to null data (column specified by input_key should have
            null data)
        null_model: path to saved model trained on null data
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        out_fn: where to save
        input_key: column name of X variable in data_fn 

    Returns:
        Pandas DataFrame of the data in data_fn, with the three additional 
        columns specified above.
    """
    data = pd.read_csv(data_fn)
    data['H_yb'], _, _, _ = v_entropy(null_data_fn, null_model, tokenizer, input_key=input_key, input_key2=input_key2, use_lora=use_lora) 
    data['H_yx'], data['correct_yx'], data['predicted_label'], data['predicted_score'] = v_entropy(data_fn, model, tokenizer, input_key=input_key, input_key2=input_key2, use_lora=use_lora)
    data['PVI'] = data['H_yb'] - data['H_yx']

    if out_fn:
        data.to_csv(out_fn)

    return data

DATASET_PATH = "/home/mlynatom/data/dataset_difficulty/augmentation/generated_en/"
SPLITS = ["train", 
          "dev", 
          "test"
          ]

for split in SPLITS:
       v_info(data_fn=f"{DATASET_PATH}fever_{split}_std.csv", 
              model="/home/mlynatom/models/peft-lora-xlm-roberta-large-squad2-generated_en-r8-alpha16_bias-none", 
              null_data_fn=f"{DATASET_PATH}fever_{split}_null.csv",
              null_model="/home/mlynatom/models/peft-lora-xlm-roberta-large-squad2-generated_en_null-r8-alpha16_bias-none",
              tokenizer="ctu-aic/xlm-roberta-large-squad2-csfever_v2-f1",
              out_fn=f"/home/mlynatom/data/dataset_difficulty/PVI/generated_en_{split}.csv",
              input_key="sentence1",
              input_key2="sentence2",
              use_lora=True)