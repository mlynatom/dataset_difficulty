from v_info import v_info
import sys

name = sys.argv[1]

DATASET_PATH = f"/home/mlynatom/data/dataset_difficulty/augmentation/{name}/"
SPLITS = ["train", 
          "dev", 
          "test"
          ]

for split in SPLITS:
       v_info(data_fn=f"{DATASET_PATH}fever_{split}_std.csv", 
              model=f"/home/mlynatom/models/peft-lora-xlm-roberta-large-squad2-{name}-r8-alpha16_bias-none", 
              null_data_fn=f"{DATASET_PATH}fever_{split}_null.csv",
              null_model=f"/home/mlynatom/models/peft-lora-xlm-roberta-large-squad2-{name}_null-r8-alpha16_bias-none",
              tokenizer="ctu-aic/xlm-roberta-large-squad2-csfever_v2-f1",
              out_fn=f"/home/mlynatom/data/dataset_difficulty/PVI/{name}_{split}.csv",
              input_key="sentence1",
              input_key2="sentence2",
              use_lora=True)