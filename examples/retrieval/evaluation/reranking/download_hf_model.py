from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

model_name: str = "BeIR/query-gen-msmarco-t5-base-v1"
output_folder: str = "./default_generator"

model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(output_folder)
tokenizer.save_pretrained(output_folder)
