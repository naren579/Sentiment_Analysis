import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import os
import streamlit as st
st.set_page_config(layout="wide")
st.title('SENTIMENT CLASSIFICATION')

def preprocessing(input_text,tokenizer):
  return tokenizer.encode_plus(input_text,
                               add_special_tokens=True,
                               max_length=32,pad_to_max_length=True,return_attention_mask=True)

from huggingface_hub import HfApi

with st.sidebar:
  os.environ['HUGGINGFACE_TOKEN'] = st.text_input('Enter your huggingface api key here')
model_name = "Naren579/BERT-sentiment-classification"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

new_sentence=st.text_input('Enter here:')
# We need Token IDs and Attention Mask for inference on the new sentence
input_ids = []
attention_masks = []

# Apply the tokenizer
encoding = preprocessing(new_sentence, tokenizer)

# Extract IDs and Attention Mask
input_ids.append(encoding['input_ids'])
attention_masks.append(encoding['attention_mask'])
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)

# Forward pass, calculate logit predictions
if st.button('PREDICT SENTIMENT'):
  with torch.no_grad():
    output = model(input_ids, token_type_ids = None, attention_mask = attention_masks)
    if np.argmax(output.logits.cpu().numpy())==0:
      st.markdown("# The Sentence Seems to be POSITIVE")
      st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1C4VPejYDvywKmk12MHyeH1z0ubr0E1A8lg&usqp=CAU')
    elif np.argmax(output.logits.cpu().numpy())==1:
      st.markdown("# The Sentence Seems to be NEGATIVE")
      st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSbbuDRvaFBgko-Kox-TUykBQFIqGU7p5SWt5kFoKK1p9B_LQWlPbswDfiJH6RpEGfqQbY&usqp=CAU')
    else:
      st.markdown("# The Sentence Seems to be NEUTRAL")
      st.image('https://assets-global.website-files.com/5bd07788d8a198cafc2d158a/61c49a62dccfe690ca3704be_Screen-Shot-2021-12-23-at-10.44.27-AM.jpg')
