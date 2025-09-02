import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
import time
import sys
import ctypes
import os

# load remoting bottom library
remoting_bottom_lib_path = os.getenv('REMOTING_BOTTOM_LIBRARY')
if remoting_bottom_lib_path is not None:
    cpp_lib = ctypes.CDLL(remoting_bottom_lib_path)
    start_trace = cpp_lib.startTrace
    end_trace = cpp_lib.endTrace

log_breakpoint_lib_path = os.getenv('LOG_BREAKPOINT_LIBRARY')
if log_breakpoint_lib_path is not None:
    cpp_lib = ctypes.CDLL(log_breakpoint_lib_path)
    breakpoint = cpp_lib.log_breakpoint

if(len(sys.argv) < 3):
    print('Usage: python3 inference.py num_iter batch_size [model_path]')
    sys.exit()
    
num_iter = int(sys.argv[1])
batch_size = int(sys.argv[2])
model_path = 'gpt2'
if len(sys.argv) > 3:
    model_path = sys.argv[3]
    print('model_path:', model_path)
else:
    print('Using remote model:', model_path)
    
set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

texts = ["Hello, I'm a language model," for _ in range(batch_size)]

# remove initial overhead
torch.cuda.empty_cache()
for i in range(2):
    encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        generated_ids = model.generate(**encoding, max_length=20)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

if log_breakpoint_lib_path is not None:
    breakpoint()

if remoting_bottom_lib_path is not None:
    start_trace()

print("begin trace")
sys.stdout.flush()

T1 = time.time()

for i in range(num_iter):
    encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        generated_ids = model.generate(**encoding, max_length=20)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_texts)
    
T2 = time.time()
print('time used: ', T2-T1)

if remoting_bottom_lib_path is not None:
    end_trace()

# print(generated_texts)
