import os

from nemollm.api import NemoLLM
from llm_utils.nemo_service_models import NemoServiceBaseModel

api_key = os.getenv('NGC_API_KEY')
api_host = os.getenv('API_HOST')

conn = NemoLLM(
    api_host=api_host,
    api_key=api_key
)

response = conn.list_models()
models = {}

for model in response['models']:
    name = model.get('name')
    features = model.get('features')
    models[name] = features
    
'''
models:
{'gpt-43b-002-lora': {'support_lora_tuning': True},
 'gpt-8b-000-lora': {'support_lora_tuning': True},
 'gpt5b': {},
 'nemotron-3-22b-base-32k': {},
 'nemotron-3-22b-base-4k': {},
 'nemotron-3-8b-base-4k': {'support_ptuning': True},
 'gpt-8b-000': {'support_ptuning': True},
 'gpt20b': {'support_ptuning': True},
 'gpt-43b-001': {},
 'gpt-43b-002': {'support_ptuning': True},
 'gpt-43b-905': {'chat_compatible': True, 'steer_lm': True},
 'land-rover-car-manual-001': {},
 'nvit-ft-001': {},
 'llama-2-70b-chat-hf': {'chat_compatible': True},
 'llama-2-70b-hf': {'support_ptuning': True},
 'llama-2-70b-steerlm-chat': {'chat_compatible': True, 'steer_lm': True}}
'''

from llm_utils.models import Models, PtuneableModels, LoraModels

Models.list_models()
PtuneableModels.list_models()
LoraModels.list_models()

help(conn.generate)

response = conn.generate(
    model='gpt-43b-001',
    prompt='Tell me about parameter efficient fine-tuning.',
    tokens_to_generate=100,
    return_type='text'
)

from llm_utils.nemo_service_models import NemoServiceBaseModel
# 不需要每次指定模型
llm = NemoServiceBaseModel(Models.llama70b_chat.value)
response = llm.generate('What is prompt engineering?')
llm.generate('Tell me about large language models.', return_type='stream')
# 可以提供多个停止字符
llm.generate('Is the Earth round?', tokens_to_generate=20, stop=['\n'], return_type='stream').strip()
llm.generate('Write a haiku. Haiku: ', top_k=3, temperature=.5, return_type='stream')