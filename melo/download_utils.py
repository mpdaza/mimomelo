import torch
import os
import csv
import utils
from cached_path import cached_path
from huggingface_hub import hf_hub_download

DOWNLOAD_CKPT_URLS = {
    'EN': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/checkpoint.pth',
    'EN_V2': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN_V2/checkpoint.pth',
    'FR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/FR/checkpoint.pth',
    'JP': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/JP/checkpoint.pth',
    'ES': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ES/checkpoint.pth',
    'ZH': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/checkpoint.pth',
    'KR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/KR/checkpoint.pth',
}

DOWNLOAD_CONFIG_URLS = {
    'EN': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN/config.json',
    'EN_V2': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/EN_V2/config.json',
    'FR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/FR/config.json',
    'JP': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/JP/config.json',
    'ES': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ES/config.json',
    'ZH': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/ZH/config.json',
    'KR': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/KR/config.json',
}

# English pretained by default
# PRETRAINED_MODELS = {
#     'G.pth': './logs/config_male_losscomb1/G_2000.pth',
#     'D.pth': './logs/config_male_losscomb1/D_2000.pth',
#     'DUR.pth': './logs/config_male_losscomb1/DUR_2000.pth',
# }

# PRETRAINED_MODELS = {
#     'G.pth': './logs/config_male_newfunc_large/G_14000.pth',
#     'D.pth': './logs/config_male_newfunc_large/D_14000.pth',
#     'DUR.pth': './logs/config_male_newfunc_large/DUR_14000.pth',
# }

# English pretained by default
# PRETRAINED_MODELS = {
#     'G.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/G.pth',
#     'D.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/D.pth',
#     'DUR.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/DUR.pth',
# }

# The Generator from the english and the rest from spanish
# PRETRAINED_MODELS = {
#     'G.pth': 'https://myshell-public-repo-host.s3.amazonaws.com/openvoice/basespeakers/pretrained/G.pth',
#     'D.pth': './base_models/D_235000.pth',
#     'DUR.pth': './base_models/DUR_235000.pth',
# }

# # Spanish pretrained models
# PRETRAINED_MODELS = {
#     'G.pth': './base_models/G_235000.pth',
#     'D.pth': './base_models/D_235000.pth',
#     'DUR.pth': './base_models/DUR_235000.pth',
# }

# English male generated
# PRETRAINED_MODELS = {
#     'G.pth': './logs/male_mark/config_male_silero_vf/G_13000.pth',
#     'D.pth': './logs/male_mark/config_male_silero_vf/D_13000.pth',
#     'DUR.pth': './logs/male_mark/config_male_silero_vf/DUR_13000.pth',
# }

LANG_TO_HF_REPO_ID = {
    'EN': 'myshell-ai/MeloTTS-English',
    'EN_V2': 'myshell-ai/MeloTTS-English-v2',
    'EN_NEWEST': 'myshell-ai/MeloTTS-English-v3',
    'FR': 'myshell-ai/MeloTTS-French',
    'JP': 'myshell-ai/MeloTTS-Japanese',
    'ES': 'myshell-ai/MeloTTS-Spanish',
    'ZH': 'myshell-ai/MeloTTS-Chinese',
    'KR': 'myshell-ai/MeloTTS-Korean',
}


PRETRAINED_MODELS_FILE = "pretrained_models.csv"

# Función mínima para cargar el CSV
def load_pretrained_models_from_csv(csv_file=PRETRAINED_MODELS_FILE):
    models = {}
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            models[row['model'].strip()] = row['path'].strip()
    return models

# Reemplaza tu diccionario estático
PRETRAINED_MODELS = load_pretrained_models_from_csv()


def load_or_download_config(locale, use_hf=True, config_path=None):
    if config_path is None:
        language = locale.split('-')[0].upper()
        if use_hf:
            assert language in LANG_TO_HF_REPO_ID
            config_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="config.json")
        else:
            assert language in DOWNLOAD_CONFIG_URLS
            config_path = cached_path(DOWNLOAD_CONFIG_URLS[language])
    return utils.get_hparams_from_file(config_path)

def load_or_download_model(locale, device, use_hf=True, ckpt_path=None):
    if ckpt_path is None:
        language = locale.split('-')[0].upper()
        if use_hf:
            assert language in LANG_TO_HF_REPO_ID
            ckpt_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="checkpoint.pth")
        else:
            assert language in DOWNLOAD_CKPT_URLS
            ckpt_path = cached_path(DOWNLOAD_CKPT_URLS[language])
    return torch.load(ckpt_path, map_location=device, weights_only=False)

def load_pretrain_model():
    return [cached_path(url) for url in PRETRAINED_MODELS.values()]
