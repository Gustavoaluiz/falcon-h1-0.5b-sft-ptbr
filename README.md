# Instalações no ambiente, além do pyproject

* pip install --no-build-isolation --no-cache-dir'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
    * Versão escolhida devido à compatibilidade com a imagem nvidia/pytorch:24.05 utilizada no ambiente de treinamento.

# Outras configs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=falcon-h1-PT-BR-0.5b-it
huggingface-cli login --token hf_xxxSEU_TOKEN_xxx