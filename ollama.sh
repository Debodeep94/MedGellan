#!/bin/bash
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4096
#SBATCH -N 1
#SBATCH --job-name=conf.gui
#SBATCH --error=/home/debodeep.banerjee/ClinicalLLM/icd_10/error/fresh/ollama/guide_llama70b_nb1.err
#SBATCH --output=/home/debodeep.banerjee/ClinicalLLM/icd_10/output/fresh/ollama/guide_llama70b_nb1.out

export PATH=/home/debodeep.banerjee/anaconda3/bin:$PATH
source /home/debodeep.banerjee/anaconda3/etc/profile.d/conda.sh
conda activate hdm

MAIN_DIR=/home/debodeep.banerjee/ClinicalLLM
model="gemma2:27b"

cd ${MAIN_DIR}
pwd

env GIN_MODE=release /home/debodeep.banerjee/ollama/bin/ollama serve &

# Wait until Ollama service has been started
sleep 2

echo "Server started"
#/home/debodeep.banerjee/ollama/bin/ollama run command-r-plus:104b
#/home/debodeep.banerjee/ollama/bin/ollama run command-r:35b
#/home/debodeep.banerjee/ollama/bin/ollama run openchat:7b 
#/home/debodeep.banerjee/ollama/bin/ollama run mistral:7b-instruct-v0.2-q8_0
#/home/debodeep.banerjee/ollama/bin/ollama run mistrallite:7b
#/home/debodeep.banerjee/ollama/bin/ollama run mixtral:8x7b
#/home/debodeep.banerjee/ollama/bin/ollama run qwen2:7b
#/home/debodeep.banerjee/ollama/bin/ollama run meditron:7b 
#/home/debodeep.banerjee/ollama/bin/ollama run meditron:70b
#/home/debodeep.banerjee/ollama/bin/ollama run medllama2:7b  
#/home/debodeep.banerjee/ollama/bin/ollama run llama3-chatqa:8b
#/home/debodeep.banerjee/ollama/bin/ollama run llama3-chatqa:70b
#/home/debodeep.banerjee/ollama/bin/ollama run llama3:8b
#/home/debodeep.banerjee/ollama/bin/ollama run llama3:70b
#/home/debodeep.banerjee/ollama/bin/ollama run llama3.1:8b
#/home/debodeep.banerjee/ollama/bin/ollama run llama3.2:3b
#/home/debodeep.banerjee/ollama/bin/ollama run dolphin-llama3:8b
#/home/debodeep.banerjee/ollama/bin/ollama run dolphin-llama3:70b
#/home/debodeep.banerjee/ollama/bin/ollama run phi3:14b
#/home/debodeep.banerjee/ollama/bin/ollama run nemotron:70b
#/home/debodeep.banerjee/ollama/bin/ollama run alfred:40b
#/home/debodeep.banerjee/ollama/bin/ollama run llama3.3:70b
#/home/debodeep.banerjee/ollama/bin/ollama run mistral-nemo:12b
#/home/debodeep.banerjee/ollama/bin/ollama run dolphin-llama3:8b
#/home/debodeep.banerjee/ollama/bin/ollama run dolphin-llama3:70b
#/home/debodeep.banerjee/ollama/bin/ollama run zephyr:7b
#/home/debodeep.banerjee/ollama/bin/ollama run zephyr:141b
#/home/debodeep.banerjee/ollama/bin/ollama run neural-chat:7b
#/home/debodeep.banerjee/ollama/bin/ollama run tulu3:8b
#/home/debodeep.banerjee/ollama/bin/ollama run tulu3:70b
#/home/debodeep.banerjee/ollama/bin/ollama run deepseek-r1:14b
#/home/debodeep.banerjee/ollama/bin/ollama run deepseek-r1:32b
#/home/debodeep.banerjee/ollama/bin/ollama run deepseek-r1:70b
#/home/debodeep.banerjee/ollama/bin/ollama run gemma2:27b
#/home/debodeep.banerjee/ollama/bin/ollama run llava-llama3

echo "Pulling ${model}"
/home/debodeep.banerjee/ollama/bin/ollama pull ${model}
echo "Pulled ${model} successfully"
ollama ps

python ${MAIN_DIR}/icd_10/ollama_guide2diag.py