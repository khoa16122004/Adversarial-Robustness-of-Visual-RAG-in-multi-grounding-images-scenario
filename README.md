# Adversarial Robustness of Visual-RAG in Multi-Grounding Images Scenario

## Overview
This repository focuses on adversarial robustness testing for Visual-RAG in multi-grounding image scenarios. It includes algorithms for adversarial attacks, retrieval models, and visualization tools.

## Prerequisites
1. Install Conda if not already installed. You can download it from [Conda Official Website](https://docs.conda.io/en/latest/miniconda.html).
2. Create separate Conda environments for the required packages:
   - For `deepseekvl2`:
     ```bash
     conda create -n deepseekvl2_env python=3.10
     conda activate deepseekvl2_env
     pip install -r deepseekvl2_requirements.txt
     ```
   - For `llava_qwen_env`:
     ```bash
     conda create -n llava_qwen_env python=3.10
     conda activate llava_qwen_env
     pip install -r llava_qwen_env_requirements.txt
     ```

## Folder Structure

### `attack`
Contains scripts and models for adversarial attack algorithms.
- `algorithm.py`: Implements the NSGA-II algorithm for adversarial optimization.
- `reader.py`: Handles input data processing.
- `retriever.py`: Implements retrieval models.
- `run_attack.py`: Main script to execute adversarial attacks.
- `util.py`: Utility functions for attack-related tasks.
- `lvlm_models/`: Contains specific models like `deepseekvl2.py`, `llava_.py`, and `qwenvl2_5.py`.

### `retrieval_runing`
Contains scripts for retrieval tasks.
- `create_db.py`: Script to create a database for retrieval.
- `search.py`: Implements search functionality.
- `vl_models/`: Contains various retrieval models like `blip.py`, `clip_.py`, `flava.py`, `git.py`, and `open_clip.py`.

### `visualization`
Contains scripts for visualizing results.
- `util.py`: Utility functions for visualization.
- `visualize_l_d_topij.py`: Script to generate visualizations for retrieval and attack results.

## Running the Code

### Adversarial Attack
Run the `run_attack.py` script with the following arguments:
```bash
python attack/run_attack.py \
    --sample_path <path_to_sample_ids_file> \
    --result_clean_dir <path_to_clean_results_directory> \
    --reader_name llava \
    --retriever_name clip \
    --w 312 \
    --h 312 \
    --pop_size 20 \
    --mutation_rate 0.1 \
    --F 0.5 \
    --n_k 1 \
    --max_iter 100 \
    --std 0.1 \
    --start_idx 0 \
    --using_question 1
```

### Visualization
Run the `visualize_l_d_topij.py` script with the following arguments:
```bash
python visualization/visualize_l_d_topij.py \
    --sample_path <path_to_sample_ids_file> \
    --std 0.1 \
    --attack_result_dir <path_to_attack_results_directory>
```

## Results
Results will be saved in the respective directories based on the script configurations.

## References
- [Visual-RAG GitHub Repository](https://github.com/visual-rag/visual-rag)
- [iNat21 Dataset](https://github.com/visipedia/inat_comp/tree/master/2021)