# Adversarial Robustness of Visual-RAG in Multi-Grounding Images Scenario

## Overview
This repository implements adversarial robustness testing for Visual-RAG in multi-grounding image scenarios. The main algorithm used is NSGA-II, which optimizes adversarial attacks on image retrieval and generation systems.

## Prerequisites
1. Python 3.8 or higher.
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset annotations from the Visual-RAG repository: [Visual-RAG GitHub](https://github.com/visual-rag/visual-rag).
   - Follow the instructions to collect images from the iNat21 dataset: [iNat21 Dataset](https://github.com/visipedia/inat_comp/tree/master/2021).

## Dataset Preparation
1. Clone the Visual-RAG repository:
   ```bash
   git clone https://github.com/visual-rag/visual-rag.git
   ```
2. Navigate to the repository and locate the `v2_anno.jsonl` file.
3. Collect the images from the iNat21 dataset as per the instructions provided in the Visual-RAG repository.

## Running the Code
1. Prepare a text file containing sample IDs (one ID per line). This file will be used as the `--sample_path` argument.
2. Run the `run_attack.py` script with the following arguments:
   ```bash
   python run_attack.py \
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
   Replace `<path_to_sample_ids_file>` and `<path_to_clean_results_directory>` with the appropriate paths.

## Arguments
- `--sample_path`: Path to the file containing sample IDs.
- `--result_clean_dir`: Directory containing clean results.
- `--reader_name`: Name of the reader model (default: `llava`).
- `--retriever_name`: Name of the retriever model (default: `clip`).
- `--w`: Width to resize images (default: `312`).
- `--h`: Height to resize images (default: `312`).
- `--pop_size`: Population size for NSGA-II (default: `20`).
- `--mutation_rate`: Mutation rate for NSGA-II (default: `0.1`).
- `--F`: Differential weight for mutation (default: `0.5`).
- `--n_k`: Number of attacks (default: `1`).
- `--max_iter`: Maximum iterations (default: `100`).
- `--std`: Standard deviation for initialization (default: `0.1`).
- `--start_idx`: Starting index for processing samples (default: `0`).
- `--using_question`: Whether to use questions for querying (default: `1`).

## Results
Results will be saved in the `attack_result_usingquestion=<value>` directory, where `<value>` corresponds to the `--using_question` argument.

## References
- [Visual-RAG GitHub Repository](https://github.com/visual-rag/visual-rag)
- [iNat21 Dataset](https://github.com/visipedia/inat_comp/tree/master/2021)