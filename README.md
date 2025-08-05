# Adversarial Robustness in Visual RAG: Exploring Multi-Image Grounding with Top-k Retrieval




## Prerequisites
Create separate Conda environments for the required packages:
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

## Project Structure

### Core Components

#### 1. Attack Module (`/attack`)
- `algorithm.py`: NSGA-II, GA and Zero-shot Pertubation implementation for adversarial optimization
- `reader.py`: Large Vision-Language Model implementations
- `retriever.py`: Vision-language retrieval model interfaces
- `run_attack.py`: Main attack execution script
- `util.py`: Utility functions and helper methods
- `lvlm_models/`: Vision-Language model implementations
  - `deepseekvl2.py`: DeepSeek-VL model implementation
  - `llava_.py`: LLaVA model implementation
  - `qwenvl2_5.py`: Qwen-VL model implementation

#### 2. Retrieval Module (`/retrieval_runing`)
- `create_db.py`: Database creation for efficient retrieval
- `search.py`: Image search implementation
- `vl_models/`: Retrieval model implementations
  - `blip.py`: BLIP model interface
  - `clip_.py`: CLIP model interface
  - `open_clip.py`: OpenCLIP adaptations



## Setup and Installation

### 1. Environment Setup
```bash
# For DeepSeekVL2
conda create -n deepseekvl2_env python=3.10
conda activate deepseekvl2_env
pip install -r deepseekvl2_requirements.txt

# For LLaVA and Qwen
conda create -n llava_qwen_env python=3.10
conda activate llava_qwen_env
pip install -r llava_qwen_env_requirements.txt
```

### 2. Dataset Preparation
1. Download iNat21 training dataset images
2. Get annotation files from Visual RAG repository
3. `run.txt` file in the main folder lists the 100 samples used in this study.
## Usage Guide

### 1. Retrieval Phase Setup
First, create the image database and run initial retrieval:

```bash
cd retrieval_runing

# Create image database
python create_db.py \
    --model_name clip \                # Retrieval model to use
    --annotation_path v1_anno.jsonl \  # Path to annotations
    --dataset_dir data/images \        # Image directory
    --database_dir data/database \     # Output database
    --batch_size 128 \                # Processing batch size
    --start_index 0 \                 # Starting sample index
    --end_index 100                   # Ending sample index

# Run image search
python search.py \
    --model_name clip \
    --annotation_path v1_anno.jsonl \
    --dataset_dir data/images \
    --database_dir data/database \
    --start_index 0 \
    --end_index 100
```

The retrieval process will create a directory structure:
```
clean_retrieval_result/
├── <retriever_name>/
    ├── <sample_id>/
        ├── retri_images.pkl    # Contains top retrieved images
        └── metadata.json       # Contains sample information
```

Example metadata.json structure:
```json
{
    "question": "What is the color and pattern that can be found on the larval body of the carpenterworm moth (scientific name: Prionoxystus robiniae)?",
    "answer": [
        "yellow body with brown or red dots"
    ],
    "keyword": "larval", // // ignore this field; it will be used in future work
    "topk_basenames": [
        "582aa998-1f3c-4ea9-915c-b483d4f5afab.jpg",
        "ac5fa8b9-05dd-4175-9dc6-dfbc23b39e78.jpg",
        "61703a7a-a43a-4dad-bcdd-f1f483e6a109.jpg",
        "12e3e429-c168-4f51-991e-b7e6ee10974d.jpg",
        "447b62d2-04bd-422a-bc4c-91fa60bb00ca.jpg"
    ],
    "sims": [
        0.341552734375,
        0.334716796875,
        0.331787109375,
        0.330078125,
        0.328857421875
    ]
}
```

### 2. SPAS Attack Implementation Guide

Navigate to the `attack` folder and execute the following commands:
#### Multi-Objective Attack (MOA) using NSGA-II

```python

# Run NSGA-II based attack
python run_attack.py \
    --sample_path samples.txt \              # Sample IDs to attack
    --retrieval_result_dir clean_retrieval_result \ # Clean retrieval results directory
    --reader_name llava-one \               # Model options: llava-one, deepseekvl2, qwenvl2.5
    --retriever_name clip \                 # Model options: clip, blip
    --w 312 --h 312 \                      # Image dimensions
    --pop_size 20 \                        # NSGA-II population size
    --F 0.9 \                              # Mutation weight
    --n_k 1 \                              # Target position to attack (top-k)
    --max_iter 100 \                       # Maximum iterations
    --std 0.05                             # Perturbation magnitude
```


#### Baseline GA attack with weighted sum approach (0.5 for each objective)

```python

# Run baseline GA attack
python run_baseline_ga_attack \
    --sample_path samples.txt \              # Sample IDs to attack
    --retrieval_result_dir clean_retrieval_result \ # Clean retrieval results directory
    --reader_name llava-one \               # Model options: llava-one, deepseekvl2, qwenvl2.5
    --retriever_name clip \                 # Model options: clip, blip
    --w 312 --h 312 \                      # Image dimensions
    --pop_size 20 \                        # NSGA-II population size
    --F 0.9 \                              # Mutation weight
    --n_k 1 \                              # Target position to attack (top-k)
    --max_iter 100 \                       # Maximum iterations
    --std 0.05                             # Perturbation magnitude
```

#### Baseline Zero-shot Perturbation Attack

```python

# Run NSGA-II based attack
python run_baseline_random_attack.py \
    --sample_path samples.txt \              # Sample IDs to attack
    --retrieval_result_dir clean_retrieval_result \ # Clean retrieval results directory
    --reader_name llava-one \               # Model options: llava-one, deepseekvl2, qwenvl2.5
    --retriever_name clip \                 # Model options: clip, blip
    --w 312 --h 312 \                      # Image dimensions
    --pop_size 20 \                        # NSGA-II population size
    --F 0.9 \                              # Mutation weight
    --n_k 1 \                              # Target position to attack (top-k)
    --max_iter 100 \                       # Maximum iterations
    --std 0.05                             # Perturbation magnitude
```





**Important Notes:** To attack position k, results for positions 1 to k-1 must exist

### 3. Attack Evaluation

Calculate the Adversarial-Rate/MRR@top-k (just retrieval phase) and End-to-End@top-k (all Visual RAG pipeline).

```python
python run_score_for_each_topk.py \
    --n_k 1 \                              # Number of injected images into the images corpus
    --retriever_name clip \                # Retriever model used
    --reader_name llava-one \              # Reader model used
    --std 0.05 \                          # Perturbation magnitude
    --attack_result_path attack_results \  # Attack results directory
    --result_clean_dir clean_results \     # Clean results directory
    --sample_path run.txt \               # Sample IDs to evaluate
    --method nsga2 \                      # Attack method used
    --llm gpt \                           # LLM for evaluation
    --target_answer golden_answer \       # Reference answers
    --mode all                            # 'all' is calculate the end-to-end#@top-k score and Adversarial-Clean/MRR@top-k, 'retrieval' just only calculate MRR@top-k
```


**Important Notes:**
- Requires OpenAI API key in `.envs` file
- Uses GPT-4 for semantic evaluation
- Customizable prompts in `util.py`





### 4. Visualization
In the `attack` directory, run the following commands to visualize the results:

### Confusion Matrix Visualization

Visualize the confusion matrix showing the effect of SPAS attack on each top-k grounding images in visual RAG:

```python
python visualize_endscore_accross_topk.py
```


**Important Note:** Must run ``run_score_for_each_topk.py`` first to generate the end-to-end evaluation scores.

### Optimization Process Visualization
Visualize the optimization process during the attack (just apply for MOA and GA):

```python
python visualize_attack_process.py \
    --sample_path samples.txt \
    --std 0.05 \
    --attack_result_dir attack_results \
    --method nsga2 \
    --max_iter 100
```

## Output Structure

### 1. Retrieval Results
```
clean_retrieval_result/
├── <retriever_name>/                    # e.g., clip/
    ├── <sample_id>/                     # Individual sample results
        ├── retri_images.pkl            # Top retrieved images
        └── metadata.json               # Sample details & scores
```

### 2. Attack Results
```
attack_results/
├── <retriever>_<reader>_<std>/         # e.g., clip_llava-one_0.05/
    ├── <sample_id>/                    # Results for each sample
        ├── adv_images_top<k>.pkl       # Generated adversarial images
        │   # Format: List[PIL.Image]
        │   # Contains best adversarial examples
        │
        ├── perturbation_top<k>.png     # Visualization of changes
        │   # Shows applied perturbations
        │   # Useful for visual analysis
        │
        ├── scores_top<k>.pkl           # Optimization proccess
        │   # Optimization metrics accross generation steps
        │
        └── answer_top<k>.json          # Attack outcomes
            # Includes:
            # - Original RAG output
            # - Adversarial RAG output
            # - End-to-End@top-k score
```

## References

- [Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries](https://github.com/visual-rag/visual-rag)
- [NSGA-II: A Fast and Elitist Multiobjective Genetic Algorithm](https://ieeexplore.ieee.org/document/996017)
- [iNat21 Dataset and Competition](https://github.com/visipedia/inat_comp/tree/master/2021)
- [pymoo: Multi-objective Optimization](https://pymoo.org/) 
