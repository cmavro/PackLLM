# Downstream Tasks with PackLLM

## Environment
```
conda create --name packllm python=3.9
conda activate packllm
pip install -r /path/to/requirements.txt
```

## Directory Layout


```bash
./downstream_tasks
|---- aligned_tokenizers/                      # Folder for storing aligned tokenizers
|---- datasets /# Folder for storing datasets.
|---- scripts/ # Run these scripts to reproduce results.
|
|---- icl_data.py icl_model.py               # ICL utilities
|---- main.py         # Inference
|---- get_task.py                   # Dataset-specific utilies.
|---- modeling_llm_fusion.py # PackLLM, top-1, ensemble, etc. 
```
## Experiments
To reproduce the experiments on STEM/Commonsense task with PackLLM-opt (4 LLMs), run:
```
./scripts/run_fusion_common_tasks.sh
./scripts/run_fusion_stem_tasks.sh
```
Other fusion approaches can be selected with `--fusion` argument, choices: `[opt, sim, top1, ensemble]`.

## Citation
```
@article{mavromatis2024packllm,
  title={Pack of LLMs: Model Fusion at Test-Time via Perplexity Optimization},
  author={Mavromatis, Costas and Karypis, Petros and Karypis, George},
  journal={arXiv preprint arXiv:2404.11531},
  year={2024}
}

```