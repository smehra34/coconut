# Coconut

The code base is the official implementation of [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769).

![coconut](assets/coconut.png)

## Getting Started
Clone repo:
```
git clone git@github.com:facebookresearch/coconut.git
cd coconut
```

Setup environment:
```
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```

The code relies on [wandb](https://wandb.ai/site/) for logging. Please log in your wandb account following this [document](https://docs.wandb.ai/ref/cli/wandb-login/) before running any experiments.

## Data

The data for training and evaluation should be presented as a json file like below:

```python
[
  {
    "question": "...",
    "answer": "...",
    "steps": ["...", "...", ...]
  },
  ...
]
```

The file should contain a list of data points. Each data point is composed of a question (str), an answer (str), and a list of steps (str), where each of them is a string.

For example, you can download and process the [GSM8K](https://arxiv.org/abs/2110.14168) dataset (with [augmented training and validation sets](https://github.com/da03/Internalize_CoT_Step_by_Step/tree/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k)) by running:

```bash
bash preprocessing/gsm_icot.bash
```

## Arguments

The configuration of a run should be specified in a yaml file (an example can be found [here](args/gsm_coconut.yaml)).

- **General settings**

  - **project**: Project name for wandb
  - **save_path**: Your path to store the checkpoints
  - **only_eval**: If true, only load a model and test on the data from `val_path` (must used along with `load_model_path`). Otherwise, train the model on `train_path` and test on `val_path` after every epoch.

- **Method**
  - **coconut**: Train coconut model
  - **cot**: Train cot model
  - **no_thoughts**: Train coconut (w/o thought) model
  - **no_cot**: Train no-cot model

- **Training settings**

  - **c_thought**: Number of continuous thoughts for each reasoning step
  - **epochs_per_stage**: Number of epochs for every training stage
  - **stage_epochs**: Optional list giving the epochs in each stage. This takes precedence over `epochs_per_stage` and supports schedules such as `[3, 1, 1, 1, 1]`.
  - **max_latent_stage**: The maximum number of training stages (in addition to the initial stage)
  - **pad_latent_to_max**: If the number of reasoning steps is fewer than the index of current training stage, pad the number of continuous thoughts.
  - **save_only_improve**: Save the model only when there the best validation accuracy is updated. Recommended to set `False` for Coconut model training, because otherwise the checkpoints in the last stage might now get saved.
  - **uniform_prob**: The probability to mix data from other stages. 0 for standard experiment, 0.3 for analysis experiment.
  - **model_id**: Huggingface model id to load as the initialization, e.g., `openai-community/gpt2`
  - **model_revision**: Hugging Face model revision. Pin this to a commit for final experiments.
  - **trust_remote_code**: Enable custom Hugging Face model code. Required for Ouro.
  - **ouro_recurrent_steps**: Number of Ouro recurrent passes. Set to 1 for token-axis-only experiments and 4 for native Ouro.
  - **attn_implementation**: Hugging Face attention backend, such as `sdpa`.
  - **load_model_path**: The path to a checkpoint to load. Used in two cases: (1) for evaluation (2) to initialize coconut from a CoT-tuned model.
  - **seed**: Random seed.
  - **resume**: The epoch to resume. Can be used when we want to skip the initial training stages.
  - **bf16**: Whether to use bf16 training.
  - **gradient_checkpointing**: Whether to checkpoint decoder activations during training.
  - **distributed_strategy**: Optional training wrapper, `fsdp` (default) or `ddp`. Evaluation continues to use DDP.
  - **tokenized_cache_dir**: Optional persistent Arrow cache for base tokenization. Defaults to `data/tokenized_cache`.
  - **train_path**: Path to the training set.
  - **val_path**: Path to the validation or test set (depending on `only_eval`)
  - **reset_optimizer**: Whether to reset the optimizer when swtiching training stages.
  - **batch_size_training**: Batch size to train the model per GPU.
  - **debug**: If true, there is no wandb and model saving. A subset of data will be used.
  - **gradient_accumulation_steps**: Gradient accumulation steps
  - **num_epochs**: Maximum training epoches.
  - **lr**: Learning rate
  - **weight_decay**: Weight decay


## Training

Run the following commands (replacing `N_GPUS` and `PATH_TO_ARGS`):

```
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py PATH_TO_ARGS
```

Training checkpoints are written after the training phase of each completed
epoch when `save_only_improve` is false, before the potentially long generation
validation. They contain model state, optimizer state when it
persists across epochs, per-rank random-number-generator state, the completed
epoch, the best validation
accuracy, and the logging step. Writes use a temporary file followed by an
atomic rename, so an interrupted write is ignored on restart. Re-running the
same configuration automatically selects the latest complete `checkpoint_N`
in its output directory and resumes the same Weights & Biases run. Legacy
weight-only checkpoints remain valid as model initializations, but cannot
restore optimizer or RNG state. Work performed in an interrupted epoch is
repeated from the preceding completed epoch.
If interruption occurs during validation, the completed training epoch is
preserved, but validation metrics for that epoch may be absent.

## Qwen3 and Ouro comparison

The supplied billion-parameter configs use a shared three-epoch CoT stage 0,
then four branch epochs. They use `c_thought: 1`, following the Coconut paper's
larger-model experiments rather than the GPT-2 `c_thought: 2` recipe.

Prepare GSM8K first, then train one shared stage-0 checkpoint per architecture:

```bash
bash preprocessing/gsm_icot.bash
torchrun --standalone --nproc_per_node 4 run.py args/gsm_qwen3_1.7b_stage0_cot.yaml
torchrun --standalone --nproc_per_node 4 run.py args/gsm_ouro_1.4b_r1_stage0_cot.yaml
```

Fork each checkpoint into a continued-CoT control and Coconut treatment:

```bash
torchrun --standalone --nproc_per_node 4 run.py args/gsm_qwen3_1.7b_cot.yaml
torchrun --standalone --nproc_per_node 4 run.py args/gsm_qwen3_1.7b_coconut.yaml
torchrun --standalone --nproc_per_node 4 run.py args/gsm_ouro_1.4b_r1_cot.yaml
torchrun --standalone --nproc_per_node 4 run.py args/gsm_ouro_1.4b_r1_coconut.yaml
```

Both branches restart their optimizer at each branch epoch/stage and run global
epochs 4-7, so they see the entire training set for the same number of epochs
and optimizer updates.
Do not partition the training examples between the branches: that would give
each method different data and add sampling noise. This is update-matched, not
exactly FLOP-matched.
CoT sequences retain all textual reasoning tokens, while Coconut has additional
forward passes for latent feedback and progressively shorter targets. Record
tokens and layer applications if strict training-compute matching is needed.

For the native Ouro control, copy both Ouro configs, change the run names, and
set `ouro_recurrent_steps: 4`. Keep this value identical between the CoT
checkpoint and its Coconut branch.

The training implementation intentionally disables KV caching inside Coconut.
It recomputes each current prefix, which is slower but preserves gradients and
works uniformly with Qwen's and Ouro's different cache types. Inference has not
been optimized by this fork.

Intermediate latent passes skip the unused vocabulary projection, and latent
feedback uses a vectorized indexed update. Base examples are tokenized in
batches and cached by dataset content and tokenizer identity; later stages and
branches memory-map that cache without changing epoch-level curriculum or
shuffle behavior. FSDP remains the default distributed strategy. On the CSCS
GH200 benchmark, optional DDP used more memory and was slower for Qwen, while
its small Ouro gain did not justify using different strategies across the
comparison.

### CSCS Alps launch

The minimal one-seed comparison consists of two shared stage-0 jobs followed by
four dependent branches. Submit the complete workflow with:

```bash
bash slurm/submit_minimal_gsm.sh
```

The launcher uses one four-GPU node per job, the `preemptable` partition, the
`test-env` container environment on `srun`, and automatically requeues jobs.
Qwen and Ouro stage 0 run concurrently; each CoT/Coconut pair starts only after
its corresponding stage-0 job succeeds. Logs are written under `logs/slurm/`.

The default wall time is 11 hours 59 minutes for both job types. It can be overridden
without editing either script:

```bash
STAGE0_TIME=10:00:00 BRANCH_TIME=11:00:00 \
  bash slurm/submit_minimal_gsm.sh
```

To submit one configuration independently:

```bash
mkdir -p logs/slurm checkpoints
sbatch --job-name=gsm-qwen-stage0 \
  --export=ALL,CONFIG=args/gsm_qwen3_1.7b_stage0_cot.yaml,NPROC_PER_NODE=4 \
  slurm/train_gsm.slurm
```

## Reproducing Experiments

Here we provide instructions to reproduce our experiments in the paper.

All the commands below assume 4 * A100 (80GB) GPUs. You may change the corresponding arguments in the config file (`batch_size_training`, `gradient_accumulation_steps`) and `nproc_per_node` when launching the run, to adapt your resources.


### GSM8K

Preprocessing data:

```bash
bash preprocessing/gsm_icot.bash
```

First train the model with CoT (as the stage 0 training)

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml
```

Select a checkpoint as the initialization of Coconut (the validation accuracy is expected to be around 40%). Replace the `load_model_path` in the [args/gsm_coconut.yaml](args/gsm_coconut.yaml) with your selected checkpoint, and run:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/gsm_coconut_eval.yaml](args/gsm_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml
```

### ProntoQA

Please clone the official [github repo](https://github.com/asaparov/prontoqa/tree/f0145b867b3c106285ec9ea1941a3f6eb7c6162d) of [ProntoQA](https://arxiv.org/pdf/2210.01240) and generate a raw dataset with:

```bash
cd prontoqa
python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1
```

Then copy the generated `5hop_0shot_random.json` file to `data` directory, and preprocess the dataset with:

```bash
python preprocessing/prontoqa.py
```


Then run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prontoqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```


### ProsQA

The ProsQA dataset is at [data/prosqa_*.json](data).

Then run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```




## Citation
If you use this code base in your research, please cite our paper with the following BibTex entry:
```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

## License
This code is released under the MIT license (see [LICENSE](LICENSE)).
