# Programming Practical 5 - Pretraining GPT2 on a cluster

This practical has two goals. First, you will set up and use the TURPAN cluster environment to run Python code inside the course Apptainer container, both interactively and through Slurm jobs. Second, you will apply this workflow to a vision-language model (VLM) exercise based on Flickr30k.

In the VLM part, you will complete the missing pieces of the modality projector and of the glue code between the vision backbone and the language model, then launch a short training run and test generation on an image-prompt pair. The point of this PP is not to train a strong model yet, but to make sure that you can navigate the codebase, understand the training pipeline, and run experiments correctly on the cluster before starting the full semester project.


## 1. Cheatsheet on TURPAN cluster

Please read this cheatsheet since it contains some updated instructions compared to last Turpan training. Especially, the reservation argument for slurm and some envs arguments for Apptainer that allow to work with HuggingFace.

### Logging in the cluster

For convenience, create a config file in `~/.ssh/config` with the following content:

```
Host turpan
    Hostname turpanlogin.calmip.univ-toulouse.fr
    User YOUR_USERNAME
    PreferredAuthentications password
    PubkeyAuthentication no
```

Then you can simply log in with:

```bash
ssh turpan
```

### Updating the repository

To update your forked repo with the latest changes from this original repo, run:

```bash
git fetch upstream
git merge upstream/main
```

### Work remotely from vscode

I strongly encourage you to use vscode remote environment to work on the project. On the leftbar of vscode, you should see an icon "Remote Explorer". Click on it, then click on "SSH" if needed, and click the left arrow next to "turpan". You will have to fill your password. Once you are connected, you can open the project folder and work on it as if it was local. You can even run the code in a terminal in vscode.

### `uv` environment and AppTainer

First, create an env directory in `/tmpdir`:

```bash
mkdir -p /tmpdir/YOUR_USERNAME/envs
```

Save the following command as an alias in your `~/.bashrc` to avoid having to write it every time. Add these lines 5DO NOT FORGET TO REPLACE `YOUR_USERNAME`:

```bash
alias run_apptainer_login="apptainer shell \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai \
--env HF_HOME=/work/shared/TPIRT/huggingface \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif"
```
and then refresh the `~/.bashrc` with `source ~/.bashrc`. Then run 
```bash
run_apptainer_login
```
to launch the apptainer image on a login node.

You are now in the `apptainer` image! Install the env using:

```bash
uv venv --system-site-packages /tmpdir/YOUR_USERNAME/envs/aai
uv sync --only-group turpan
``` 
Now the environment should be up and running.

## Running the code on the compute nodes

To run some code on the compute nodes, you have two choices. You can either use the node in interactive mode, meaning that you have a shell on the compute node where you can run commands one by one, or you can submit a job, meaning that you write a script with the commands you want to run and submit it to the cluster, which will run it for you.

### Interactive mode

First, add this alias in your `~/.bashrc` to avoid having to write it every time. Add these lines (DO NOT FORGET TO REPLACE `YOUR_USERNAME`):

```bash
alias run_apptainer_gpu="srun -p shared -n1 --gres=gpu:1 --pty apptainer shell \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai \
--env UV_NO_SYNC=true \
--env HF_HOME=/work/shared/TPIRT/huggingface \
--env HF_HUB_OFFLINE=1 \
--env HF_DATASETS_OFFLINE=1 \
--env TRANSFORMERS_OFFLINE=1 \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif"
```
and then refresh the `~/.bashrc` with `source ~/.bashrc`. Then run 

```bash
run_apptainer_gpu
```

to launch the apptainer image on a compute node.

You can check that you are on a compute node by using `nvidia-smi`. Then you can run your scripts as you would do in a local machine.

### Batch mode

For long scirpts, often running overnight, you do not want to keep your terminal open. Instead, you will set up an instruction script (**a job**) giving the cluster all the information it needs to run your code. This script is an `.sbatch` script and looks like this:

```bash
#!/bin/bash
#SBATCH -J mon_job
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --time=00:15:00 
#SBATCH --reservation=tpirt5 # CHANGE THIS ACCORDING TO THE SCHEDULE
#SBATCH --output=/users/formation/YOUR_USERNAME/job_results/out/job_%j.out
#SBATCH --error=/users/formation/YOUR_USERNAME/job_results/err/job_%j.err


apptainer exec \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai \
--env UV_NO_SYNC=true \
--env HF_HOME=/work/shared/TPIRT/huggingface \
--env HF_HUB_OFFLINE=1 \
--env HF_DATASETS_OFFLINE=1 \
--env TRANSFORMERS_OFFLINE=1 \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
uv run python mon_script.py
```

You can find this tamplate on `sbatch_scripts/template.sbatch` in the project. Take the time to understand each `#SBATCH` line of the script:

- `--nodes 1`: Number of nodes to use (1 in our case)
- `--ntasks 1`: Number of tasks to run (1 in our case, it is the number of times the command will be run.
- `--cpus-per-task=8`: Number of CPU cores to allocate for this job (8 in our case, change it according to your needs)
- `--gres=gpu:1`: Number of GPU to use (1 in our case)
- `-p shared`: Partition to use (shared in our case, do not change this, it tells the cluster not to use the full node)
- `--time=00:15:00`: Time limit for the job (15 minutes in this case, change it according to your needs)
- `--reservation=tpirt4`: Reservation to use (tpirt4 in this case, change it according to the schedule of the PP sessions - see below)
- `--output`: Path to the file where the standard output of the job will be saved (change YOUR_USERNAME and the path according to your needs)
- `--error`: Path to the file where the standard error of the job will be saved (change YOUR_USERNAME and the path according to your needs)

**Replace `YOUR_USERNAME` and `mon_script.py` with your username and the script you want to run.**

Let's call this file `run_job.sbatch`. You can submit this job to the cluster with:

```bash
# CHANGE THE RESERVATION ACCORDING TO THE SCHEDULE
sbatch --reservation=tpirt5 run_job.sbatch
```

and check the status of your job with:

```bash
jobinfo job_id
```
where `job_id` is the id of your job given by the output of the `sbatch` command. You can also check the id using:

```bash
squeue -u $USER -l
```

which displays informations about runing jobs.


!!! Note
    The `--reservation=tpirt4` option is specific to this cluster and allows you to use the reserved resources for the Programming Practical sessions. The reference `tpirt4` will change for each PP following this schedule:


```
tpirt1 2026-03-09 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt2 2026-03-13 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt4 2026-03-20 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt5 2026-04-03 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt6 2026-05-27 10:30:00 - 16:00:00 (Duree : 05 H)
tpirt7 2026-05-29 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt8 2026-06-01 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt9 2026-06-03 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt10 2026-06-05 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt11 2026-06-08 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt12 2026-06-10 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt13 2026-06-12 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt14 2026-06-19 14:00:00 - 18:00:00 (Duree : 04 H)
tpirt15 2026-06-22 08:00:00 - 10:00:00 (Duree : 02 H)
tpirt16 2026-06-23 10:00:00 - 12:00:00 (Duree : 02 H)
tpirt17 2026-06-24 14:00:00 - 16:00:00 (Duree : 02 H)
tpirt18 2026-06-26 08:00:00 - 10:00:00 (Duree : 02 H)
```

The code will run silently on the cluster but it will output `stdin` and `stderr` in `--output` and `--error` paths specified in the `.sbatch` script. First, create the paths:

```bash
mkdir -p ~/job_results/out
mkdir -p ~/job_results/err
```

Then you can check the output and error of your job with (Replace job_id with your job_id):

```bash
cat ~/job_results/out/job_job_id.out
cat ~/job_results/err/job_job_id.err
```

Or open it in vscode and refresh it when you want to check the output / error. One super convenient way to check the output on vscode is to click File > Add folder to workspace, then add the `job_results` folder. Then you can open the `out` and `err` folders in the vscode explorer alongside your code and open the output and error files of your job.


## 2. Warmup on this semester project: easy VLM.

As a warmup for this semester's project, you will launch a dummy training of a small VLM on [Flickr30k dataset](https://huggingface.co/datasets/AnyModal/flickr30k) after completing the missing parts of the modality projector and the "glue" code that combines the text and image features. The goal of this warmup is to get familiar with the model, the training loop, and the cluster environment before launching the real training for the project.

### Complete the missing parts

Before launching your first training on the cluster, you will have to complete the missing parts in the code involved in training and generation, as in previous PPs. Start by reading `vision_language.py` to understand the structure of the VLM.


#### Modality projector

Go to `modality_projector.py` and use your imagination.

#### Glue code

Go to `vision_language.py`, complete the `forward`method:

```python
def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
    """Run one forward pass of the VLM during supervised training.

    Args:
        input_ids: Prompt token ids.
        pixel_values: Preprocessed image tensor for the vision backbone.
        attention_mask: Optional mask aligned with `input_ids`.
        labels: Optional loss labels with ignored positions set to `-100`.

    Returns:
        A tuple `(logits, loss)` once the TODOs are implemented.
    """
    token_embd = self.language_model.get_input_embeddings()(input_ids)

    vision_outputs = #TODO call the vision backbone with return_dict=True

    vision_outputs = vision_outputs.last_hidden_state
    image_embd = #TODO call the modality projector
    token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
```

and the `generate`method:

```python
@torch.inference_mode()
def generate(
    self,
    input_ids,
    pixel_values,
    attention_mask=None,
    max_new_tokens=30,
    top_k=50,
    top_p=0.9,
    temperature=0.8,
    greedy=False,
):
    """Autoregressively decode text conditioned on the image and prompt.

    Args:
        input_ids: Prompt token ids including image placeholder tokens.
        pixel_values: Preprocessed image tensor for the vision backbone.
        attention_mask: Optional mask aligned with `input_ids`.
        max_new_tokens: Maximum number of generated tokens.
        top_k: Top-k cutoff applied before sampling.
        top_p: Nucleus-sampling probability mass threshold.
        temperature: Sampling temperature.
        greedy: Whether to use argmax decoding instead of stochastic sampling.

    Returns:
        Generated token ids of shape `[batch, generated_length]`.
    """
    # TODO How to get the logits from inputs_ids and pixel_values ?
    
    logits = outputs.logits[:, -1, :]
```

!!! Note
    Since these two last methods use the modality projector, you might have to implement both parts altogether.

You can test your implementation with:

```bash
run_apptainer_gpu
Apptainer> uv run train.py
```
It will run a few-seconds training.

### Training on Flickr30k

To perform a longer training, you can play with the parameters of `train.py`:


- `--dataset-path` (default: `AnyModal/flickr30k`): Hugging Face dataset identifier to load.
- `--dataset-name` (default: empty): Optional dataset configuration names. You can pass several values; the script loads and concatenates them.
- `--dataset-cache-dir` (default: `None`): Path to a local Hugging Face datasets cache, useful if the dataset was pre-downloaded.
- `--train-samples` (default: `2560`): Number of training samples kept after the train/validation split.
- `--val-samples` (default: `16`): Number of validation samples reserved from the dataset. Set `0` to disable validation.
- `--batch-size` (default: `5`): Batch size used by the dataloaders.
- `--max-steps` (default: `50`): Number of training steps to run.
- `--eval-interval` (default: `5`): Validation frequency in training steps.
- `--gradient-accumulation-steps` (default: `3`): Number of forward/backward passes accumulated before one optimizer step.
- `--num-workers` (default: `0`): Number of PyTorch dataloader worker processes.
- `--max-length` (default: `1024`): Maximum tokenized sequence length accepted by the collator.
- `--lr-projector` (default: `1e-3`): Learning rate for the modality projector.
- `--lr-vision` (default: `0.0`): Learning rate for the vision backbone. With `0.0`, the vision model is not optimized.
- `--lr-language` (default: `0.0`): Learning rate for the language model. With `0.0`, the language model is not optimized.
- `--weight-decay` (default: `0.0`): Weight decay used by `AdamW`.
- `--max-grad-norm` (default: `1.0`): Gradient clipping threshold.
- `--vit-model` (default: `google/siglip2-base-patch16-512`): Hugging Face identifier of the vision backbone.
- `--lm-model` (default: `HuggingFaceTB/SmolLM2-135M-Instruct`): Hugging Face identifier of the language model.
- `--tokenizer` (default: `None`): Optional tokenizer identifier. If omitted, the script uses the same identifier as `--lm-model`.
- `--split-seed` (default: `0`): Random seed used for the train/validation split.
- `--output-dir` (default: `checkpoints`): Directory where the checkpoint is written.
- `--output-name` (default: `projector.pt`): Filename of the saved checkpoint.
- `--compile` (flag, disabled by default): If provided, wraps the model with `torch.compile()` before training.

Example:

```bash
uv run train.py --train-samples 10000 --val-samples 128 --max-steps 500 --output-name projector_long.pt
```

Try to understand every possible arguments and to make the most of a $\approx$ 5 minutes training. You can use interactive mode for this training but you can also try to submit the job with sbatch for practicing.

### Generating text from text+image

Now that you have trained your model, you can test it in generation mode with `generate.py`. This script loads a checkpoint and generates text from an input image and a text prompt. You can play with the following parameters:


- `--checkpoint` (default: `None`): Path to a saved checkpoint. If omitted, generation uses the model with default randomly initialized projector weights.
- `--image` (required): Path to the input image.
- `--prompt` (default: `What is in the image?`): Text prompt appended after the image placeholder tokens.
- `--max-new-tokens` (default: `40`): Maximum number of tokens to generate.
- `--top-k` (default: `50`): Top-k cutoff used during sampling.
- `--top-p` (default: `0.9`): Nucleus sampling threshold.
- `--temperature` (default: `0.8`): Sampling temperature.
- `--greedy` (flag, disabled by default): If provided, uses greedy decoding instead of stochastic sampling.

Example:

```bash
uv run generate.py --checkpoint checkpoints/projector.pt --image my_image.jpg --prompt "Describe this image."
```
Fetch any image you want e.g. on google image. Click on the image, and in the original website right click and select "copy image url". It gets you a url like `https://www.mydomain.com/my_image.jpg`. Then you can download it with `wget`:

```bash
wget https://www.mydomain.com/my_image.jpg -O my_image.jpg)
```

and use it for generation.
