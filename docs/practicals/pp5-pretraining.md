# Programming Practical 5 - Pretraining GPT2 on a cluster

In this Programming Practical, you will **pretrain and finetune a GPT-2 language model** on a GPU cluster managed by **SLURM**. You will learn how to:

- Connect to a remote cluster and set up your environment.
- Write and submit **SLURM batch jobs** — the standard way to run workloads on HPC clusters, ubiquitous in the AI industry. Mastering SLURM is a crucial skill for any AI practitioner who needs to train models at scale.
- Complete the missing parts of a GPT-2 implementation (data loading, training loop, and text generation).
- Pretrain a small GPT-2 from scratch on a French Philosophy corpus.
- Finetune a pretrained GPT-2 on the same corpus and compare results.

The code is based on Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master?tab=readme-ov-files), a minimalist but complete GPT-2 implementation.
## 1. Logging in the cluster

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

Change your password at first login:

```bash
passwd
```

## 2. Description of the cluster

This cluster is made of 15 compute node with 2 Nvidia A100 and 80Gb of RAM each. 

For the storage, you have access to three storage spaces:
- a 10Gb home directory (`~`), for storing code and softwares.
- a 1Tb `/work/formation/YOUR_USERNAME` directory, dedicated to computation input/output and results.
- a virtually infinite `/tmpdir/YOUR_USERNAME` directory, where you will store environments.

At logging, you will be in the login node, in your home directory.

!!! Note
    The compute nodes are not connected to the internet, so the scripts that automatically download stuff from the internet (datasets, weights, etc.) has to be first run on the login node. Once the data is downloaded, if the script automatically retrieve downloaded data, it can be run on the compute nodes.

## 3. Setting up the environment

You will use `uv` like in the previous practicals but with some subtlelties. 

The cluster has a shared environment with a pytorch version optimized for the compute nodes that can be accessed with `apptainer`, which is similar to `docker` but adapted for clusters. 

What you will do is to create a `uv` environment on top of the pytorch apptainer image. This way, you can access the optimized pytorch version while still being able to install custom dependencies with `uv`.

First, install `uv` on the login node:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Pull the repo

First, clone your forked repo in the home directory. For convenience, you can clone the repo with ssh

![alt text](../assets/image.png)

Create an ssh key in the login node with `ssh-keygen` and add the content of `~/.ssh/id_rsa.pub` to your GitHub account.

![alt text](../assets/image2.png)

click 'SSH and GPG keys', then 'New SSH key', then paste the content of `~/.ssh/id_rsa.pub` (you can access it with `cat ~/.ssh/id_rsa.pub`) in the 'Key' field, give it a title and click 'Add SSH key'.

Then you can clone your forked repo with:

```bash
git clone your_forked_URL
cd repo_name
```

Then add this repo as a remote to pull updates

```bash
git remote add upstream https://github.com/paulnovello/Advanced-AI
```

To update your forked repo with the latest changes from this original repo, run:

```bash
git fetch upstream
git merge upstream/main
```

### Work remotely from vscode

I strongly encourage you to use vscode remote environment to work on the project. On the leftbar of vscode, you should see an icon "Remote Explorer". Click on it, then click on "SSH" if needed, and click the left arrow next to "turpan". You will have to fill your password. Once you are connected, you can open the project folder and work on it as if it was local. You can even run the code in a terminal in vscode.

### Create the `uv` environment

Then you have to launch the `apptainer` image and create your env on top of it. 

First, create an env directory in `/tmpdir`:

```bash
mkdir -p /tmpdir/YOUR_USERNAME/envs/aai
```

Then, go to your project directory with `cd` and launch the apptainer image on the **login** node (do not forget to replace YOUR_USERNAME):

```bash
apptainer shell --env PATH=$HOME/.local/bin:$PATH --env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif
```

!!! Hint
    Once you have completed this command with YOUR_USERNAME, save it as an alias in your `~/.bashrc` to avoid having to write it every time. Add these lines:

    ```bash
    alias run_apptainer_login="apptainer shell --env PATH=$HOME/.local/bin:$PATH --env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif"
    ```
    and then refresh the `~/.bashrc` with `source ~/.bashrc`. Then you can simply run `run_apptainer_login` to launch the apptainer image on a compute node.


You are now in the `apptainer` image! Install the env using:

```bash
uv venv --system-site-packages /tmpdir/YOUR_USERNAME/envs/aai
uv sync --only-group turpan
``` 

Now the environment should be up and running.

## 4. Running the code on the compute nodes

To run some code on the compute nodes, you have two choices. You can either use the node in interactive mode, meaning that you have a shell on the compute node where you can run commands one by one, or you can submit a job, meaning that you write a script with the commands you want to run and submit it to the cluster, which will run it for you.

### Interactive mode

Launch an interactive session on a compute node with (do not forget to replace YOUR_USERNAME):

```bash
srun -p shared -n1 --gres=gpu:1 --pty apptainer shell --env PATH=$HOME/.local/bin:$PATH  --env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif
```

!!! Hint
    Once you have completed this command with YOUR_USERNAME, save it as an alias in your `~/.bashrc` to avoid having to write it every time. Add these lines:

    ```bash
    alias run_apptainer_gpu="srun -p shared -n1 --gres=gpu:1 --pty apptainer shell --env PATH=$HOME/.local/bin:$PATH  --env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif"
    ```
    and then refresh the `~/.bashrc` with `source ~/.bashrc`. Then you can simply run `run_apptainer_gpu` to launch the apptainer image on a compute node.

You can check that you are on a compute node by using `nvidia-smi`. Then you can run your scripts as you would do in a local machine.

### Batch mode

This is where things get complicated :)

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
#SBATCH --reservation=tpirt4
#SBATCH --output=/users/formation/YOUR_USERNAME/job_results/out/job_%j.out
#SBATCH --error=/users/formation/YOUR_USERNAME/job_results/err/job_%j.err


apptainer exec \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/YOUR_USERNAME/envs/aai \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
uv run --no-sync python mon_script.py
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
sbatch --reservation=tpirt4 run_job.sbatch
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


**Exercice** Try to create a script `test_script.py` that prints "Hello world!" and submit it to the cluster with the `.sbatch` script template. Check the output and error files to see if it worked. 

## 5. Pretraining GPT2

The code you will use is taken from Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master?tab=readme-ov-files), which is a minimalist implementation of GPT-2. You will find the code in `PP5: Pretraining GPT2` folder.

### Complete the missing parts

Before launching your first training on the cluster, you will have to complete the missing parts in the code involved in training and generation, as in previous PPs. Start by reading `model.py` and `train.py` to understand the overall structure.

---

#### `get_batch` in `train.py`

**What it does:** a simple data loader that samples random chunks from a memory-mapped binary file of token ids. Each call returns a batch of input-target pairs for next-token prediction.

The data is stored as a flat array of token ids. For a language model, the input `x` is a window of `block_size` consecutive tokens, and the target `y` is the same window shifted by one position (i.e., for each token in `x`, the target is the next token in the sequence).

```python
def get_batch(split):
    ...
    data = np.memmap(...)
    ix = # TODO: sample batch_size random starting indices in [0, len(data) - block_size)
    x  = # TODO: for each index i in ix, extract a chunk of block_size tokens as input
    y  = # TODO: for each index i in ix, extract a chunk of block_size tokens shifted by 1 as target
    ...
```

- `ix`: use `torch.randint` to sample `batch_size` random starting positions. The upper bound should ensure that a full window of `block_size` tokens fits.
- `x`: for each starting index `i`, slice between `i` and `i + block_size`, convert to `np.int64`, wrap in a tensor, and stack all slices into a `(batch_size, block_size)` tensor.
- `y`: same as `x`, but shifted by one.

---

#### Training loop in `train.py`

**What it does:** the forward-backward pass inside the gradient accumulation loop. The model receives input tokens `X` and must produce predictions that are compared against the targets `Y` using cross-entropy loss.

```python
for micro_step in range(gradient_accumulation_steps):
    with ctx:
        logits = # TODO: forward pass through the model
        loss   = # TODO: compute the cross-entropy loss
        loss = loss / gradient_accumulation_steps
    ...
```

- `logits`: call the model on `X`. The output has shape `(batch_size, block_size, vocab_size)`.
- `loss`: use `F.cross_entropy`. You need to reshape `logits` to `(B*T, vocab_size)` and `Y` to `(B*T)`. Use `ignore_index=-1`.

---

#### `generate` in `model.py`

**What it does:** autoregressive text generation. Given a conditioning sequence of token ids, the model predicts one token at a time, appends it to the sequence, and repeats.

The generation loop has the following steps:

1. **Forward pass** — run the (possibly cropped) context through the model to get logits.
2. **Select & scale** — take only the logits at the last time step and divide by the temperature.
3. **Top-k filtering** — optionally zero out all logits outside the top-k most likely tokens.
4. **Sample** — convert logits to probabilities with softmax, then sample one token.
5. **Append** — concatenate the new token to the running sequence.

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits = # TODO: call the model on idx_cond
        logits = # TODO: select the logits at the last time step (index -1), divide by temperature
        if top_k is not None:
            v, _ = # TODO: use torch.topk to get the top-k values
            # TODO: set all logits below the smallest top-k value (v[:, [-1]]) to -float('Inf')
        probs    = # TODO: apply softmax to get probabilities
        idx_next = # TODO: sample from the distribution (torch.multinomial)
        idx      = # TODO: concatenate idx_next to idx along dim 1
    return idx
```

- For the forward pass, simply call the model — the model's `forward` method returns logits of shape `(batch, seq_len, vocab_size)`.
- `logits[:, -1, :]` selects the predictions for the last position. Dividing by `temperature` controls randomness: lower temperature → more deterministic.
- `torch.topk` returns the `k` largest values; use the smallest of those as a threshold to mask everything else to `-Inf`.
- `F.softmax(logits, dim=-1)` converts to probabilities; `torch.multinomial(probs, num_samples=1)` draws one sample per row.
- `torch.cat((idx, idx_next), dim=1)` grows the sequence by one token.


### Pretraining on French Philosophy

The dataset you will use is a collection of French Philosophy books from the [Gutenberg project](https://www.gutenberg.org/). You can find the dataset in `data/french_philosophy`. It is already split in a training and a validation set.

The training configuration is in `config/train_french_philosophy.py`. It defines a small GPT-2 (6 layers, 6 heads, 384-dim embeddings) trained for 5000 iterations on 256-token contexts. Take a moment to read the config file and understand the hyperparameters.

**Exercice** Create an `.sbatch` script (based on `sbatch_scripts/template.sbatch`) that launches the pretraining. You will need to:

1. Copy the template and adapt `YOUR_USERNAME` and the script path.
2. Set the command to `uv run --no-sync python train.py config/train_french_philosophy.py`.
3. Set an appropriate time limit (30 minutes should be sufficient).
4. Submit your job with `sbatch --reservation=tpirt4 your_script.sbatch`.
5. Monitor progress with `squeue -u $USER -l` and check the output/error files in `~/job_results/`.


At the end of the training, you can sample from the model (from the apptainer) using:

```bash
uv run --no-sync python sample.py --out_dir=out-french-philosophy --start="Your prompt here"
```
**Exercice** Try to run it on a compute node in interactive mode to see how using GPUs accelerate the sampling.

### Finetuning on French Philosophy

Instead of training from scratch, you can start from the pretrained GPT-2 (124M parameters) weights and finetune on the French Philosophy corpus. The configuration is in `config/finetune_french_philosophy.py`. Notice the key differences with pretraining from scratch: lower learning rate, no learning rate decay, dropout enabled, and the model is initialized from `gpt2` weights.

!!! Note
    The finetuning config uses `init_from = "gpt2"`, which downloads the pretrained weights from HuggingFace. Since compute nodes have no internet access, you must first run the script once on the **login node** so that the weights are cached. Run: `uv run --no-sync python train.py config/finetune_french_philosophy.py` on the login node before submitting the job. This should detect that you are not in a compute node and stop before training.

**Exercice** Create an `.sbatch` script that launches the finetuning. You will need to:

1. Copy the template and adapt `YOUR_USERNAME` and the script path.
2. Set the command to `uv run --no-sync python train.py config/finetune_french_philosophy.py`.
3. Set an appropriate time limit (30 minutes should be sufficient).
4. Submit your job and monitor its progress as before.
5. Compare the final validation loss with the pretrained-from-scratch model. Which one is better?

At the end of the training, you can sample from the finetuned model (from the apptainer) using:

!!! Note
    Because of tiktoken trying to download the tokenizer files and the apptainer blocking the HTTP, you will need to create a conda env:

    ```bash
    module load conda
    conda create --name tiktoken_env
    conda activate tiktoken_env
    conda install tiktoken
    python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
    ``` 

    This will download the files and normally you will be able to use tiktoken in the apptainer.


```bash
uv run --no-sync python sample.py --out_dir=out-french-philosophy-ft --start="Your prompt here"
```

**Exercice** Try to run it on a compute node in interactive mode to see how using GPUs accelerate the sampling.