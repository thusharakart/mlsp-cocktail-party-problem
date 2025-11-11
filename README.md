# mlsp-cocktail-party-problem

Project in Machine Learning for Signal Processing. Goal: to solve the cocktail party problem for 2+ speaker mixed audio signals, evaluating various simple methods and more advanced models.


## Setup
The `Makefile` in the project root contains various commands that simplify environment setup, dataset downloads, and running experiments. These can be run in a terminal using `make <command>` (inspect `Makefile`) from the project ROOT. This way everyone on the team can run the exact same commands with the correct arguments, reducing errors and ensuring consistency.

Any python script can also be run directly from a terminal using `python src/<script name>.py <arguments>` from the project ROOT with the appropriate arguments. This allows more specific arguments than the `Makefile` commands, and so is useful during development.

Use whichever you're more comfortable with.

### First Time Setup
1. Clone the repository:
```bash
git clone https://github.com/MaxTheTech/mlsp-cocktail-party-problem.git
cd <repo path>
```

2. Create conda environment `mlsp-project`, install requirements and activate it:
```bash
make environment-setup
conda activate mlsp-project
```
Or if environment already exists, you only need to activate it.

### Update requirements
If you only want to update the environment packages specified in `environment.yml`:
```bash
make environment-update
```



## Datasets
Do not push locally saved datasets to remote repository. Git will ignore all files with `data/` subdirectory.

### LibriMix
Generates mixed speaker datasets from the LibriSpeech single-speaker dataset, with the generated data specifically formulated to help with training speaker separation models. Source: https://github.com/JorisCos/LibriMix?tab=readme-ov-file

We are generating Libri2Mix with final size of around 30 GB, which includes both training, validation and test splits.

Generation datasets: LibriSpeech dev (clean), LibriSpeech test (clean), LibriSpeech train 100 hours (clean), and WHAM for noise.

Generation parameters: 2-speaker mix, 16 kHz frequency, minimum mode (mixture has same length as shortest source speaker sample), both clean (no noise) + noise mixtures (WHAM noise).

We chose these parameters to ensure a smaller and more manageable dataset size. Running the most expansive parameters would lead to a size of 430GB of data for Libri2Mix and 332GB for Libri3Mix, according to the repo source.




## Training models
Up to individual model group members depending on their model.


## Evaluating and testing models
Up to individual model group members depending on their model.







