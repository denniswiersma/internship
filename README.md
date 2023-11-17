# Internship
NOTE: The contents of this repository have note yet been licensed.

This repository contains code and writings for my internship project at the Computational Transcriptomics research group at [University Medical Center Groningen](https://umcgresearch.org/).
The project is part of the fourth (and final) year of the bachelor Bio-information technology at [Hanze University of Applied Sciences](https://www.hanze.nl/en).

## Annotation of large-scale transcriptomic data using machine learning techniques

[adaptation of project proposal introduction]

## Getting started

### Acquiring code & data

This repository can be cloned with one simple terminal command:
```bash
git clone https://github.com/denniswiersma/internship.git
```
or if you prefer an ssh based approach:
```bash
git clone git@github.com:denniswiersma/internship.git
```

The input data consists of a mixing matrix derived from publicly available mRNA expression profiles by using Consensus Independent Component Analysis.
More information on this process can be found in [paper 1](https://doi.org/10.1186/s40170-021-00272-7) and [paper 2](https://doi.org/10.1038/s41467-021-21671-w).

In addition to the mixing matrix, a file containing sample cancer type annotations is to be provided.

Please note that the program will try to load the data into data frames with column names.
Make sure the data you provide follows this format. See example below.

[Example of mixing matrix data]

### Dependencies
This project has dependencies for both Python and R.
The latter of these, unfortunately, requires some manual installation.

#### Python
[Poetry](https://python-poetry.org/) is a python packaging and dependency management tool for python.
The easiest way to install all the required dependencies is to first install Poetry by following [their installation instructions](https://python-poetry.org/docs/#installation).
Once this is done, navigate to this project and run the following command:
```bash
poetry install
```

If you wish to (manually) install dependencies into your own environment, please refer to the `pyproject.toml` file.
You'll find all required Python dependencies under `[tool.poetry.dependencies]`.

#### R
Since, internally, an R environment is used to run certain machine learning algorithms, some R dependencies need to be installed.
If you have a [working R installation](https://www.r-project.org/) you can open an interactive R console in your terminal by running the `R` command.
Run the following command in this console to install all required R dependencies:
```R
install.packages(c("partykit", "glmnet", "grDevices"))
```

### Configuration
To get started configuring, make a copy of `config_template.toml` and name it `config.toml`.
A few configurations need to be changed from their defaults while the rest is optional.


- `data.locations` | Paths to files containing the mixing matrix and cancer type annotations.
- `data.columns` | Names of the columns of the mixing matrix and annotations.
- `output.locations` | Paths to where the output of certain parts of the analysis should be saved. (optional)
- `execution.cores` | Number of cores to use where multiprocessing is applicable. (optional)

Output directories will be divided into subdirectories named by a run ID made up of a datetime stamp and a randomly generated two byte long hexadecimal string.
A new run ID is generated for each fit, and thus, any output generated will be saved to its respective directory.
Which data is or isn't saved depends on which functions are called. As an example, this simplified code:

```python
ctree = Ctree()

ctree.fit()

ctree.save()
ctree.plot()
```

Would result in the following output directory structure:

```
output
├── ctree
│   └── 20230101120000_7ca166 <- runID: 2023-01-01 12:00:00 + random string
│       ├── ctree.pkl <- saved model object
│       └── ctree.png <- plot
```


### Recommended usage
Code contained in the `src/ml/` directory may be viewed as library code, and can therefore be called by user made scripts.
For analysing data, training models, and reproducing methods it is recommended to use [Jupyter Notebooks](https://docs.jupyter.org/en/latest/) and call the code from there.
Please examine examples in the `notebooks/` directory for guidance.

## Organisation
Ideally, every (top-level) folder in this repository will contain a readme file discussing its contents.
As this is the project's root, all top-level contents are discussed below.

### documents
Contains documented that need to be produced as part of the internship project.
Documents are written in [LaTeX](https://www.latex-project.org/) so that they can be compiled by anyone.
Instructions for compilation will be provided in the directory's readme at a later date.

Currently contains:
- pva | Plan of approach.
- log | Week by week project log. Abandoned by week five and might therefore be removed and/or replaced by supplementary methods.

### eda
Contains python scripts used in the Exploratory Data Analysis phase of the project.
It is unknown if this code functions as expected, since it does not utilise the current data pipeline.

### notebooks
Contains a separate Jupyter Notebook for each `Model` implementation used to perform the analyses for the given model.
See [Recommended usage](https://github.com/denniswiersma/internship#recommended-usage).

### src
Contains all source code that can be used for your own endeavours.

### config_template.toml
Contains a template for `config.toml`.
See [Configuration](https://github.com/denniswiersma/internship#configuration).
