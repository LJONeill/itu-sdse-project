# ITU BDS MLOPS'25 - Project

#### Prior to working with this project, ensure these versions are installed:

    dvc 3.63.0
    Docker 28.5.2
    git 2.34.1
    dagger v0.19.6
    go 1.25.5

#### Other versions may work, but these are known to work.

# THE PROJECT
## The overall purpose of the Model Artifact is to identify users on the website that are new possible customers. This is done by collecting behaviour data from the users as input, and the target is whether they converted/turned into customers -- essentially a classification problem.

## The diagram below provides an overview of the structure that this project follows.

![Project architecture](./docs/project-architecture.png)

#### The cookiecutter files contain 5 py files which make up the majority of the code from the original jupyter notebook. They are split in adherance to standard data science MLOPS project structure workflow as seen in the diagram below:

![Standard ML Project Structure](./docs/standard-ml-structure.png)

#### The following illustrates the repository structure for this project

├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The README for this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- Contains a file which allows for the dvc pull of the raw data from it's source.
│                          This promotes version control.
│
├── docs               <- A folder containing the images used in this README.md
│
├── models             <- Trained models and their performance metrics
│
├── notebooks          <- Jupyter notebooks containing all the original code for this project, 
│    │                     which is an excerpt and re-written example from a real production model.
│    │             
│    └── main.ipynb    <- This contains the project task outline as well as the data processing part of the ML pipeline
│
├── pyproject.toml     <- Project configuration file with package metadata for src and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
└── src   <- Source code for use in this project.
     │
     ├── config.py               <- Store useful variables and configuration
     │
     ├── dataset.py              <- Scripts to download or generate data
     │
     ├── features.py             <- Code to create features for modeling
     │
     ├── model_selection.py      <- Code to run model comparison and serving       
     │             
     └── train.py                <- Code to train models

### In order to run this project, clone this repository, then type the following into your terminal from the root directory:

    go run pipeline.go

### Subsequently, the most important output, the trained model named 'model' can be found in the models folder