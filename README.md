# CostAwareCR
This is the replication package for our paper "A Multi-Objective Effort-Aware Approach for Early Code Review Prediction and Prioritization".

We provide the necessary data/scripts to replicate our study including: 

1- [Python scripts to train the ML models](https://github.com/stilab-ets/CostAwareCR/blob/main/CostAwareCR_main.py). <br />
2- [The results for the validation data](https://github.com/stilab-ets/CostAwareCR/tree/main/results).<br />
3- [Datasets for the studied projects](https://github.com/stilab-ets/CostAwareCR/tree/main/Datasets).<br />
4- [R scripts for statistical analysis](https://github.com/stilab-ets/CostAwareCR/tree/main/R%20scripts).<br />
5- [The Conda envirement](https://github.com/stilab-ets/CostAwareCR/blob/main/CostAwareCR_conda_env.yml).<br /> 

# How to run CostAwareCR
To run the code of our approach please follow these steps:

1- Clone the the repo.<br />
2- Create the conda environment and install the code's dependencies using the following command:  <br />
```properties
 conda env create --name "envname" --file=CostAwareCR_conda_env.yml
```
3- Activate the created environment.<br />
4- Update the global variables in the code accordingly.<br />
5- Run the Python script.<br />


# How to cite?

Please, use the following bibtex entry:
```tex
@article{chouchen2024multi,
  title={A multi-objective effort-aware approach for early code review prediction and prioritization},
  author={Chouchen, Moataz and Ouni, Ali},
  journal={Empirical Software Engineering},
  volume={29},
  number={1},
  pages={29},
  year={2024},
  publisher={Springer}
}
```
