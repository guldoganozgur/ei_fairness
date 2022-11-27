# Equal Improvability Fairness
## Installation instructions

First install the repo.

To create virtual environment:


```shell
cd ei_fairness
conda env create --file environment.yml
```

Note: Assuming you installed TeX. Since, the figures require TeX. To get TeX, you can follow the instructions at that [link](https://www.latex-project.org/get/).

## Repeating Experiments

For each figure in the experiments section, there is a separate python notebook. The correspondence for each figure:

For figure 3, ```Tradeoffs.ipynb```.\
For figure 4&5, ```PopulationDynamics-Gaussian.ipynb```.

To repeat experiments for the table 2, 3, and 4, for each dataset and model (Logistic Regression and Multilayer Perceptron) there is a python notebook. That python notebook explains the hyperparameter selection and implements the experiments for that setting.
