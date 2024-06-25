# Shapley FDA experiments
This project studies the value of games with a continuum of players.
Note that the project ``functional_neural_networks`` is a clone from the [Functional Neural Networks repository](https://github.com/FlorianHeinrichs/functional_neural_networks).

To run the project, first, the data must be generated:
```python
python utils.data_generator.py
```

Once the data is generated, then the Shapley values can be obtained:
```python
python utils.workflow.py
```

Finally, to analise the results, use [this notebook](https://github.com/pachoning/shapley_fda_experiments/blob/main/notebooks/analises_shapley_experiments.ipynb). To analyse tecator dataset, unse [this notebook](https://github.com/pachoning/shapley_fda_experiments/blob/main/notebooks/real_data.ipynb).