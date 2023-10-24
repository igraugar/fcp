# Forward Composition Propagation algorithm

This repository contains the supplementary material for the paper: 

Grau, I., Nápoles, G., Bello, M., Salgueiro, Y., & Jastrzebska, A. (2023). Forward Composition Propagation for Explainable Neural Reasoning. IEEE Computational Intelligence Magazine, vol. ##, pp. ##-## (in press). [arXiv preprint arXiv:2112.12717](https://arxiv.org/abs/2112.12717).

## Install

LSTCN can be installed from [PyPI](https://pypi.org/project/lstcn)

<pre>
pip install fcp-xai
</pre>

## Background

Forward Composition Propagation (FCP) explains the predictions of feed-forward neural networks trained on structured classification problems. The FCP algorithm is executed on a post-hoc basis, i.e., once the learning process is completed. 

After running FCP, each neuron is described by a composition vector indicating the role of each problem feature in that neuron. Composition vectors are initialized using a given input instance and subsequently propagated through the whole network until reaching the output layer. The sign of each composition value indicates whether the corresponding feature excites or inhibits the neuron, while the absolute value quantifies its impact. 

The example below shows the results of FCP on the german credit dataset as a measure of feature importance in the last layer. However, FCP is more than a feature importance method, as it assigns composition values to the inner layers of the network as well.

## Example usage

```python
from fcp.fcp import forward_composition
compositions = forward_composition(estimator=model, x=instance)
```

where `estimator` is a trained neural network model (`keras.models.Model`) and `x` is an array-like describing the instance to be explained.

FCP can also be used as a global feature importance method by computing the composition values of all instances for a given class and aggregating their values:

<p align="center">
  <img src="https://github.com/igraugar/fcp/blob/main/experiments/fcp_german.png?raw=true" width="1400" />
</p>

For more details see the [example notebook](experiments/example_german_vs_shap_lrp.ipynb).