# Memory Wrap
Official repository of the paper "A self-interpretable module for deep image classification on small data". *Biagio La Rosa, Roberto Capobianco and Daniele Nardi.*  Applied Intelligence (2022). https://doi.org/10.1007/s10489-022-03886-6

The repository contains the PyTorch code to replicate paper results and a guide to use Memory Wrap in your own projects.
## Description
Memory Wrap is an extension to image classification models that improves both data-efficiency and model interpretability, adopting a sparse content-attention mechanism between the input and some memories of past training samples.

![alt text](images/architectures.png "Architectures")

 Memory Wrap outperforms standard classifiers when it learns from a limited set of data, and it reaches comparable performance when it learns from the full dataset. Additionally, its structure and content-attention mechanisms make predictions interpretable, compared to standard classifiers.
# Library
## Installation
This repository contains a PyTorch implementation of Memory Wrap. To install Memory Wrap run the following command:
```
pip install memorywrap
```

The library contains two main classes:
- *MemoryWrapLayer*: it is the Memory Wrap variant described in the paper that uses both the input encoding and the memory encoding to compute the output;
- *BaselineMemory*: it is the baseline that uses only the memory encoding to compute the output.

## Usage
### Instantiate the layer
```python
memorywrap = MemoryWrapLayer(encoder_output_dim, output_dim, head=None, classifier=None, distance='cosine')
```
or
```python
memorywrap = BaselineMemory(encoder_output_dim, output_dim, head=None, classifier=None, distance='cosine')
```
where:
- *encoder_output_dim* (int) is the output dimension of the last layer of the encoder
- *output_dim* (int) is the desired output dimensione. In the case of the paper output_dim is equal to the number of classes;
- *head* (torch.nn.Module): Read head used to project the key and query. It can be a linear or non-linear layer. Input dimensions must be equal to encoder_output_dim (in this case 1280). If None, it is fixed as a linear layer with input and output dimension equal to the input dimension of MemoryWrap(encoder_output_dim). (See https://www.nature.com/articles/nature20101 for further information)
- *classifier* (torch.nn.Module): Classifier on top of MemoryWrap. Inputs dimensions must be equal at encoder_output_dim*2 for MemoryWrapLayer and encoder_output_dim for BaselineMemory. By default is an MLP as described in the paper. An alternative is to use a linear layer. (e.g. torch.nn.Linear(encoder_output_dim*2, output_dim). Default: torch.nn.Sequential( torch.nn.Linear(encoder_output_dim*2, encoder_output_dim*4), torch.nn.ReLU(), torch.nn.Linear(encoder_output_dim*4, output_dim)
- *distance* (str): Distance to use to compute the similarity between input and memory set. Allowed values are: cosine, l2 and dot for respectively cosine similarity, l2 distance and dot product distance. Default=cosine


### Forward call
Add the forward call to your forward function.
```python
output_memorywrap = memorywrap(input_encoding,memory_encoding,return_weights=False)
```
where *input_encoding* and *memory_encoding* are the outputs of the the encoder of rispectively the current input and the memory set, and *return_weights* is a flag telling to the layer if it has to also return the sparse content weights. If you have set the flag *return_weights* to True, then *output_memorywrap* is a Tuple where the first element is the output and the second one are the content weights associated to each element in the memory_encoding.


# Jupyter Notebook
You can find in the <a href="https://colab.research.google.com/drive/1OPjcpTH7X8EV1ev361iuhVzd2Jfp9kFA">following link </a> a jupyter noteebook that explains step by step the process of extending a deep neural network with Memory Wrap.

# Paper Code
To replicate paper results please consult the directory "paper" and the associated README.md. The directory contains all the scripts needed to download the datasets, run the scripts and replicate the results.

# Results

## Performance
MobileNet-v2 Mean Accuracy

| Dataset | Variant        | 1000  | 2000 | 5000 | Full |
| ------------------ | ------------------ |---------------- | -------------- | -------------- | -------------- |
|SVHN| Std   |      42.71       |     70.87        | 85.52 | 95.95 |
|| Memory Wrap   |     **66.93**         |       **81.44**      | **88.68** | 95.63 |
|CIFAR10| Std   |    38.57         |    50.36         | 72.77 | 88.78|
|| Memory Wrap   |    **43.87**          |     **57.12**        | 75.33 | 88.49 |
|CINIC10| Std   |     29.61        |       36.40      | 50.41 |78.85 |
|| Memory Wrap  |      **32.34**        |    **39.48**         | **52.18**  |78.88 |


## Explanatory Images

### Memory Images
They show the samples in the memory set that actively contribute to the prediction of the deep neural network. <br>
<img src="images/svhnmem.png" alt="drawing" width="150"/>
<img src="images/cifarmem.png" alt="drawing" width="150"/>

### Explanation by examples, counterfactuals, and their attribution
Images showing the current input, a similar sample classified in the same class (explanation by example) and a similar sample classified in a different class (counterfactual).<br>
<img src="images/heatmap1.png" alt="drawing" width="150"/>
<img src="images/heatmap2.png" alt="drawing" width="150"/>

# Referenced repositories and libraries
- Implementantion of the architectures are taken from Kuangliu repository:
<a href="https://github.com/kuangliu/pytorch-cifar">https://github.com/kuangliu/pytorch-cifar</a>
- The <a href="https://captum.ai/">Captum library</a> has been used to apply the Integrated gradients method.

# Citation
Please cite our paper as:
```tex
@Article{LaRosa2022,
  author    = {Biagio {La Rosa} and Roberto Capobianco and Daniele Nardi},
  journal   = {Applied Intelligence},
  title     = {A self-interpretable module for deep image classification on small data},
  year      = {2022},
  month     = {aug},
  doi       = {10.1007/s10489-022-03886-6},
  publisher = {Springer Science and Business Media {LLC}},
}
```
