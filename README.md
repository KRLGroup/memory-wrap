# Memory Wrap: a Data-Efficient and Interpretable Extension to Image Classification Models  

This repository is the official implementation of Memory Wrap: a Data-Efficient and Interpretable Extension to Image Classification Models.

## Requirements

To run the code you need to install the following packages:

- Python 3.8.5
- PyTorch 1.7 (https://pytorch.org/)
- Torchvision 0.8.1 (included in PyTorch)
- absl-py 0.10 (https://pypi.org/project/absl-py/)
- captum 0.3.0 (https://captum.ai/)
- entmax 1.0 (https://github.com/deep-spin/entmax)
- pyyaml 5.3.1 (https://pypi.org/project/PyYAML/)
- Numpy 1.19 (included in Python)
matplotlib 3.3.2 (included in captum)
- scipy 1.5.2 (https://www.scipy.org/)

To install all the packages inside a new Conda environment please type the following commands:

```
conda create --name MemoryWrap
conda activate MemoryWrap
conda install python=3.8
pytorch torchvision torchaudio -c pytorch
conda install -c anaconda absl-py
conda install -c pytorch captum
conda install -c anaconda pyyaml
conda install -c anaconda scipy
```

## Repository Structure
This is the description of repository structure. Note that folders inside "images" dir and "models" dir will be created at running time only when you run the respective scripts to generate images or models.
```  
   Memory Wrap
   ├── architectures
   │   ├── effientnet.py
   │   ├── mobilenet.py
   │   ├── resnet.py
   │   └── memory.py
   ├── config
   │   └── train.yaml
   ├── images
   │   ├── mem_images
   │   |   ├── SVHN
   │   |   |   ├── memory
   │   |   |   └── encoder_memory
   │   |   └── CIFAR10
   │   |       ├── memory
   │   |       └── encoder_memory
   |   └── saliency
   │       ├── SVHN
   │       |   ├── memory
   │       |   └── encoder_memory
   │       └── CIFAR10
   │           ├── memory
   │           └── encoder_memory
   ├── models
   │   ├── CIFAR10 
   │   |   ├── memory
   │   |   └── encoder_memory
   │   └── SVHN
   │       ├── memory
   │       └── encoder_memory
   ├── README.md
   ├── train.py
   ├── eval.py
   ├── explanation_accuracy.py
   ├── generate_heatmaps.py
   ├── generate_memory_images.py
   ├── aux.py
   └── datasets.py
```

## Architectures
This directory contains the PyTorch modules of MobileNet, EfficientNet, ResNet and Memory Wrap.

## memory.py
The memory.py file contains the modules needed to create a Memory Wrap. It contains 4 classes:
- Identity: module used to replace the last layers of standard classifiers with Memory Wrap (see an old discussion on https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648)
- MLP: implementation of multi-layer perceptron used as last layer of Memory Wrap
- MemoryWrap: module that implements the variant of Memory Wrap that uses only the memory content to output the prediction
- EncoderMemoryWrap: module that implements the variant of Memory Wrap that uses both the memory content and the encoder to output the prediction
## Configuration File (for training)
### config/train.yaml

It is the configuration file used to store the parameters for
the training process. In order to obtain the results described
in the paper you have to change the following parameters, choosing
one of the listed values, and then run the train.py script:
- model: [mobilenet, efficientnet, resnet18]
- train_examples: [1000, 2000, 5000, 100000] 
- dataset_name: [SVHN, CIFAR10]

> When you set the number of training samples (train_examples parameter) to 100000, you are doing a training process using the whole training dataset.

List of other parameters of interest:
- save (default: True): a flag that allows train.py to save the model after each run
- dataset_dir (default: ../datasets): dir where datasets are stored or path where they will be stored
- runs (default: 30): number of runs to complete in order to summarize results and to stop the training script

Other parameters (changing their values will change also the results):
- batch_size_train (default: 128): batch size of training batches
- batch_size_test (default: 500): batch size of testing batches. Note that each batch shares the same memory. Increasing too much this parameter could change in a significant way the results. 
- optimizer:
  - learning_rate (default: 1e-1)
  - weight_decay (default: 5e-4)
  - momentum (default: 0.9)
  - nesterov (default: True)
- dataset configurations:
   - SVHN:
     - num_classes (default:10)
     - num_epochs (default:40)
     - opt_milestones (default: [20, 30]): milestones after which the learning rate is decreased
     - mem_examples (default:100): number of samples in memory
   - CIFAR10:
     - num_classes (default: 10)
     - num_epochs (default: 300)
     - opt_milestones (default: [150, 225]): milestones after which the learning rate is decreased
     - mem_examples (default:100): number of samples in memory
## Training

To train the models in the paper, run this command:

```train
python3 train.py modality=<MODALITY> --continue_train=False --log_interval=100
```
- **modality** indicates the type of model to be trained. It can be *"std"* for classical architectures,*"memory"* for the Memory Wrap variant that uses only the
 memory or *"encoder_memory"* for the Memory Wrap variant that uses both the 
 encoder and the memory. 
 - **continue_train** is a boolean flag that allows retrieving the scores
 of previous stopped runs and it restarts the script using that information to fix
 seeds and global statistics.
 -- **log_interval**: log interval between prints during the training process

**Expected Output**: At the end of runs, the script will print one of the lines in Table 1 of the paper.

Saving Path: if the parameter "save" is True on the train.yaml file, then models are saved on the following relative path:
```
models/DATASET_NAME/MODALITY/MODEL_NAME/TRAIN_EXAMPLES/
```
 using as name "NUM_RUN.pt". where NUM_RUN is the index of the current run.

> Example: the memory model of the second run using mobilenet as encoder and trained on CIFAR10 using 2000 samples in the training set will be stored on: <br> models/CIFAR10/memory/mobilenet/2000/2.pt

If you want to modify this path, please edit the section "saving/loading stuff" in the train.py script.
## Evaluation

To evaluate your trained model, run:

```eval
python eval.py --path_model=<PATH_MODEL> --dataset=<NAME_DATASET> --modality=<MODALITY>
```
- **path_model**: it is the path where the model is stored
- **dataset** (*CIFAR10* or *SVHN*): dataset used to train the model
- **modality** (*std*, *memory* or *encoder_memory*): indicates the type of trained model.

To compute the explanation accuracy over different runs, run:
```
python3 explanation_accuracy.py --path_models=<DIR_MODELS> --dataset=<NAME_DATASET> --modality=<MODALITY>
```
- **path_models**: it is the path where models are stored
- **dataset** (*CIFAR10* or *SVHN*): dataset used to train the model
- **modality** (*std*, *memory* or *encoder_memory*): indicates the type of trained model.

**Expected Output**:
```
Explanation accuracy (mean):0.00       (std_dev):0.00    counterfactual acc mean:0.00       counterfactual std:0.00    example acc mean:0.00   example std:0.00
```
Where *Explanation accuracy*  is the metric shown in Table 3. *counterfactual mean* refers to the performance shown in Table 4. *example acc* refers to the accuracy reached by the model for inputs where the sample in memory associated with the highest weight is an explanation by example.


## Generate Memory Images
Script to generate images containing the input and the samples in memory associated with a positive weight.
<br>**Note:** Memory used to generate  memory images is DIFFERENT from memory used to heatmaps of generate_heatmaps.py scripts, due to different batch sizes and the usage of two iterators on the same datasets (for heatmaps).

Command:
```
python3 generate_memory_images.py --path_model=<PATH_MODEL> --dataset=<NAME_DATASET> --modality=<MODALITY>
```
- **path_model**: it is the path where the model is stored
- **dataset** (*CIFAR10* or *SVHN*): dataset used to train the model
- **modality** (*std*, *memory* or *encoder_memory*): indicates the type of trained model.

Images will be stored in the following path:
```
images/mem_images/DATASET_NAME/MODALITY/MODEL_NAME/
```
To modify the path, please edit the value of variable *dir_save*.
## Generate Heatmaps
Script to generate heatmaps of images in the training dataset. Heatmaps are computed using the Integrated Gradient attribution method, using as a baseline a white image.
 <br>**Note:** Memory used to generate heatmaps is DIFFERENT from memory used to generate memory images of generate_memory_image.py scripts, due to different batch size the usage of two iterators on the same datasets (for heatmaps script).

Command:
```
 generate_heatmaps.py --path_model=<PATH_MODEL> --dataset=<NAME_DATASET> --modality=<MODALITY> --batch_size_test=3
```
- **path_model**: it is the path where the model is stored
- **dataset** (*CIFAR10* or *SVHN*): dataset used to train the model
- **modality** (*std*, *memory* or *encoder_memory*): indicates the type of trained model.
- **batch_size_test**: number of testing samples to show inside one image (tested on batch size < 3)

To change the baseline modify the variable *baseline_memory* and *baseline_input* inside the script. Note that changing the baseline will change also the heatmaps, as explained in the appendix.

> Note: if the process is killed or you receive an OOM message, it is likely that you have not enough VRAM in your GPU in order to apply Integrated Gradients. In this case modify the first instruction in run method to use your cpu (variable "device").

Heatmaps will be stored in the following path:
```
images/saliency/DATASET_NAME/MODALITY/MODEL_NAME/
```
To modify the path, please edit the value of variable *dir_save*.

NB: The scripts generates one image at time!
## Pre-trained Models
You can find a pre-trained model used to generate
 part of the images of the paper in the following path:
```
models/CIFAR10/memory/mobilenet/5000/pretrained.pt
```
To get models for the other configurations, please run the 
training script described above.

## Seeds
We fix the seed at the beginning of each sensible script with the following instructions:
```
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
```

For the training script we change the seed at each run in the following manner:
```
seed = NUM_RUN
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
```
where NUM_RUN is the number of the current run.

## Auxiliary files

### datasets.py
It contains functions to splits the datasets and to get PyTorch Dataloaders for the training and testing phase.
### aux.py
It contains auxiliary functions used to train and load models.
