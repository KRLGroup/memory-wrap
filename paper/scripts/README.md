# Scripts 
All the scripts in this directory assume that the trained models are supposed to **contain the information needed** like the dataset, the number of training samples, the modality, etc. This information are automatically stored if you use our training script. Otherwise you have to manually modify the lines about the checkpoint**

## Generate Memory Images
Script to generate images containing the input and the samples in memory associated with a positive weight.
<br>**Note:** Memory used to generate  memory images is DIFFERENT from memory used to heatmaps of generate_heatmaps.py scripts, due to different batch sizes and the usage of two iterators on the same datasets (for heatmaps).

Command:
```
python3 generate_memory_images.py --path_model=<PATH_MODEL> --dir_dataset='../datasets/'
```
- **path_model**: it is the path where the model is stored
- **dir_dataset**: dir where datasets are stored

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
 generate_heatmaps.py --path_model=<PATH_MODEL> --dir_dataset='../datasets/' --batch_size_test=3
```
- **path_model**: it is the path where the model is stored
- **batch_size_test**: number of testing samples to show inside one image (tested on batch size < 3)
- **dir_dataset**: dir where datasets are stored

To change the baseline modify the variable *baseline_memory* and *baseline_input* inside the script. Note that changing the baseline will change also the heatmaps, as explained in the appendix.

> Note: if the process is killed or you receive an OOM message, it is likely that you have not enough VRAM in your GPU in order to apply Integrated Gradients. In this case modify the first instruction in run method to use your cpu (variable "device").

Heatmaps will be stored in the following path:
```
images/saliency/DATASET_NAME/MODALITY/MODEL_NAME/
```
To modify the path, please edit the value of variable *dir_save*.

NB: The scripts generates one image at time!

## Memory-Based Modules

To reproduce the experiments about memory-based modules please run the following command to use Matching Networks:
Command:
```
python3 run_matching.py --continue_train=False --optimizer='sgd'
```

and the following command to use ProtoNet:
```
python3 run_protonet.py --continue_train=False --optimizer='sgd'
```

Both the scripts share these arguments:
- **optimizer** String to select the optimizer. Admissible values are sgd and adam.
- **continue_train** boolean to restart a previous interrupted training process.

## Explanations By Examples
To reproduce the experiments about explanations by examples, please run the following command

Command:
```
python3 run_exp_by_examples.py --dir_models=<PATH_MODELS> --dir_dataset='../datasets/' --metric=<METRIC>
```
- **dir_models**: it is the path where the models for which you want to train autoencoder are stored
- **dir_dataset**: is the directory where the dataset is stored.
- **metric**: Metric to be used for evaluating the explanations. Admissible values are 'prediction' to use the prediction non-representativeness and
            'input' to use the input non-representativeness

The script prints the metric scores for all the competitors considered in the paper. 
## Counterfactuals 

To reproduce the experiments about counterfactuals using the post-hoc method based on prototypes and the intrinsic method use the following commands:

First train the autoencoder for the set of models you want to explain. This is needed to compute the scores to evaluate the methods.

Command:
```
python3 train_aes_svhn.py --dir_models=<PATH_MODELS> --saving_path=<SAVING_PATH> --dir_dataset='../datasets/'
```
- **dir_models**: it is the path where the models for which you want to train autoencoder are stored
- **saving_path**: dir where autencoders will be save
- **dir_dataset**: is the directory where the dataset is stored.

Then run the run_counterfactuals.py script to get the counterfactuals and their scores.

Command:
```
python3 run_counterfactuals.py --path_models=<PATH_MODELS> --path_aes=<PATH_AES>  --dir_dataset='../datasets/' --algo=<ALGO>
```
where 
- **path_models**: is the path where the models for which you want to get counterfactuals are stored. Its value should be the same of dir_models argument of the previous script. 
- **path_aes**: is the path where the autoencoders for the models of the path_models arguments are stored. Its value should be the same of saving_path argument of the previous script. 
- **dir_dataset**: is the directory where the dataset is stored.
- **algo**: algorithm to use to compute the counterfactuals. Supported values for the algo argument are 'proto' to use the post-hoc method, and 'memory' to use the intrinsic method of Memory Wrap.

**(IMPORTANT)** Note that:
- The experiment has been done only on the SVHN dataset 
- This script assume that you have already run the train_aes_svhn.py using path_models as the path_models argument and the path_aes as the saving path argument.
- Some parameters are hard-coded to be consistent with the original experiments and repository (e.g., batch size)