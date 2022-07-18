
import os
import time

import torch # type: ignore
import numpy as np
import absl.flags
import absl.app

import utils.datasets as datasets
import utils.utils as utils 

# user flags
absl.flags.DEFINE_string("path", None, "dir path where models are stored or path of the model to evaluate")
absl.flags.DEFINE_string("dir_dataset", 'datasets/', "dir path where datasets are stored")
absl.flags.mark_flag_as_required("path")
FLAGS = absl.flags.FLAGS

   

def run_experiment(path:str,dataset_dir:str):
    """ Function to evaluate a set of models inside a dir.
    It prints mean and standard deviation.

    Args:
        path (str): dir path
        dataset_dir (str): dir where datasets are stored
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    loss_criterion = torch.nn.CrossEntropyLoss()
    if os.path.isdir(path):
        list_models = [name_file for name_file in os.listdir(path) if name_file.endswith('.pt')]
    else:
        
        list_models = [os.path.basename(path)]
        path = os.path.dirname(path)
    run_acc = []

    for _, name_model in enumerate(sorted(list_models)):

        # load model
        run,_ = name_model.split('.')
        run = int(run)-1
        utils.set_seed(run)
        checkpoint = torch.load(os.path.join(path,name_model))
        #   load model
        modality = checkpoint['modality']

        model_name = checkpoint['model_name']
        model = utils.get_model(model_name,checkpoint['num_classes'],model_type=modality)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # load data
        mem_examples = checkpoint['mem_examples']
        train_examples = checkpoint['train_examples']
        load_dataset = getattr(datasets, 'get_'+checkpoint['dataset_name'])
        _, _, test_loader, mem_loader = load_dataset(dataset_dir,batch_size_train=128, batch_size_test=500,batch_size_memory=mem_examples,size_train=train_examples,seed=run)

        if modality == 'memory' or modality == 'encoder_memory':
            cum_acc =  []
            init_eval_time = time.time()
            for _ in range(5):
                test_acc, _ = utils.eval_memory(model,test_loader, mem_loader,loss_criterion,device)
                cum_acc.append(test_acc)
            acc_mean = np.mean(cum_acc)
            end_eval_time = time.time()
        else:
            init_eval_time = time.time()
            acc_mean, _  = utils.eval_std(model,test_loader,loss_criterion,device)
            end_eval_time = time.time()

        # stats
        run_acc.append(acc_mean)


        # log
        print("Run:{} | Accuracy {:.2f} | Mean Accuracy:{:.2f} | Std Dev:{:.2f} |  \tE:{:.2f}min".format(run+1,acc_mean, np.mean(run_acc), np.std(run_acc),(end_eval_time -init_eval_time)/60))







def main(argv=None):

    run_experiment(FLAGS.path,FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)