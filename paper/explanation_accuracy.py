import utils.datasets as datasets
import torch # type: ignore
import numpy as np
import absl.flags
import absl.app
import os
import utils.utils as utils
from typing import Tuple

# user flags
absl.flags.DEFINE_string("path_models", None, "Path of the trained model")
absl.flags.DEFINE_string("dir_dataset", 'datasets/', "dir path where datasets are stored")
absl.flags.mark_flag_as_required("path_models")

FLAGS = absl.flags.FLAGS


def get_explanation_accuracy(model:torch.nn.Module,loader:torch.utils.data.DataLoader,mem_loader:torch.utils.data.DataLoader,device:torch.device)-> Tuple[float, float, float]:
    """ Method to compute the explanation accuracy for different settings, as
    described in the paper. Explanation accuracy checks how many times the
    sample in the memory set with the highest weight is predicted in the same
    class of the current sample.
image

    Args:
        model (torch.nn.Module): Trained model to evaluate
        loader (torch.utils.data.DataLoader): loader containing testing
            samples where evaluate the model
        mem_loader (torch.utils.data.DataLoader): loader containing training
            samples to be used as memory set
        device (torch.device): device where the model is stored

    Returns:
        Tuple[float, float, float]: Explanation accuracy for the sample with
        the highest weight in memory, explanation accuracy for the sample with
        the highest weight in memory when it is a counterfactual, explanation accuracy for the sample with the highest weight in memory when it is an explanation by example.
    """
    model.eval()
    expl_max = 0

    top_counter = 0
    correct_counter = 0
    top_example = 0
    correct_example = 0

    with torch.no_grad():
        for index_batch, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            memory, _ = next(iter(mem_loader))
            memory = memory.to(device)

            # get output
            output, rw = model(data,memory, return_weights=True)
            pred = output.data.max(1, keepdim=True)[1]
            
            #  auxiliar memory to get memory output
            aux_mem, _ = next(iter(mem_loader))
            aux_mem = aux_mem.to(device)

            # get memory output
            exp_output  = model(memory,aux_mem)
            exp_pred = exp_output.data.max(1,keepdim=True)[1]
            
            # get index of sample with highest weight
            _, index_max = torch.max(rw,dim=1)
            sorted_exp_pred = exp_pred[index_max]
            for row in range(len(sorted_exp_pred)):
                if sorted_exp_pred[row] != pred[row]:
                    # counterfactual
                    correct_counter += pred[row].eq(target[row].data.view_as(pred[row])).sum().item()
                    top_counter+=1
                else:
                    # explanation by example
                    correct_example += pred[row].eq(target[row].data.view_as(pred[row])).sum().item()
                    top_example+=1
            # explanation accuracy
            expl_max += pred.eq(exp_pred[index_max].data.view_as(pred)).sum().item()
            print("batch:{}\{}".format(index_batch,len(loader)),end='\r')
        counter_accuracy = 100.*(torch.true_divide(correct_counter,top_counter))
        example_accuracy = 100.*(torch.true_divide(correct_example,top_example))
        explanation_accuracy = 100.*(torch.true_divide(expl_max,len(loader.dataset)))

    return explanation_accuracy, counter_accuracy, example_accuracy

def run_evaluation(path:str,dataset_dir:str, num_runs:int=5):
    """ Function to print the explanation accuracy of a set of models inside a dir.  It prints the mean and standard deviation of explanation accuracy of the top sample in memory on different settings (see paper).

    Args:
        path (str): dir path
        dataset_dir (str): dir where datasets are stored
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))

    # load model
    expl_acc = []
    expl_acc_counter = []
    expl_acc_ex = []
    list_models = [name_file for name_file in os.listdir(path) if name_file.endswith('.pt')]
    for indx, name_model in enumerate(list_models):

        # load model
        checkpoint = torch.load(path+name_model, map_location=device)
        model_name = checkpoint['model_name']
        modality = checkpoint['modality']
        dataset_name = checkpoint['dataset_name']
        load_dataset = getattr(datasets, 'get_'+dataset_name)

        model = utils.get_model(model_name,checkpoint['num_classes'],model_type=modality)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        #load data
        _, _, test_loader, mem_loader = load_dataset(dataset_dir,batch_size_train=checkpoint['train_examples'], batch_size_test=500,batch_size_memory=100)
        print("Loaded models:{}/{}".format(indx+1,len(list_models),end='\r'))


        # perform validation
        cum_acc =  []
        cum_acc_counter = []
        cum_acc_ex = []
        for index_run in range(num_runs):
            print(f"Model:{name_model}\tRun:{index_run+1}/{num_runs}")
            exp_max_acc, counter_acc,example_acc = get_explanation_accuracy(model,test_loader,mem_loader,device)
            cum_acc.append(exp_max_acc)
            cum_acc_counter.append(counter_acc)
            cum_acc_ex.append(example_acc)

        # store mean of model's accuracies
        expl_acc.append(np.mean(cum_acc))
        expl_acc_counter.append(np.mean(cum_acc_counter))
        expl_acc_ex.append(np.mean(cum_acc_ex))

        # log for each model
        print("Explanation accuracy max (mean):{:.2f}\t(std_dev):{:.2f}\t  counterfactual mean:{:.2f}\t  counterfactual std:{:.2f}\t  example mean:{:.2f}\t  example std:{:.2f}".format(np.mean(expl_acc),np.std(expl_acc),np.mean(expl_acc_counter),np.std(expl_acc_counter),np.mean(expl_acc_ex),np.std(expl_acc_ex)))

    # final summary
    print()
    print("Explanation accuracy (mean):{:.2f}\t(std_dev):{:.2f}\t  counterfactual acc mean:{:.2f}\t  counterfactual std:{:.2f}\t  example acc mean:{:.2f}\t  example std:{:.2f}".format(np.mean(expl_acc),np.std(expl_acc),np.mean(expl_acc_counter),np.std(expl_acc_counter),np.mean(expl_acc_ex),np.std(expl_acc_ex)))


def main(args):

    run_evaluation(FLAGS.path_models,FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)