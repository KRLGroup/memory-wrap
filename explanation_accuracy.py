import datasets
import torch # type: ignore
import numpy as np
import random
import absl.flags
import absl.app
import os
import aux

# user flags
absl.flags.DEFINE_string("path_models", None, "Path of the trained model")
absl.flags.DEFINE_string("dataset", None, "Dataset to test (SVHN, CIFAR10 or CIFAR100)")
absl.flags.DEFINE_string("modality", None, "Fixed or random")

absl.flags.mark_flag_as_required("dataset")
absl.flags.mark_flag_as_required("path_models")
absl.flags.mark_flag_as_required("modality")

FLAGS = absl.flags.FLAGS

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance

# fixed seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

def eval_memory(model,loader,mem_loader,device):
    model.eval()
    expl_max = 0

    top_counter = 0
    correct_counter = 0
    top_example = 0
    correct_example = 0

    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
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

        counter_accuracy = 100.*(torch.true_divide(correct_counter,top_counter))
        example_accuracy = 100.*(torch.true_divide(correct_example,top_example))
        explanation_accuracy = 100.*(torch.true_divide(expl_max,len(loader.dataset)))

    return explanation_accuracy, counter_accuracy, example_accuracy

def run_evaluation(path, dataset_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    
    # load data   
    load_dataset = getattr(datasets, 'get_'+dataset_name)

    # load model
    expl_acc = []
    expl_acc_counter = []
    expl_acc_ex = []
    list_models = [name_file for name_file in os.listdir(path) if name_file.endswith('.pt')]
    for indx, name_model in enumerate(list_models):

        # load model
        checkpoint = torch.load(path+name_model)
        model_name = checkpoint['model_name']
        model = aux.get_model(model_name,checkpoint['num_classes'],model_type=FLAGS.modality)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        #load data
        _, _, test_loader, mem_loader = load_dataset('../datasets',batch_size_train=checkpoint['train_examples'], batch_size_test=500,batch_size_memory=100)
        print("Loaded models:{}/{}".format(indx+1,len(list_models),end='\r'))


        # perform validation
        cum_acc =  []
        cum_acc_counter = []
        cum_acc_ex = []
        for _ in range(10):
            exp_max_acc, counter_acc,example_acc = eval_memory(model,test_loader,mem_loader,device)
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


def main():

    run_evaluation(FLAGS.path_models,FLAGS.dataset)

if __name__ == '__main__':
  absl.app.run(main)