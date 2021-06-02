
import torch # type: ignore
import numpy as np
import absl.flags
import absl.app
import os
import datasets
import aux
import time
from typing import Tuple
# user flags
absl.flags.DEFINE_string("path", None, "dir path where models are stored")
absl.flags.mark_flag_as_required("path")
FLAGS = absl.flags.FLAGS


def major_voting_baseline(model:torch.nn.Module,loader:torch.utils.data.DataLoader,mem_loader:torch.utils.data.DataLoader,loss_criterion:torch.nn.modules.loss,device:torch.device)-> Tuple[float, float,float]:
    """ Method to generate the major voting baseline of a pytorch model that has
    a Memory Wrap variant as last layer. The model takes an image and a memory
    samples as input and return the logits.

    Args:
        model (torch.nn.Module): Trained model to evaluate
        loader (torch.utils.data.DataLoader): loader containing testing
            samples where evaluate the model
        mem_loader (torch.utils.data.DataLoader): loader containing training
            samples to be used as memory set
        loss_criterion (torch.nn.modules.loss): loss to use to evaluate the
            error
        device (torch.device): device where the model is stored

    Returns:
        Tuple[float, float,float]: Accuracy of Memory Wrap, accuracy of major
        voting algorithm using the predictions, accuracy of major voting
        algorithm using the targets
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    correct_mvo = 0
    correct_mvy = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            memory, y = next(iter(mem_loader))
            memory = memory.to(device)
         
            aux_memory, _ =  next(iter(mem_loader))
            aux_memory = aux_memory.to(device)
            y = y.to(device)

            output,rw  = model(data,memory,return_weights=True)
            loss = loss_criterion(output, target) 
            pred = output.data.max(1, keepdim=True)[1]
            # compute memory outputs
            memory_outputs = model(memory,aux_memory)
            _, memory_predictions = torch.max(memory_outputs, 1)
            mem_val,memory_sorted_index = torch.sort(rw,descending=True)

            for ind in range(len(data)):
                # M_c u M_e : set of sample with a positive impact on prediction
                m_ec = memory_sorted_index[ind][mem_val[ind]>0]
                pred_mec = memory_predictions[m_ec]
                y_mec = y[m_ec]
                mv_output, _ = torch.mode(pred_mec)
                mv_y, _ = torch.mode(y_mec)
                
                correct_mvo += mv_output.eq(target[ind].data.view_as(mv_output)).sum().item()
                correct_mvy += mv_y.eq(target[ind].data.view_as(mv_y)).sum().item()
            
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss += loss.item()
            
        
        test_accuracy = 100.*(torch.true_divide(correct,len(loader.dataset)))
        mvo_accuracy = 100.*(torch.true_divide(correct_mvo,len(loader.dataset)))
        mvy_accuracy = 100.*(torch.true_divide(correct_mvy,len(loader.dataset)))
    return test_accuracy,  mvo_accuracy, mvy_accuracy
    

def run_experiment(path:str):
    """ Function to evaluate the major voting baseline in a set of models inside
    a dir. It prints mean and standard deviation.

    Args:
        path (str): dir path
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    loss_criterion = torch.nn.CrossEntropyLoss()
    list_models = [name_file for name_file in os.listdir(path) if name_file.endswith('.pt')]
    run_acc = []
    run_mvo = []
    run_mvy = []
    for _, name_model in enumerate(sorted(list_models)):

        # load model
        run,_ = name_model.split('.')
        run = int(run)-1
        aux.set_seed(run)
        checkpoint = torch.load(path+name_model)
        #   load model
        modality = checkpoint['modality']

        model_name = checkpoint['model_name']
        model = aux.get_model(model_name,checkpoint['num_classes'],model_type=modality)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # load data
        mem_examples = checkpoint['mem_examples']
        train_examples = checkpoint['train_examples']
        load_dataset = getattr(datasets, 'get_'+checkpoint['dataset_name'])
        _, _, test_loader, mem_loader = load_dataset('../datasets',batch_size_train=128, batch_size_test=500,batch_size_memory=mem_examples,size_train=train_examples,seed=run)

        if modality == 'memory' or modality == 'encoder_memory':
            cum_acc =  []
            cum_mvo_acc = []
            cum_mvy_acc = []
            init_eval_time = time.time()
            for _ in range(5):
                test_acc,  mvo_accuracy, mvy_accuracy = major_voting_baseline(model,test_loader, mem_loader,loss_criterion,device)
                cum_acc.append(test_acc)
                cum_mvo_acc.append(mvo_accuracy)
                cum_mvy_acc.append(mvy_accuracy)
            acc_mean = np.mean(cum_acc)
            mvo_mean = np.mean(cum_mvo_acc)
            mvy_mean = np.mean(cum_mvy_acc)
            end_eval_time = time.time()
            run_mvo.append(mvo_mean)
            run_mvy.append(mvy_mean)
        else:
            init_eval_time = time.time()
            acc_mean, _  = aux.eval_std(model,test_loader,loss_criterion,device)
            end_eval_time = time.time()

        # stats
        run_acc.append(acc_mean)


        # log
        print("Run:{} | Accuracy {:.2f} | Mean Accuracy:{:.2f} | Std Dev:{:.2f} | MVO Accuracy:{:.2f} | Std Dev:{:.2f} | MVY Accuracy:{:.2f} | Std Dev:{:.2f} \tE:{:.2f}min".format(run+1,acc_mean, np.mean(run_acc), np.std(run_acc),np.mean(run_mvo), np.std(run_mvo),np.mean(run_mvy), np.std(run_mvy),(end_eval_time -init_eval_time)/60))







def main(argv=None):

    run_experiment(FLAGS.path)

if __name__ == '__main__':
  absl.app.run(main)