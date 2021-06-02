
import torch # type: ignore
import numpy as np
import absl.flags
import absl.app
import os
import datasets
import aux 
import time

# user flags
absl.flags.DEFINE_string("path", None, "dir path where models are stored")
absl.flags.mark_flag_as_required("path")
FLAGS = absl.flags.FLAGS

   

def run_experiment(path:str):
    """ Function to evaluate a set of models inside a dir.
    It prints mean and standard deviation.

    Args:
        path (str): dir path
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    loss_criterion = torch.nn.CrossEntropyLoss()
    list_models = [name_file for name_file in os.listdir(path) if name_file.endswith('.pt')]
    run_acc = []
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
            init_eval_time = time.time()
            for _ in range(5):
                test_acc, _ = aux.eval_memory(model,test_loader, mem_loader,loss_criterion,device)
                cum_acc.append(test_acc)
            acc_mean = np.mean(cum_acc)
            end_eval_time = time.time()
        else:
            init_eval_time = time.time()
            acc_mean, _  = aux.eval_std(model,test_loader,loss_criterion,device)
            end_eval_time = time.time()

        # stats
        run_acc.append(acc_mean)


        # log
        print("Run:{} | Accuracy {:.2f} | Mean Accuracy:{:.2f} | Std Dev:{:.2f} |  \tE:{:.2f}min".format(run+1,acc_mean, np.mean(run_acc), np.std(run_acc),(end_eval_time -init_eval_time)/60))







def main(argv=None):

    run_experiment(FLAGS.path)

if __name__ == '__main__':
  absl.app.run(main)