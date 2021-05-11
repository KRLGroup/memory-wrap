import absl.flags
import absl.app
import torch # type: ignore
import numpy as np
import random
import datasets
import aux

# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_string("dataset", None, "Dataset to test (SVHN or CIFAR10)")
absl.flags.DEFINE_string("modality", None, "Fixed or random")

absl.flags.mark_flag_as_required("dataset")
absl.flags.mark_flag_as_required("path_model")
absl.flags.mark_flag_as_required("modality")

FLAGS = absl.flags.FLAGS

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)


def run_evaluation(path, dataset_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    
    # load model
    checkpoint = torch.load(path)
    model_name = checkpoint['model_name']
    model = aux.get_model(model_name,checkpoint['num_classes'],model_type=FLAGS.modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # load data
    mem_examples = checkpoint['mem_examples']
    train_examples = checkpoint['train_examples']
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    _, _, test_loader, mem_loader = load_dataset('../datasets',batch_size_train=train_examples, batch_size_test=500,batch_size_memory=mem_examples,size_train=train_examples)
    
    
    # perform validation
    loss_criterion = torch.nn.CrossEntropyLoss()
    if FLAGS.modality == 'std':
        best_acc, best_loss = aux.eval_std(model,test_loader,loss_criterion,device)
        print("Best Loss:{:.4f} | Accuracy {:.2f} ".format(best_loss,best_acc))
    else:
        cum_acc =  []
        for _ in range(10):
            best_acc, best_loss = aux.eval_memory(model,test_loader,mem_loader,loss_criterion,device)
            cum_acc.append(best_acc)
        best_acc = np.mean(cum_acc)
        print("Best Loss:{:.4f} | Accuracy {:.2f}".format(best_loss,best_acc))



def main():

    run_evaluation(FLAGS.path_model,FLAGS.dataset)

if __name__ == '__main__':
  absl.app.run(main)