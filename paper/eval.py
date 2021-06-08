import absl.flags
import absl.app
import torch # type: ignore
import numpy as np
import datasets
import aux

# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_string("dir_dataset", 'datasets/', "dir path where datasets are stored")
absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS


def run_evaluation(path:str,dataset_dir:str):
    """ Function to evaluate a single model.
    Args:
        path (str): model path
        dataset_dir (str): dir where datasets are stored
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    # load model
    checkpoint = torch.load(path)

    model_name = checkpoint['model_name']
    modality = checkpoint['modality']
    model = aux.get_model(model_name,checkpoint['num_classes'],model_type=modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # load data
    dataset_name = checkpoint['dataset_name']
    mem_examples = checkpoint['mem_examples']
    train_examples = checkpoint['train_examples']
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    _, _, test_loader, mem_loader = load_dataset(dataset_dir,batch_size_train=train_examples, batch_size_test=500,batch_size_memory=mem_examples,size_train=train_examples)
    
    
    # perform validation
    loss_criterion = torch.nn.CrossEntropyLoss()
    if modality == 'std':
        best_acc, best_loss = aux.eval_std(model,test_loader,loss_criterion,device)
        print("Loss:{:.4f} | Accuracy {:.2f} ".format(best_loss,best_acc))
    else:
        cum_acc =  []
        for _ in range(10):
            best_acc, best_loss = aux.eval_memory(model,test_loader,mem_loader,loss_criterion,device)
            cum_acc.append(best_acc)
        best_acc = np.mean(cum_acc)
        print("Best Loss:{:.4f} | Accuracy {:.2f}".format(best_loss,best_acc))



def main(argv=None):

    run_evaluation(FLAGS.path_model,FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)