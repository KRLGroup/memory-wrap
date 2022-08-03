import sys
import os

import torch
import numpy as np
import absl.flags
import absl.app
from sklearn.neighbors import KNeighborsClassifier

sys.path.append('..')
import utils.datasets as datasets
import utils.utils as aux
from wrappers import densenet, googlenet, mobilenet, resnet, shufflenet
from wrappers import efficientnet

# user flags
absl.flags.DEFINE_string("dir_models", None,
                         "dir path where models are stored")
absl.flags.DEFINE_string("dir_dataset", '../datasets/',
                         "dir path where datasets are stored")
absl.flags.DEFINE_enum("metric", None, ['input', 'prediction'],
                       "Metric to be used")
absl.flags.mark_flag_as_required("dir_models")
absl.flags.mark_flag_as_required("metric")
FLAGS = absl.flags.FLAGS


def get_ebe_index(sorted_indices, pred_v, pred_t):
    sorted_pred = pred_v[sorted_indices]
    in_memory = (sorted_pred == pred_t.unsqueeze(dim=1).repeat(
                        1, sorted_indices.shape[1]))
    rows_with_ebe = in_memory.any(1)
    if rows_with_ebe.sum() == 0:
        print("problema con i dati ! ebe function")
        exit()
    ebes = in_memory.double().argmax(1)
    index_of_ebe = torch.gather(sorted_indices, 1, ebes.unsqueeze(1)).squeeze()
    return index_of_ebe


def get_model_wrapper(model_name: str, num_classes: int,
                      model_type: str) -> torch.nn.Module:
    """ Utility function to retrieve the requested model.

    Args:
        model_name (str): One of ['resnet18, efficientnet , mobilenet,
            shufflenet, googlenet, densenet]
        num_classes (int): Number of output units.
        model_type (str): Specify the model variant. 'std' is the standard
            model, 'memory' is the baseline that uses only the memory and
            'encoder_memory' is Memory Wrap

    Raises:
        ValueError: if model_name is not one of ['resnet18, efficientnet,
            mobilenet, shufflenet, googlenet, densenet]
        ValueError: if model_type is not one of ['std', 'memory',
            'encoder_memory']

    Returns:
        torch.nn.Module: PyTorch model
    """
    if model_type not in ['memory', 'encoder_memory', 'std']:
        raise ValueError(f"modality (model type) must be one of "
                         f"[\'memory\',\'encoder_memory\',\'std\'],"
                         f"not {model_type}.")
    if model_name == 'efficientnet':
        if model_type == 'memory':
            model = efficientnet.MemoryEfficientNetB0(num_classes)
        elif model_type == 'encoder_memory':
            model = efficientnet.EncoderMemoryEfficientNetB0(num_classes)
        else:
            model = efficientnet.EfficientNetB0(num_classes)
    elif model_name == 'resnet18':
        if model_type == 'memory':
            model = resnet.MemoryResNet18()
        elif model_type == 'encoder_memory':
            model = resnet.EncoderMemoryResNet18()
        else:
            model = resnet.ResNet18()
    elif model_name == 'shufflenet':
        if model_type == 'memory':
            model = shufflenet.MemoryShuffleNetV2(net_size=0.5)
        elif model_type == 'encoder_memory':
            model = shufflenet.EncoderMemoryShuffleNetV2(net_size=0.5)
        else:
            model = shufflenet.ShuffleNetV2(net_size=0.5)
    elif model_name == 'densenet':
        if model_type == 'memory':
            model = densenet.memory_densenet_cifar()
        elif model_type == 'encoder_memory':
            model = densenet.encoder_memory_densenet_cifar()
        else:
            model = densenet.densenet_cifar()
    elif model_name == 'googlenet':
        if model_type == 'memory':
            model = googlenet.MemoryGoogLeNet()
        elif model_type == 'encoder_memory':
            model = googlenet.EncoderMemoryGoogLeNet()
        else:
            model = googlenet.GoogLeNet()
    elif model_name == 'mobilenet':
        if model_type == 'memory':
            model = mobilenet.MemoryMobileNetV2(num_classes)
        elif model_type == 'encoder_memory':
            model = mobilenet.EncoderMemoryMobileNetV2(num_classes)
        else:
            model = mobilenet.MobileNetV2(num_classes)
    else:
        raise ValueError("Error: input model name is not valid!")
    return model


def run_experiment(path: str, dataset_dir: str, metric: str):
    """ Function to evaluate the major voting baseline in a set of models
        inside a dir. It prints mean and standard deviation.

    Args:
        path (str): dir path
        dataset_dir (str): dir where datasets are stored
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Device:{device}")
    list_models = [name_file for name_file in os.listdir(path)
                   if name_file.endswith('.pt')]

    run_chpt1_differences = []
    run_mem_differences = []
    run_dknn1_differences = []
    run_random_differences = []
    for _, name_model in enumerate(sorted(list_models)):

        # load model
        run, _ = name_model.split('.')
        run = int(run)-1
        aux.set_seed(run)
        checkpoint = torch.load(path+name_model)
        # load model
        modality = checkpoint['modality']

        model_name = checkpoint['model_name']
        model = get_model_wrapper(model_name, checkpoint['num_classes'],
                                  model_type=modality)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # load data
        mem_examples = checkpoint['mem_examples']
        train_examples = checkpoint['train_examples']
        load_dataset = getattr(datasets, 'get_'+checkpoint['dataset_name'])
        train_loader, _, test_loader, mem_loader = load_dataset(
                dataset_dir, batch_size_train=256, batch_size_test=256,
                batch_size_memory=mem_examples, size_train=train_examples,
                seed=run)

        model.eval()

        # Final CNN weights for computing twin contributions
        if modality == 'std':
            weights = model.linear.weight
        else:
            weights = model.mw.classifier.fc2.weight

        train_conts = list()
        train_preds = list()
        train_encoded = list()
        train_inputs = list()

        if metric == 'input':
            loss_criterion = torch.nn.L1Loss()
        else:
            loss_criterion = torch.nn.CrossEntropyLoss()

        # GLOBAL
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                imgs, _ = data
                imgs = imgs.to(device)
                if modality == 'std':
                    logits, encoded_imgs = model(imgs)
                else:
                    memory, _ = next(iter(mem_loader))
                    memory = memory.to(device)
                    logits, encoded_imgs = model(imgs, memory)
                pred = torch.argmax(logits, axis=1)

                x_cont = weights[pred] * encoded_imgs

                train_conts.extend(x_cont)
                train_preds.extend(pred)
                train_encoded.extend(encoded_imgs)
                train_inputs.extend(imgs)
        train_inputs = torch.stack(train_inputs)
        train_encoded = torch.stack(train_encoded)
        train_conts = torch.stack(train_conts)
        train_preds = torch.stack(train_preds)

        # test knn
        chpt1_differences = []
        mem_differences = []
        dknn1_differences = []
        random_difference = []
        tot = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # reset the data to train k-NN
                train_conts = list()
                train_preds = list()
                imgs, _ = data
                imgs = imgs.to(device)
                tot += len(imgs)
                if modality == 'std':
                    # launch an exception if the model is not a std model
                    raise Exception('The model is not a std model')
                else:
                    memory, _ = next(iter(mem_loader))
                    memory = memory.to(device)
                    logits, encoded_imgs, memory_weights = model(
                        imgs, memory, return_weights=True)
                pred = torch.argmax(logits, axis=1)

                # fit knn using only memory samples
                aux_memory, _ = next(iter(mem_loader))
                aux_memory = aux_memory.to(device)
                exp_logits, exp_x = model(memory, aux_memory)
                exp_pred = torch.argmax(exp_logits, axis=1)

                x_cont = weights[exp_pred] * exp_x
                chp_kNN = KNeighborsClassifier(n_neighbors=1,
                                               algorithm="brute")
                dkNN = KNeighborsClassifier(n_neighbors=1,
                                            algorithm="brute")
                chp_kNN.fit(x_cont.cpu().numpy(), exp_pred.cpu().numpy())
                dkNN.fit(exp_x.cpu().numpy(), exp_pred.cpu().numpy())
                n_neighbors = mem_examples

                x_cont = weights[pred] * encoded_imgs

                p_nns_chp = chp_kNN.kneighbors(x_cont.cpu().numpy(),
                                               n_neighbors=n_neighbors,
                                               return_distance=False)
                p_nns_dknn = dkNN.kneighbors(encoded_imgs.cpu().numpy(),
                                             n_neighbors=n_neighbors,
                                             return_distance=False)

                p_nns_chp = torch.tensor(p_nns_chp).squeeze().to(device)
                p_nns_dknn = torch.tensor(p_nns_dknn).squeeze().to(device)

                # get index of sample with highest weight
                sorted_memory_weights_idx = torch.argsort(memory_weights,
                                                          descending=True)

                # Memory Explanation By Examples (ebe)
                index_of_ebe = get_ebe_index(sorted_memory_weights_idx,
                                             exp_pred, pred)
                log_mem = exp_logits[index_of_ebe]

                # CHP Explanation By Examples (ebe)
                index_of_ebe = get_ebe_index(p_nns_chp, exp_pred, pred)
                log_chp1 = exp_logits[index_of_ebe]

                # dKNN Explanation By Examples (ebe)
                index_of_ebe = get_ebe_index(p_nns_dknn, exp_pred, pred)
                log_dknn1 = exp_logits[index_of_ebe]

                # Random Explanation By Examples (ebe)
                random_neighbors = torch.stack(
                    [torch.randperm(len(p_nns_chp[row])) for row
                        in range(len(p_nns_chp))]).to(device)
                index_of_ebe = get_ebe_index(random_neighbors, exp_pred, pred)
                log_random = exp_logits[index_of_ebe]

                # Targets
                if metric == 'input':
                    considered_pred = logits
                else:
                    considered_pred = pred

                # Compute scores
                mem_differences.append(
                    loss_criterion(log_mem, considered_pred).item())
                chpt1_differences.append(
                    loss_criterion(log_chp1, considered_pred).item())
                dknn1_differences.append(
                    loss_criterion(log_dknn1, considered_pred).item())
                random_difference.append(
                    loss_criterion(log_random, considered_pred).item())

                # Log stuff
                print(f"({i / len(test_loader):.2f})\t\t"
                      f"chp:{np.mean(chpt1_differences):.4f}\t"
                      f"mem:{np.mean(mem_differences):.4f}\t"
                      f"knn*:{np.mean(dknn1_differences):.4f}\t"
                      f"\rand:{np.mean(random_difference):.4f}", end='\r')

        # Log stuff
        run_chpt1_differences.append(np.mean(chpt1_differences))
        run_mem_differences.append(np.mean(mem_differences))
        run_dknn1_differences.append(np.mean(dknn1_differences))
        run_random_differences.append(np.mean(random_difference))

        print(f"Run:{run+1} | "
              f"CHP {np.mean(run_chpt1_differences):.4f} "
              f"Std Dev:{np.std(run_chpt1_differences):.4f} | "
              f"KNN*:{np.mean(run_dknn1_differences):.4f} "
              f"Std Dev:{np.std(run_dknn1_differences):.4f}  | "
              f"MEM:{np.mean(run_mem_differences):.4f} "
              f"Std Dev:{np.std(run_mem_differences):.4f} | "
              f"RANDOM {np.mean(run_random_differences):.4f}  "
              f"Std Dev:{np.std(run_random_differences):.4f}")


def main(argv=None):
    run_experiment(FLAGS.dir_models, FLAGS.dir_dataset, FLAGS.metric)


if __name__ == '__main__':
    absl.app.run(main)
