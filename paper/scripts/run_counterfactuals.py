import os
import sys
import time

sys.path.append('..')

import torch
import numpy as np
import torch.nn as nn
import absl.flags
import absl.app

from architectures.autoencoder import AutoEncoder
import utils.datasets as datasets
import utils.utils as aux
import utils.counterfactuals_utils as counterfactuals_utils


FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("path_models", None, "Path of the trained model")
absl.flags.DEFINE_string("path_aes", None, "Path of the trained autoencoder model")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "Path of the trained autoencoder model")
absl.flags.DEFINE_enum("algo", None, ['memory', 'proto'], "Algorithm for computing counterfactuals")
absl.flags.mark_flag_as_required("path_models")
absl.flags.mark_flag_as_required("algo")


def get_counterfactuals_indices(sorted_indices:torch.Tensor, predictions:torch.Tensor,
                                targets:torch.Tensor) -> torch.Tensor:
    """
    Get the indices of the counterfactuals in memory.

    Given a sorted indices tensor, the predictions and the targets, this function returns
    the indices of the counterfactuals in memory. Namely, it extracts the indices from the
    sorted indices for which the prediction is different from the target.

    Args:
        sorted_indices (torch.Tensor): List of indices sorted by weights.
        predictions (torch.Tensor): List of predictions.
        targets (torch.Tensor): List of targets.
    Returns:
        torch.Tensor: List of indices of the counterfactuals in memory.
    """
    sorted_pred = predictions[sorted_indices]
    in_memory = (sorted_pred != targets.unsqueeze(dim=1).repeat(1,sorted_indices.shape[1]))
    rows_with_ebe = in_memory.any(1)
    if rows_with_ebe.sum() == 0:
        raise Exception("No ebe found! Check the predictions and targets.")
    counterfactuals_indices = in_memory.double()
    return counterfactuals_indices


def get_prototypes_from_memorynet(model: torch.nn.Module, enc_model: torch.nn.Module,
            memory: torch.Tensor,  auxiliary_memory: torch.Tensor) -> None:
    """
    Compute the prototypes from a network with Memory Wrap.

    Args:
        model (torch.nn.Module): Model with Memory Wrap.
        enc_model (torch.nn.Module): Autoencoder model.
        memory (torch.Tensor): Memory.
        auxiliary_memory (torch.Tensor): Auxiliary memory.
    Returns:
        class_prototypes (torch.Tensor): Prototypes for each class.

    """
    # get prediction and encoding for the whole dataset
    if enc_model is not None:
        enc_data = enc_model(memory)
        logits = model(memory,auxiliary_memory)
    else:
        logits, enc_data = model(memory,auxiliary_memory)
    preds = torch.argmax(logits, axis=1)
    num_classes = logits.shape[1]

    # compute prototypes
    class_prototypes = []
    for label in range(num_classes):
        class_prototypes.append(enc_data[preds == label].mean(dim=0))
    class_prototypes = torch.stack(class_prototypes, dim=0)

    return class_prototypes



def get_closest_prototypes(model, input_data, class_prototypes, class_encodings,
            k=None, k_type='mean', target_class=None):
    """
    Return the closest prototype to the input in the target class list.

    Args:
        model (torch.nn.Module): Model with Memory Wrap.
        input_data (torch.Tensor): Batch Input data.
        class_prototypes (torch.Tensor): Prototypes for each class.
        class_encodings (torch.Tensor): Encodings for each class.
        k (int): Number of closest prototypes to return.
        k_type (str): Type of the k-nearest neighbors.
        target_class (int): Target class.
    Returns:
        torch.Tensor: Closest prototypes.
    """

    dist_proto = {}
    encoded_input = model(input_data)

    class_dict = class_prototypes if k is None else class_encodings
    selected_proto = []
    for _ in range(encoded_input.shape[0]):
        for class_proto, prototype in enumerate(class_dict):
            if class_proto not in target_class:
                continue
            if k is None:
                dist_proto[class_proto] = torch.linalg.norm(encoded_input - prototype)
            elif k is not None:
                dist_k = torch.linalg.norm(encoded_input.reshape(encoded_input.shape[0], -1) -
                                        prototype.reshape(prototype.shape[0], -1), axis=1)
                idx = torch.argsort(dist_k)[:k]
                if k_type == 'mean':
                    dist_proto[class_proto] = torch.mean(dist_k[idx])
                else:
                    dist_proto[class_proto] = dist_k[idx[-1]]

                class_prototypes[class_proto] = torch.unsqueeze(
                    torch.mean(prototype[idx], dim=0), dim=0)

        id_proto = min(dist_proto, key=dist_proto.get)  # type: ignore[type-var]

        selected_proto.append(class_prototypes[id_proto])  # type: ignore[type-var]
    return torch.stack(selected_proto)


def run_proto_cf(path_models:str, path_aes:str, path_dataset:str):
    """
    Main function. Run the counterfactuals generation and print the scores.

    Args:
        path_models (str): Path of the trained models.
        path_aes (str): Path of the trained autoencoder models for the path_models.
        path_dataset (str): Path where the dataset is stored.
    """
    # load name of models in the path_models dir
    list_models = [name_file for name_file in os.listdir(path_models) if name_file.endswith('.pt')]

    # lists to store the scores of each run
    runs_scores_im1 = []
    runs_scores_iim1 = []
    runs_scores_im2 = []

    for index_model, name_model in enumerate(list_models):

        # set seed for reproducibility using the index of the model as seed
        seed = int(name_model[:-3])-1
        aux.set_seed(seed)

        ####################################################
        ######### load classification model ################
        ####################################################
        checkpoint = torch.load(path_models+name_model)
        model_type = checkpoint['modality']
        model = aux.get_model(checkpoint['model_name'], checkpoint['num_classes'],
                    model_type=model_type)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
        model.eval()
        print(f"Loaded models:{index_model+1}/{len(list_models)}",end='\r')

        ####################################################
        ################ load dataset  #####################
        ####################################################
        dataset_dir = path_dataset
        dataset_name = checkpoint['dataset_name']
        load_dataset = getattr(datasets, 'get_'+dataset_name)

        # note that fulldataset_loader returns the full dataset at each iteration
        dataset_size = checkpoint['train_examples']
        training_loader,_,  test_loader, mem_loader = load_dataset(dataset_dir,
                batch_size_train = 128, size_train = dataset_size, batch_size_test = 64,
                batch_size_memory = checkpoint['mem_examples'], seed=seed)
        # sanity check
        print("loaded model")

        ####################################################
        ################ load autoencoders  ################
        ####################################################
        print("loading autoencoder")

        # Autoencoder trained using the full dataset
        ae_fulldataset = AutoEncoder(num_channels=3)
        ae_fulldataset.load_state_dict(torch.load(f"{path_aes}/{seed}/full.pt"))
        ae_fulldataset = ae_fulldataset.cuda()
        ae_fulldataset.eval()

        # List of autoencoders trained using only one class at time
        class_specific_aes = []
        for idx_class in range(10):
            class_specific_aes.append(AutoEncoder(num_channels=3))
            class_specific_aes[idx_class].load_state_dict(torch.load(
               f"{path_aes}/{seed}/class_{idx_class}.pt"))
            class_specific_aes[idx_class] = class_specific_aes[idx_class].cuda()
            class_specific_aes[idx_class].eval()

        ####################################################
        ################ compute counterfactuals  ##########
        ####################################################

        # compute feature range
        min_feat = torch.tensor(float('inf'))
        max_feat = torch.tensor(float('-inf'))

        for _, data in enumerate(training_loader):
            batch, _ = data
            min_feat = torch.minimum(torch.min(batch), min_feat)
            max_feat = torch.maximum(torch.max(batch), max_feat)
        feature_range = (min_feat, max_feat)


        # compute prototypes for each class
        if FLAGS.algo == 'proto':
            memory_input, _ = next(iter(mem_loader))
            fulldataset_loader,_,  _, _ = load_dataset(dataset_dir,
                        batch_size_train = checkpoint['train_examples'],
                        size_train = checkpoint['train_examples'],
                        batch_size_test = 1,
                        batch_size_memory = checkpoint['mem_examples'],
                        seed=seed)
            class_prototypes = get_prototypes_from_memorynet(model=model.cpu(),
                        enc_model =ae_fulldataset.encoder.cpu(),
                        memory=next(iter(fulldataset_loader))[0],
                        auxiliary_memory=memory_input)

            del fulldataset_loader
        ae_fulldataset = ae_fulldataset.cuda()
        model = model.cuda()

        # hyper parameters (default values of aliba)
        classes = 10
        k = None # number of neighbors
        target_class = None
        k_type = None


        # logging stuff
        print("Computing counterfactuals...")
        samples_im1_scores = []
        samples_iim1_scores = []
        samples_im2_scores = []
        starting_time = time.time()

        for batch_idx, data in enumerate(test_loader):

            ####################################################
            ################ Input  ############################
            ####################################################
            img_cpu, _ = data # keep copy on the cpu to compute closest prototype
            memory_input, _ = next(iter(mem_loader))

            # move to GPU
            img_gpu = img_cpu.cuda()
            memory_input = memory_input.cuda()

            ####################################################
            ############ Input Prediction ######################
            ####################################################
            # get output
            with torch.no_grad():
                logits, weights = model(img_gpu,memory_input, return_weights=True)

                # get predicted class and its one-hot encoding
                predicted_class = torch.argmax(logits, axis=1)
                onehot_prediction = nn.functional.one_hot(predicted_class, num_classes=classes)

                ####################################################
                ############ Memory Predictions ####################
                ####################################################
                # use memory as input and get predictions
                aux_memory, _ = next(iter(mem_loader))
                aux_memory = aux_memory.cuda()
                memory_logits = model(memory_input,aux_memory)
                memory_predictions =  torch.argmax(memory_logits, axis=1)


            ####################################################
            ########## Memory Counterfactuals ##################
            ####################################################
            # find counterfactuals candidates among the memory samples
            sorted_weights, sorted_weights_indices = torch.sort(weights,descending=True)
            counterfactuals_candidates = get_counterfactuals_indices(sorted_weights_indices,
                        memory_predictions, predicted_class)

            if len(counterfactuals_candidates) == 0:
                # rare case of no candidates in memory
                continue

            # count how many candidates have a weight greater then zero
            count_counterfactuals = torch.count_nonzero(counterfactuals_candidates*sorted_weights,1)

            index_samples_with_cf = count_counterfactuals.nonzero()
            samples_with_cf = (counterfactuals_candidates*sorted_weights)[index_samples_with_cf]


            if samples_with_cf.size()[0] > 1:

                # extract the counterfactual associated with the highest weight
                index_highest_weight= torch.argmax(samples_with_cf,2)
                index_top_counterfactual = sorted_weights_indices[0,index_highest_weight]

                prediction_top_counterfactual = memory_predictions[index_top_counterfactual]


                target_class = prediction_top_counterfactual



                if FLAGS.algo == 'proto':
                    with torch.no_grad():
                        prototype = get_closest_prototypes(
                            ae_fulldataset.encoder.cpu(),
                            img_gpu[torch.flatten(index_samples_with_cf)].cpu(),
                            class_prototypes, None, k, k_type, target_class)
                        prototype = prototype.cuda()
                    ae_fulldataset =  ae_fulldataset.cuda()
                    counterfactual = counterfactuals_utils.attack(model,
                            img_gpu[torch.flatten(index_samples_with_cf)],
                            onehot_prediction[torch.flatten(index_samples_with_cf)],
                            memory=memory_input,
                            enc_model=ae_fulldataset.encoder,
                            target_class=target_class, class_proto=class_prototypes,
                            feature_range=feature_range, proto_val=prototype,
                            verbose=False)
                else :
                    # case algo = 'memory'
                    counterfactual = memory_input[index_top_counterfactual]



            else:
                counterfactual = None

            if counterfactual is not None:
                orig_pred_class = torch.flatten(torch.argmax(onehot_prediction, axis=1)
                        [count_counterfactuals.nonzero()])

                counterfactual = counterfactual.cuda()
                counterfactual = counterfactual.squeeze(1)
                cf_pred_class = torch.argmax(model(counterfactual,aux_memory), axis=1)

                for or_class, cf_class in zip(orig_pred_class,cf_pred_class):
                    ae_cf = class_specific_aes[cf_class.item()]
                    ae_orig = class_specific_aes[or_class.item()]
                    im1_score = counterfactuals_utils.compute_im1_score(counterfactual, ae_cf,
                                                                        ae_orig)
                    iim1_score = counterfactuals_utils.compute_iim1_score(counterfactual, ae_cf,
                                                                          ae_orig)
                    im2_score = counterfactuals_utils.compute_im2_score(counterfactual, ae_orig,
                                                                        ae_fulldataset)
                    samples_im1_scores.append(im1_score)
                    samples_iim1_scores.append(iim1_score)
                    samples_im2_scores.append(im2_score.item())
                print(f'{batch_idx}/{len(test_loader)}\t' \
                      f'im1 score: {np.mean(samples_im1_scores):.3f}\t'\
                      f'iim1 score: {np.mean(samples_iim1_scores):.3f}\t'\
                      f'im2score:{np.mean(samples_im2_scores):.3f}\t'\
                      f'ETA:{(time.time() - starting_time)/60:.1f}',end='\r')

        runs_scores_im1.append(np.mean(samples_im1_scores))
        runs_scores_iim1.append(np.mean(samples_iim1_scores))
        runs_scores_im2.append(np.mean(samples_im2_scores))
        print(f'{index_model}/{len(list_models)}\t'
              f'run im1_score mean: {np.mean(runs_scores_im1):.3f}\t'
              f'std: {np.std(runs_scores_im1):.3f}\t'\
              f'run iim1_score mean: {np.mean(runs_scores_iim1):.3f}\t'
              f'std: {np.std(runs_scores_iim1):.3f}\t'\
              f'run im2_score mean: {np.mean(runs_scores_im2):.4f}\t'
              f'std: {np.std(runs_scores_im2):.3f}\t'
              f'ETA:{(time.time() - starting_time)/60:.1f}')

def main(args):

    run_proto_cf(FLAGS.path_models, FLAGS.path_aes, FLAGS.dir_dataset)

if __name__ == '__main__':
    absl.app.run(main)
