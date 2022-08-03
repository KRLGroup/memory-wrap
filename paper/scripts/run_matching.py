# reference https://github.com/oscarknagg/few-shot/blob/master/few_shot/matching.py
import time
import os
import pickle
from typing import List, Tuple
import sys

import torch
import torch.nn as nn
import numpy as np
import absl.flags
import absl.app
import yaml

sys.path.append('..')
import utils.utils as utils

# user flags
absl.flags.DEFINE_bool("continue_train", False, "std, memory or mlp")
absl.flags.DEFINE_enum("optimizer", None, ['sgd', 'adam'],
                       "Optimizer to use, sgd or adam")
FLAGS = absl.flags.FLAGS

# constants
_EPSILON = 1e-6


class MatchingNetwork(nn.Module):
    def __init__(self, encoder: torch.nn.Module, fce: bool,
                 num_input_channels: int,
                 lstm_layers: int, lstm_input_size: int, unrolling_steps: int,
                 device: torch.device):
        """Creates a Matching Network as described in Vinyals et al.

        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            fce: Whether or not to us fully conditional embeddings
            num_input_channels: Number of color channels the model expects
                input data to contain.
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that
                embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and
                Attention LSTM. This is determined by the embedding dimension
                f the few shot encoder which is in turn determined by the size
                of the input data.
            unrolling_steps: Number of unrolling steps to run the Attention
                LSTM
            device: Device on which to run computation
        """
        super(MatchingNetwork, self).__init__()
        self.fce = fce
        self.num_input_channels = num_input_channels
        self.encoder = encoder
        if self.fce:
            self.g = BidrectionalLSTM(lstm_input_size, lstm_layers).to(
                    device, dtype=torch.float)
            self.f = AttentionLSTM(lstm_input_size,
                                   unrolling_steps=unrolling_steps).to(
                    device, dtype=torch.float)

    def forward(self, inputs):
        pass


class BidrectionalLSTM(nn.Module):
    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings
            (FCE)
        of the support set as described in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be
                the same in order to implement the skip connection described in
                Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching
        # Networks
        self.lstm = nn.LSTM(input_size=size,
                            num_layers=layers,
                            hidden_size=size,
                            bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates
        # initial hidden states
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written
        # in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):
    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE)
           of the query set as described in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be
                the same in order to implement the skip connection described
                in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set
                to compute. Analogous to number of layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size,
                                     hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise ValueError("Support and query set have different "
                             "embedding dimension!")

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).cuda().float()
        c = torch.zeros(batch_size, embedding_dim).cuda().float()

        for k in range(self.unrolling_steps):
            # Calculate hidden state cf. equation (4) of appendix A.2
            h = h_hat + queries

            # Calculate softmax attentions between hidden states and support
            # set embeddings
            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # Calculate readouts from support set embeddings cf. equation (5)
            readout = torch.mm(attentions, support)

            # Run LSTM cell cf. equation (3)
            # h_hat, c = self.lstm_cell(queries,
            #           (torch.cat([h, readout], dim=1), c))
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h


def train_model(model: torch.nn.Module,
                loaders: List[torch.utils.data.DataLoader],
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                loss_criterion: torch.nn.modules.loss, num_epochs: int,
                device: torch.device) -> torch.nn.Module:
    """ Function to train a model with a Memory Wrap layer (in the paper both
    the baseline variant and Memory Wrap)

    Args:
        model (torch.nn.Module): Model with a Memory Wrap layer to be trained
        loaders (List[torch.utils.data.DataLoader]): Loaders containing
            dataset subsets to be used to train the model. The loaders[0]
            element is the training dataset, while loaders[1] contain the
            dataset used to sample memory sets
        optimizer (torch.optim.Optimizer): PyTorch optimizer to use to perform
            training step
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate
        scheduler to adaptive adjusting the learning rate during training
        loss_criterion (torch.nn.modules.loss): criterion to use to compute
            the loss
        num_epochs (int): number of epoch to train the model
        device (torch.device): device where the model is stored

    Returns:
        torch.nn.Module: the trained model
    """
    train_loader, mem_loader = loaders

    # training process
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, y) in enumerate(train_loader):
            optimizer.zero_grad()
            # input
            data = data.to(device)
            y = y.to(device)
            support_set, y_ss = next(iter(mem_loader))
            support_set = support_set.to(device)
            # get embedding x
            with torch.cuda.amp.autocast():
                embedding_input = model.encoder(data)
                embedding_support = model.encoder(support_set)
                # Calculate the fully conditional embedding, g, for support
                # set samples as described in appendix A.2 of the paper.
                # g takes the form of a bidirectional LSTM with a
                # skip connection from inputs to outputs
                embedding_support, _, _ = model.g(
                    embedding_support.unsqueeze(1))
                embedding_support = embedding_support.squeeze(1)

                # Calculate the fully conditional embedding, f, for the query
                # set samples as described in appendix A.1 of the paper.
                queries = model.f(embedding_support, embedding_input)
                # Efficiently calculate distance between all queries and all prototypes
                # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
                distances = utils.vector_distance(queries,
                                                  embedding_support, 'cosine')

                # Calculate "attention" as softmax over support-query distances
                attention = (-distances).softmax(dim=1)

                # Calculate predictions as in equation (1) from Matching Networks
                # y_hat = \sum_{i=1}^{k} a(x_hat, x_i) y_i
                y_onehot = torch.nn.functional.one_hot(y_ss, 10).cuda().float()
                y_pred = torch.mm(attention, y_onehot)

                # Calculated loss with negative log likelihood
                clipped_y_pred = y_pred.clamp(_EPSILON, 1 - _EPSILON)
                loss = loss_criterion(clipped_y_pred.log(), y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # log stuff
            print(f'Train Epoch: {epoch} [{batch_idx} '
                  f'/{len(train_loader)}]\t', end='\r')

        # increase scheduler step for each epoch
        if scheduler is not None:
            scheduler.step()

    return model


def eval(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
         mem_loader: torch.utils.data.DataLoader,
         loss_criterion: torch.nn.modules.loss,
         device: torch.device) -> Tuple[float, float]:
    """ Method to evaluate a pytorch model that has a Memory Wrap variant
        as last layer. The model takes an image and a memory samples as
        input and return the logits.

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
        Tuple[float, float]: Accuracy and loss in the given test dataset
    """
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            support_set, y_ss = next(iter(mem_loader))
            support_set = support_set.to(device)
            with torch.cuda.amp.autocast():
                embedding_input = model.encoder(data)
                embedding_support = model.encoder(support_set)
                # Calculate the fully conditional embedding, g, for support
                # set samples as described in appendix A.2 of the paper.
                # g takes the form of a bidirectional LSTM with a
                # skip connection from inputs to outputs
                embedding_support, _, _ = model.g(
                        embedding_support.unsqueeze(1))
                embedding_support = embedding_support.squeeze(1)

                # Calculate the fully conditional embedding, f, for the query
                # set samples as described in appendix A.1 of the paper.
                queries = model.f(embedding_support, embedding_input)
                # Efficiently calculate distance between all queries and all prototypes
                # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
                distances = utils.vector_distance(queries, embedding_support,
                                                  'cosine')

                # Calculate "attention" as softmax over support-query distances
                attention = (-distances).softmax(dim=1)

                # Calculate predictions as in equation (1) from Matching Networks
                # y_hat = \sum_{i=1}^{k} a(x_hat, x_i) y_i
                y_onehot = torch.nn.functional.one_hot(y_ss, 10).cuda().float()
                y_pred = torch.mm(attention, y_onehot)

                # Calculated loss with negative log likelihood
                # Clip predictions for numerical stability
                clipped_y_pred = y_pred.clamp(_EPSILON, 1 - _EPSILON)
                loss = loss_criterion(clipped_y_pred.log(), target)

                pred = y_pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            test_loss += loss.item()

        test_accuracy = 100.*(torch.true_divide(correct, len(loader.dataset)))
    return test_accuracy,  test_loss


def run_experiment(config: dict):
    """Method to run an experiment. Each experiment is composed by n
    runs, defined in the config dictionary, where in each of them a new
    model is trained.

    Args:
        config (dict): Dictionary containing the configuration of the models
            to train.
        modality (str): Model type. One of [std, memory, encoder_memory] where
            std is the standard model, memory is the baseline that uses only
            the memory and encoder_memory is Memory Wrap
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))

    modality = 'std'
    # get dataset info
    dataset_name = config['dataset_name']
    config['dataset_dir'] = '../datasets/'
    num_classes = config[dataset_name]['num_classes']

    # training parameters
    loss_criterion = torch.nn.CrossEntropyLoss()
    # saving/loading stuff
    save = config['save']
    path_saving_model = '../models/{}/matching/{}/{}/'.format(
        dataset_name, config['model'], config['train_examples'])
    if save and not os.path.isdir(path_saving_model):
        os.makedirs(path_saving_model)

    # optimizer parameters
    learning_rate = float(config['optimizer']['learning_rate'])
    weight_decay = float(config['optimizer']['weight_decay'])
    nesterov = bool(config['optimizer']['nesterov'])
    momentum = float(config['optimizer']['momentum'])
    dict_optim = {'lr': learning_rate, 'momentum': momentum,
                  'weight_decay': weight_decay, 'nesterov': nesterov}
    opt_milestones = config[dataset_name]['opt_milestones']

    run_acc = []
    initial_run = 0
    if FLAGS.continue_train:
        # load model
        print("Restarting training process\n")
        info = pickle.load(open(path_saving_model + "conf.p", "rb"))
        initial_run = info['run_num']
        run_acc = info['accuracies']
    for run in range(initial_run, config['runs']):
        run_time = time.time()
        utils.set_seed(run)
        torch.cuda.init()
        model_encoder = utils.get_model(config['model'], num_classes,
                                        model_type=modality)
        model_encoder = model_encoder.to(device)
        num_input_features = model_encoder.linear.in_features
        model_encoder.linear = torch.nn.Identity()
        fce = True
        model = MatchingNetwork(model_encoder, fce, num_input_channels=3,
                                lstm_layers=1,
                                lstm_input_size=num_input_features,
                                unrolling_steps=2, device=device)

        # training parameters
        if FLAGS.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), **dict_optim)
        elif FLAGS.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(config['optimizer']['learning_rate']))
        else:
            raise ValueError("Optimizer not supported")
        if dataset_name == 'CINIC10':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config[dataset_name]['num_epochs'])
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=opt_milestones)

        # get dataset
        train_loader, _, test_loader, mem_loader = utils.get_loaders(
            config, run, balanced=False)
        train_time = time.time()
        model = train_model(model, [train_loader, mem_loader], optimizer,
                            scheduler, loss_criterion,
                            config[dataset_name]['num_epochs'],
                            device=device)
        init_eval_time = time.time()
        best_acc, _ = eval(model, test_loader, mem_loader, loss_criterion,
                           device)
        end_eval_time = time.time()

        # stats
        run_acc.append(best_acc)

        # save
        if save and path_saving_model:
            saved_name = "{}.pt".format(run+1)
            save_path = os.path.join(path_saving_model, saved_name)
            torch.save({'model_state_dict': model.state_dict(),
                        'train_examples': config['train_examples'],
                        'mem_examples': config[config['dataset_name']]
                                              ['mem_examples'],
                        'model_name': config['model'],
                        'num_classes': num_classes, 'modality': modality,
                        'dataset_name': config['dataset_name']},
                       save_path)
            info = {'run_num': run+1, 'accuracies': run_acc}
            pickle.dump(info, open(path_saving_model + "conf.p", "wb"))

        # log
        print(f"Run:{run+1} |  Accuracy {best_acc:.2f} | "
              f"Mean Accuracy:{np.mean(run_acc):.2f} | "
              f"Std Dev Accuracy:{np.std(run_acc):.2f}\t"
              f"T:{(train_time -run_time)/60:.2f}min\t"
              f"E:{(end_eval_time - init_eval_time)/60:.2f}")


def main(argv):
    config_file = open(r'../config/train.yaml', 'r')
    config = yaml.safe_load(config_file)

    print(f"Model:{config['model']}\nSizeTrain:{config['train_examples']}\n")
    run_experiment(config)


if __name__ == '__main__':
    absl.app.run(main)
