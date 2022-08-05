

import sys
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import absl.flags
import absl.app

sys.path.append('..')

from architectures.autoencoder import AutoEncoder
import utils.utils as utils
import utils.datasets as datasets

FLAGS = absl.flags.FLAGS

absl.flags.DEFINE_string("dir_models", None, "Directory of the trained model")
absl.flags.DEFINE_string("saving_path", None, "Directory where to store the trained autoencoders")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "Path of the trained autoencoder model")
absl.flags.mark_flag_as_required("dir_models")
absl.flags.mark_flag_as_required("saving_path")

# hyperparameters
NUM_EPOCHS_AE = 4


def run_experiment(path_models, saving_path=None, dir_dataset=None):
    # load name of models in the path_models dir
    list_models = [name_file for name_file in os.listdir(path_models) if name_file.endswith('.pt')]
    for name_model in list_models:
        # set seed for reproducibility using the index of the model as seed
        seed = int(name_model[:-3])-1
        utils.set_seed(seed)

        ####################################################
        ######### load classification model ################
        ###################################################
        checkpoint = torch.load(path_models+name_model)
        model_type = checkpoint['modality']
        model = utils.get_model(checkpoint['model_name'], checkpoint['num_classes'],
                                model_type=model_type)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.cuda()
        model.eval()

        ####################################################
        ################ load dataset  #####################
        ####################################################
        dataset_dir = dir_dataset

        trainset, _, testset = datasets.get_SVHN_dataset(data_dir=dataset_dir, size_train=100000,
                                                         seed=seed)


        # train one AE for each class
        for id_class in range(10):

            # extract indices of samples associated witht the given class that
            # are also included in the reduced dataset
            idx_class = np.where((trainset.dataset.dataset.labels==id_class)==1)[0]
            idx_subset = trainset.indices
            valid_indices = [id for id in idx_subset if id in idx_class]
            dset_train = torch.utils.data.dataset.Subset(trainset.dataset.dataset, valid_indices)
            dataloader = torch.utils.data.DataLoader(
            dset_train, batch_size=32, shuffle=True, num_workers=1)

            idx_test = testset.labels==id_class
            dset_test = torch.utils.data.dataset.Subset(testset, np.where(idx_test==1)[0])
            testloader = torch.utils.data.DataLoader(
            dset_test, batch_size=4, shuffle=False, num_workers=1)
            # Train AE
            print(f"Training autoencoder for class {id_class}")
            autoencoder = AutoEncoder(num_channels=3)
            autoencoder = autoencoder.cuda()
            loss_criterion_ae = nn.MSELoss()
            optimizer = optim.Adam(autoencoder.parameters())

            autoencoder.train()
            avg_loss_ae = []
            for epoch in range(NUM_EPOCHS_AE):
                for _,data in enumerate(dataloader):
                    optimizer.zero_grad()
                    img, _ = data
                    img = img.cuda()
                    # ===================forward=====================
                    output = autoencoder(img)
                    loss = loss_criterion_ae(output, img)
                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss_ae.append(loss.item())
                # ===================log========================
                print(f'epoch [{epoch +1}/{NUM_EPOCHS_AE}], loss:{np.mean(avg_loss_ae):.4f}')

            # Test AE
            autoencoder.eval()
            avg_loss_ae_test = []
            with torch.no_grad():
                for _, data in enumerate(testloader):
                    img, _ = data
                    img = img.cuda()
                    output = autoencoder(img)
                    loss = loss_criterion_ae(output, img)
                    avg_loss_ae_test.append(loss.item())
                print(f'Test loss: {np.mean(avg_loss_ae_test):.4f}')
            print("Saving the model...")
            seed_path_saving = f'{saving_path}/{seed}/'
            if not os.path.isdir(seed_path_saving):
                os.makedirs(seed_path_saving)
            torch.save(autoencoder.state_dict(), f"{seed_path_saving}class_{id_class}.pt")


        dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=1)

        testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=1)




        # Train AE
        print("training autoencoder full")
        autoencoder = AutoEncoder(num_channels=3)
        autoencoder = autoencoder.cuda()
        loss_criterion_ae = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters())

        autoencoder.train()
        avg_loss_ae = []
        for epoch in range(NUM_EPOCHS_AE):
            for data in dataloader:
                optimizer.zero_grad()
                img, _ = data
                img = img.cuda()
                # ===================forward=====================
                output = autoencoder(img)
                loss = loss_criterion_ae(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss_ae.append(loss.item())
            # ===================log========================
            print(f'epoch [{epoch +1}/{NUM_EPOCHS_AE}], loss:{np.mean(avg_loss_ae):.4f}')


        # Test AE
        autoencoder.eval()
        avg_loss_ae_test = []
        with torch.no_grad():
            for _, data in enumerate(testloader):
                img, _ = data
                img = img.cuda()
                output = autoencoder(img)
                loss = loss_criterion_ae(output, img)
                avg_loss_ae_test.append(loss.item())
            print(f'Test loss: {np.mean(avg_loss_ae_test):.4f}')

        # Save AE
        torch.save(autoencoder.state_dict(), f"{seed_path_saving}/full.pt")


def main(args):

    run_experiment(FLAGS.dir_models, FLAGS.saving_path, FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)
