import os
import torch # type: ignore
import torchvision # type: ignore
import numpy as np
import random
import matplotlib.pyplot as plt # type: ignore
import absl.flags
import absl.app
import datasets
import aux

# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_string("dataset", None, "Dataset to test (SVHN, CIFAR10)")
absl.flags.DEFINE_string("modality", None, "Fixed or random")
absl.flags.DEFINE_integer("batch_size_test", 3, "Number of samples for each image")

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


def run(path, dataset_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))    
    # load model
    checkpoint = torch.load(path)
    model = aux.get_model( checkpoint['model_name'],checkpoint['num_classes'],model_type=FLAGS.modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()


    # load data
    train_examples = checkpoint['train_examples']
    if FLAGS.dataset == 'CIFAR10':
        name_classes= ['airplane','automobile',	'bird',	'cat','deer','dog',	'frog'	,'horse','ship','truck']
    else:
        name_classes = range(checkpoint['num_classes'])
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    undo_normalization = getattr(datasets, 'undo_normalization_'+dataset_name)
    batch_size_test = FLAGS.batch_size_test
    _, _, test_loader, mem_loader = load_dataset('../datasets',batch_size_train=50, batch_size_test=batch_size_test,batch_size_memory=100,size_train=train_examples)
    memory_iter = iter(mem_loader)
    
    #saving stuff
    dir_save = "images/mem_images/"+FLAGS.dataset+"/"+FLAGS.modality+"/" + checkpoint['model_name'] + "/"
    if not os.path.isdir(dir_save): 
        os.makedirs(dir_save)

    def get_image(image, revert_norm=True):
        if revert_norm:
            im = undo_normalization(image)
        else:
            im = image
        im = im.squeeze().cpu().detach().numpy()
        transformed_im = np.transpose(im, (1, 2, 0))
        return transformed_im


    for batch_idx, (images, _) in enumerate(test_loader):
        try:
            memory, _ = next(memory_iter)
        except StopIteration:
            memory_iter = iter(mem_loader)
            memory, _ = next(memory_iter)
                
        images = images.to(device)
        memory = memory.to(device)

        # compute output
        outputs,rw = model(images,memory,return_weights=True)
        _, predictions = torch.max(outputs, 1)

        # compute memory outputs
        mem_val,memory_sorted_index = torch.sort(rw,descending=True)
        fig = plt.figure(figsize=(batch_size_test*2, 4),dpi=300)
        columns = batch_size_test
        rows = 2
        for ind in range(len(images)):
            input_selected = images[ind].unsqueeze(0)

            # M_c u M_e : set of sample with a positive impact on prediction
            m_ec = memory_sorted_index[ind][mem_val[ind]>0]

            # get reduced memory
            reduced_mem = undo_normalization(memory[m_ec])
            npimg = torchvision.utils.make_grid(reduced_mem,nrow=4).cpu().numpy()

            # build and store image

            fig.add_subplot(rows, columns, ind+1)
            plt.imshow((get_image(input_selected)* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
            plt.title('Prediction:{}'.format(name_classes[predictions[ind]]))
            plt.axis('off')
            ax2 = fig.add_subplot(rows, columns, batch_size_test+1+ind)
            plt.imshow((np.transpose(npimg, (1,2,0))* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
            plt.title('Used Samples')
            plt.axis('off')
        fig.tight_layout()
        fig.savefig(dir_save+str(batch_idx*batch_size_test+ind)+".png")
        plt.close()
        print('{}/{}'.format(batch_idx,len(test_loader)),end='\r')


def main(argv):

    run(FLAGS.path_model,FLAGS.dataset)

if __name__ == '__main__':
  absl.app.run(main)