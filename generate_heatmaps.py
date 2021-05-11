import torch # type: ignore
import numpy as np
import random
import absl.flags
import absl.app
import os
import datasets
import aux
import matplotlib.pyplot as plt # type: ignore
from matplotlib import figure # type: ignore
import captum.attr # type: ignore


# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")

absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)




def visualize_image_mult_attr(
    attr,
    original_image,
    method,
    sign,
    titles = None,
    fig_size = (4, 6),
    use_pyplot = True
):
    r"""
    Modified version of visualize_image_mult_attr of Captum library.
    Visualizes attribution using multiple visualization methods displayed
    in a 1 x k grid, where k is the number of desired visualizations.

    Args:

        attr (numpy.array): Numpy array corresponding to attributions to be
                    visualized. Shape must be in the form (H, W, C), with
                    channels as last dimension. Shape must also match that of
                    the original image if provided.
        original_image (numpy.array, optional):  Numpy array corresponding to
                    original image. Shape must be in the form (H, W, C), with
                    channels as the last dimension. Image can be provided either
                    with values in range 0-1 or 0-255. This is a necessary
                    argument for any visualization method which utilizes
                    the original image.
        methods (list of strings): List of strings of length k, defining method
                        for each visualization. Each method must be a valid
                        string argument for method to visualize_image_attr.
        signs (list of strings): List of strings of length k, defining signs for
                        each visualization. Each sign must be a valid
                        string argument for sign to visualize_image_attr.
        titles (list of strings, optional):  List of strings of length k, providing
                    a title string for each plot. If None is provided, no titles
                    are added to subplots.
                    Default: None
        fig_size (tuple, optional): Size of figure created.
                    Default: (8, 6)
        use_pyplot (boolean, optional): If true, uses pyplot to create and show
                    figure and displays the figure after creating. If False,
                    uses Matplotlib object oriented API and simply returns a
                    figure object without showing.
                    Default: True.
        **kwargs (Any, optional): Any additional arguments which will be passed
                    to every individual visualization. Such arguments include
                    `show_colorbar`, `alpha_overlay`, `cmap`, etc.


    Returns:
        2-element tuple of **figure**, **axis**:
        - **figure** (*matplotlib.pyplot.figure*):
                    Figure object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same figure provided.
        - **axis** (*matplotlib.pyplot.axis*):
                    Axis object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same axis provided.

    Examples::

        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> ig = IntegratedGradients(net)
        >>> # Computes integrated gradients for class 3 for a given image .
        >>> attribution, delta = ig.attribute(orig_image, target=3)
        >>> # Displays original image and heat map visualization of
        >>> # computed attributions side by side.
        >>> _ = visualize_mutliple_image_attr(attribution, orig_image,
        >>>                     ["original_image", "heat_map"], ["all", "positive"])
    """
    if titles is not None:
        assert len(attr) == len(titles), (
            "If titles list is given, length must " "match that of methods list."
        )
    if use_pyplot:
        plt_fig = plt.figure(figsize=fig_size)
    else:
        plt_fig = figure.Figure(figsize=fig_size)
    el_4cols = int(len(attr)/len(original_image))
    plt_axis = plt_fig.subplots(len(original_image), el_4cols)

    # When visualizing one
    if len(attr) == 1:
        plt_axis = [plt_axis]
    for i in range(len(attr)):
        row = int(i/(len(attr)/len(original_image)))
        image = original_image[row]
        column = int(i%el_4cols)
        if attr[i] is None:
            meth = "original_image"
            captum.attr.visualization.visualize_image_attr(
            attr[i],
            original_image=image,
            method=meth,
            sign=sign,
            plt_fig_axis=(plt_fig, plt_axis[row][column]),
            use_pyplot=False,
            title=titles[i] if titles else None,
        )
        else:
            meth = method
            captum.attr.visualization.visualize_image_attr(
            attr[i],
            original_image=image,
            method=meth,
            sign=sign,
            plt_fig_axis=(plt_fig, plt_axis[row][column]),
            show_colorbar=True,
            use_pyplot=False,
            alpha_overlay=0.4,
            cmap='plasma',
            outlier_perc=20,
            title=titles[i] if titles else None,
        )
        plt_axis[row][column].set_box_aspect(1)
    plt_fig.tight_layout()
    if use_pyplot:
        plt.show()
    return plt_fig, plt_axis


def run(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))
    batch_size_test=1
    
    # load model
    checkpoint = torch.load(path)
    modality = checkpoint['modality']
    if modality not in ['memory','encoder_memory']:
        raise ValueError(f'Model\'s modality (model type) must be one of [\'memory\',\'encoder_memory\'], not {modality}.')
    dataset_name = checkpoint['dataset_name']

    model = aux.get_model( checkpoint['model_name'],checkpoint['num_classes'],model_type= modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()


    # load data
    train_examples = checkpoint['train_examples']
    if dataset_name == 'CIFAR10':
        name_classes= ['airplane','automobile',	'bird',	'cat','deer','dog',	'frog'	,'horse','ship','truck']
    else:
        name_classes = range(checkpoint['num_classes'])
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    undo_normalization = getattr(datasets, 'undo_normalization_'+dataset_name)
    _, _, test_loader, mem_loader = load_dataset('../datasets',batch_size_train=50, batch_size_test=batch_size_test,batch_size_memory=100,size_train=train_examples)
    memory_iter = iter(mem_loader)
    def get_image(image, revert_norm=True):
        if revert_norm:
            im = undo_normalization(image)
        else:
            im = image
        im = im.squeeze().cpu().detach().numpy()
        transformed_im = np.transpose(im, (1, 2, 0))
        return transformed_im

    #saving stuff
    dir_save = "images/saliency/"+dataset_name+"/"+modality+"/" + checkpoint['model_name'] + "/"
    if not os.path.isdir(dir_save): 
        os.makedirs(dir_save)

    # run heatmap
    saliency = captum.attr.IntegratedGradients(model)
    show_grad = "positive"
    type_viz = "blended_heat_map"

    for batch_idx, (images, _) in enumerate(test_loader):
        try:
            memory, _ = next(memory_iter)
        except StopIteration:
            memory_iter = iter(mem_loader)
            memory, _ = next(memory_iter)
        try:
            aux_memory, _ = next(memory_iter)
        except StopIteration:
            memory_iter = iter(mem_loader)
            aux_memory, _ = next(memory_iter)

        images = images.to(device)
        memory = memory.to(device)
        aux_memory = aux_memory.to(device)

        #compute output
        outputs,rw = model(images,memory,return_weights=True)
        _, predictions = torch.max(outputs, 1)

        # compute memory outputs
        memory_outputs = model(memory,aux_memory)
        _, memory_predictions = torch.max(memory_outputs, 1)
        mem_val,memory_sorted_index = torch.sort(rw,descending=True)

        # build baselines for heatmap
        baseline_input = torch.ones_like(images[0].unsqueeze(0))
        images.requires_grad = True

        for ind in range(len(images)):
            model.zero_grad()
            input_selected = images[ind].unsqueeze(0)

            # M_c u M_e : set of sample with a positive impact on prediction
            m_ec = memory_sorted_index[ind][mem_val[ind]>0]
            pred_mec = memory_predictions[m_ec]
            mc = memory_sorted_index[ind][(pred_mec != predictions[ind]).nonzero(as_tuple=True)].tolist()
            me = memory_sorted_index[ind][(pred_mec == predictions[ind]).nonzero(as_tuple=True)].tolist()

            #get images
            current_image = get_image(images[ind])
            if mc:
                top_counter_index = mc[0]
                counter_image = get_image(memory[top_counter_index])
            if me:
                top_example_index = me[0]
                example_memory_image = get_image(memory[top_example_index])

            # get reduced memory
            reduced_mem = memory[m_ec]
            baseline_memory = torch.ones_like(reduced_mem)
            reduced_mem.requires_grad = True

            # compute gradients
            (grad_im,grad_mem) = saliency.attribute((input_selected,reduced_mem), 
baselines=(baseline_input,baseline_memory), target=predictions[ind].item(), internal_batch_size=2, n_steps=40 )
            grad_input = get_image(grad_im,False)
            if mc:
                # get counterfactual with the highest weight
                top_counter_index_grad = (pred_mec != predictions[ind]).nonzero(as_tuple=True)[0][0]
                grad_counter = get_image(grad_mem[top_counter_index_grad],False)
            if me:
                # get explanation by example with the highest weight
                top_example_index_grad = (pred_mec == predictions[ind]).nonzero(as_tuple=True)[0][0]
                grad_example = get_image(grad_mem[top_example_index_grad],False)



            # visualize
            if mc and  me:
                # case where there are counterfactuals and explanation by examples in memory
                fig,_ = visualize_image_mult_attr([None,grad_input,None,grad_example,None,grad_counter],[current_image,example_memory_image,counter_image],type_viz,show_grad,['Input\nPredict:{}'.format(name_classes[predictions[ind]]),'Saliency','Example\nw:{:.2f}'.format(rw[ind][top_example_index]),'Saliency', 'Counterfactual\nw:{:.2f}'.format(rw[ind][top_counter_index]),'Saliency'],use_pyplot=False)
            elif mc:
                # cases where there are only counterfactuals  in memory
                fig,_ = visualize_image_mult_attr([None,grad_input,None,grad_counter],[current_image,counter_image],type_viz,show_grad,['Input\nPredict:{}'.format(name_classes[predictions[ind]]),'Saliency', 'Counterfactual\nw:{:.2f}'.format(rw[ind][top_counter_index]),'Saliency'],use_pyplot=False)
            elif me:
                # cases where there are only explanations by expamples in memory
                fig,_ = visualize_image_mult_attr([None,grad_input,None,grad_example],[current_image,example_memory_image],type_viz,show_grad,['Input\nPredict:{}'.format(name_classes[predictions[ind]]),'Saliency','Example\nw:{:.2f}'.format(rw[ind][top_example_index]),'Saliency'],use_pyplot=False)

            fig.savefig(dir_save+str(batch_idx*batch_size_test+ind)+".png")
            plt.close()
            print('{}/{}'.format(batch_idx,len(test_loader)),end='\r')


def main(args):
    run(FLAGS.path_model)

if __name__ == '__main__':
  absl.app.run(main)