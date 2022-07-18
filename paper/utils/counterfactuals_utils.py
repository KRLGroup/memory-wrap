# Porting from https://github.com/SeldonIO/alibi/blob/master/alibi/explainers/cfproto.py
from typing import Optional, Union

import torch
import numpy as np

EPSILON = 1e-8

def compute_im1_score(counterfactual:torch.Tensor, autoencoder_counterfactualclass:torch.nn.Module,
                      autoencoder_inputclass:torch.nn.Module) -> float:
    """Compute IM1 score for the generated counterfactual.

    Compute the IM1 score as a function of the recostruction errors of an autoencoder trained on
    the countefactual class and an autoencoer trained on the input class, as defined in the
    paper https://arxiv.org/pdf/1907.02584.pdf.


    Args:
        counterfactual (torch.Tensor): input to evaluate
        autoencoder_counterfactualclass (torch.nn.Module): autoencoder trained only using samples of
                    the counterfactual's class
        autoencoder_inputclass (torch.nn.Module): autoencoder trained only using samples of the
                    original input's class

    Returns:
        float: IM1 score
    """
    error_cf_class = torch.norm(counterfactual - autoencoder_counterfactualclass(counterfactual),2)
    error_input_class = torch.norm(counterfactual - autoencoder_inputclass(counterfactual),2)
    im1_score = error_cf_class / (error_input_class+EPSILON)
    return im1_score.item()


def compute_iim1_score(counterfactual:torch.Tensor, autoencoder_counterfactualclass:torch.nn.Module,
                       autoencoder_inputclass:torch.nn.Module) -> float:
    """Compute IM1 score for the generated counterfactual.

    Compute the IM1 score as a function of the recostruction errors of an autoencoder trained on
    the countefactual class and an autoencoer trained on the input class, as defined in the paper
    https://arxiv.org/pdf/1907.02584.pdf.


    Args:
        counterfactual (torch.Tensor): input to evaluate
        autoencoder_counterfactualclass (torch.nn.Module): autoencoder trained only using samples
                of the counterfactual's class
        autoencoder_inputclass (torch.nn.Module): autoencoder trained only using samples of the
                original input's class

    Returns:
        float: IM1 score
    """
    error_cf_class = torch.norm(counterfactual - autoencoder_counterfactualclass(counterfactual),2)
    error_input_class = torch.norm(counterfactual - autoencoder_inputclass(counterfactual),2)
    im1_score = error_input_class/(error_cf_class+EPSILON)
    return im1_score.item()


def compute_im2_score(counterfactual:torch.Tensor, autoencoder_counterfactualclass:torch.nn.Module,
                      autoencoder_fulldataset:torch.nn.Module) -> float:
    """Compute IM2 score con the generated counterfactual.

    Compute the IM2 score as a function of similarity between the reconstructed counterfactuals of
    an autoencoder trained on the countefactual class and an autoencoer trained on full dataset as
    defined in the paper https://arxiv.org/pdf/1907.02584.pdf.

    Args:
        counterfactual (torch.Tensor): Generated counterfactual to evaluate
        autoencoder_counterfactualclass (torch.nn.Module): autoencoder trained only using samples
                of the counterfactual's class
        autoencoder_fulldataset (torch.nn.Module): autoencoder trained only using the full dataset

    Returns:
        float: IM2 score
    """
    error = torch.norm(autoencoder_counterfactualclass(counterfactual) \
                        - autoencoder_fulldataset(counterfactual),2)
    im2_score = error / (torch.norm(counterfactual)+EPSILON)
    return im2_score


def compare(predictions: Union[float, int, torch.Tensor], target: int, kappa:float) -> bool:
    """
    Compare predictions with target labels and return whether counterfactual conditions hold.

    Parameters
    ----------
    predictions
        Predicted class probabilities or labels.
    target
        Target or predicted labels.

    Returns
    -------
    Bool whether counterfactual conditions hold.
    """

    if not isinstance(predictions, (float, int, np.int64)) and len(predictions.size()) > 0:
        predictions = torch.clone(predictions.detach())
        predictions[target] += kappa  # type: ignore
        predictions = torch.argmax(predictions)  # type: ignore
    return predictions != target


def bisect_lambda(cf_found, l_step, lam, lam_lb, lam_ub,batch_size):
    for batch_idx in range(batch_size):
         # minimum number of CF instances to warrant increasing lambda
        if cf_found[batch_idx][l_step] >= 5:
            lam_lb[batch_idx] = max(lam[batch_idx], lam_lb[batch_idx])
            if lam_ub[batch_idx] < 1e9:
                lam[batch_idx] = (lam_lb[batch_idx] + lam_ub[batch_idx]) / 2
            else:
                lam[batch_idx] *= 10

        elif cf_found[batch_idx][l_step] < 5:
            # if not enough solutions found so far, decrease lambda by a factor of 10,
            # otherwise bisect up to the last known successful lambda
            lam_ub[batch_idx] = min(lam_ub[batch_idx], lam[batch_idx])
            if lam_lb[batch_idx] > 0:
                lam[batch_idx] = (lam_lb[batch_idx] + lam_ub[batch_idx]) / 2
            else:
                lam[batch_idx] /= 10

    return lam, lam_lb, lam_ub


def update_adv_s(adv_post_gradient_step, adv_pre_gradient_step, global_step, feature_range):
    zt =torch.divide(global_step, (global_step + torch.tensor(3.0)))
    updated_adv_s = adv_post_gradient_step + (zt * (adv_post_gradient_step - adv_pre_gradient_step))
    # map to feature space
    updated_adv_s = torch.minimum(updated_adv_s, feature_range[1])
    updated_adv_s = torch.maximum(updated_adv_s, feature_range[0])
    return updated_adv_s


def update_adv(adv_s, orig, beta, feature_range):

    cond = [torch.gt(torch.subtract(adv_s, orig), beta),
            torch.le(torch.abs(torch.subtract(adv_s, orig)), beta),
            torch.lt(torch.subtract(adv_s, orig), -beta)]
    upper = torch.minimum(torch.subtract(adv_s, beta),
                          feature_range[1])
    lower = torch.maximum(torch.add(adv_s, beta),
                          feature_range[0])

    assign_adv = torch.multiply(
        cond[0], upper) + torch.multiply(cond[1], orig) + torch.multiply(cond[2],      lower)
    return assign_adv


def compute_l2_loss(delta_s, shape):
    ax_sum = torch.arange(1, len(shape)).tolist()
    #l2 = torch.sum(torch.pow(delta, 2), dim=ax_sum)
    l2_s = torch.sum(torch.pow(delta_s, 2), dim=ax_sum)
    return l2_s


def compute_l1_loss(delta_s, shape):
    ax_sum = torch.arange(1, len(shape)).tolist()
    #l1 = torch.sum(torch.abs(delta), dim=ax_sum)
    l1_s = torch.sum(torch.abs(delta_s), dim=ax_sum)
    return l1_s


def compute_l1_l2_loss(delta, delta_s, beta, shape):
    ax_sum = list(np.arange(1, len(shape)))
    l2 = torch.sum(torch.pow(delta, 2), dim=ax_sum)
    l2_s = torch.sum(torch.pow(delta_s, 2), dim=ax_sum)
    l1 = torch.sum(torch.abs(delta), dim=ax_sum)
    l1_s = torch.sum(torch.abs(delta_s), dim=ax_sum)
    l1_l2 = l2 + torch.multiply(l1, beta)
    l1_l2_s = l2_s + torch.multiply(l1_s, beta)
    return l1_l2, l1_l2_s


def compute_proto_loss(model, adv_s, target_proto, theta):
    loss_proto_s = theta * torch.pow(torch.linalg.vector_norm(model(adv_s) \
                                     - target_proto,dim=[1,2,3]), 2)
    return loss_proto_s


def get_score(model, X: torch.Tensor, adv_class: int, orig_class: int, class_proto,
              eps: float = 1e-10) -> float:
    """
    Parameters
    ----------
    X
        Instance to encode and calculate distance metrics for.
    adv_class
        Predicted class on the perturbed instance.
    orig_class
        Predicted class on the original instance.
    eps
        Small number to avoid dividing by 0.

    Returns
    -------
    Ratio between the distance to the prototype of the predicted class for the original
        instance and the prototype of the predicted class for the perturbed instance.
    """

    X_enc = model(X)  # type: ignore[union-attr]
    adv_proto = class_proto[adv_class]
    orig_proto = class_proto[orig_class]
    dist_adv = torch.linalg.norm(X_enc - adv_proto)
    dist_orig = torch.linalg.norm(X_enc - orig_proto)

    return dist_orig / (dist_adv + eps)  # type: ignore[return-value]


def attack(model:torch.nn.Module,X: torch.tensor, Y: torch.tensor, memory: torch.tensor,
           enc_model=None, kappa=0, target_class: Optional[list] = None, class_proto=None,
           threshold: float = 0., c_steps=2, c_init=1, beta=0.1, theta=100., feature_range=(-1, 1),
           max_iterations=1000, proto_val=None, learning_rate_init=1e-1, clip=1000, verbose=True):
    """
    Find a counterfactual (CF) for instance `X` using a fast iterative shrinkage-thresholding
    algorithm (FISTA).

    Parameters
    ----------
    X
        Instance to attack.
    Y
        Labels for `X` as one-hot-encoding.
    target_class
        List with target classes used to find closest prototype. If ``None``, the nearest prototype
        except for the predict class on the instance is used.
    k
        Number of nearest instances used to define the prototype for a class. Defaults to using all
        instances belonging to the class if an encoder is used and to 1 for k-d trees.
    k_type
        Use either the average encoding of the k nearest instances in a class (``k_type='mean'``) or
        the k-nearest encoding in the class (``k_type='point'``) to define the prototype of
        that class. Only relevant if an encoder is used to define the prototypes.
    threshold
        Threshold level for the ratio between the distance of the counterfactual to the prototype
        of the predicted class for the original instance over the distance to the prototype of the
        predicted class for the counterfactual. If the trust score is below the threshold, the
        proposed counterfactual does not meet the requirements.
    verbose
        Print intermediate results of optimization if ``True``.
    print_every
        Print frequency if verbose is ``True``.
    log_every
        `tensorboard` log frequency if write directory is specified.

    Returns
    -------
    Overall best attack and gradients for that attack.
    """

    # shape of the input
    shape = X.shape
    batch_size = X.shape[0]

    # set the lower and upper bounds for the constant 'c' to scale the attack loss term
    # these bounds are updated for each c_step iteration
    const_lb = torch.zeros(batch_size).cuda()
    const_ub = torch.ones(batch_size) * 1e10
    const_ub = const_ub.cuda()
    const = torch.ones(batch_size) * c_init
    const = const.cuda()

    # init values for the best attack instances for each instance in the batch
    overall_best_dist = [1e10] * batch_size
    overall_best_attack = torch.zeros_like(X)
    overall_best_attack = overall_best_attack.cuda()

     # define torch variable for constant used in FISTA optimization
    const = torch.ones(batch_size) * c_init
    const = const.cuda()
    X_num = X

    # init parameters for FISTA optimization
    orig =X_num
    adv_s = torch.zeros(shape).cuda()
    adv_s.requires_grad = True

    # hyperparameters for gradient descent
    power = 0.5
    end_learning_rate = 0
    decay_steps = max_iterations

    # iterate over nb of updates for 'c'
    for _ in range(c_steps):

        # init variables
        global_step = torch.tensor(0.0)

        # assign variables for the current iteration
        # reset here
        adv = X_num
        adv_s = X_num
        # reset current best distances and scores
        current_best_dist = [1e10] * batch_size
        current_best_proba = [-1] * batch_size

        adv_s.requires_grad = True
        delta = orig - adv
        delta_s = orig - adv_s
        target_proto = proto_val
        for i in range(max_iterations):
            #target_proto = proto_val.detach().clone()
            adv_s.requires_grad = True

            delta = orig - adv
            delta_s = orig - adv_s

            # l1, l2, l1_l2 losses
            loss_l1_s = compute_l1_loss(delta_s, shape)

            loss_l2_s = compute_l2_loss(delta_s, shape)

            pred_proba = model(adv,memory)
            loss_proto_s = compute_proto_loss(
                    enc_model, adv_s, target_proto, theta)

            # torch loss combined
            #loss_opt = loss_attack_s + loss_l2_s + loss_ae_s + loss_proto_s
            loss_opt = torch.multiply(beta,loss_l1_s) + loss_l2_s  + loss_proto_s

            # zeroing gradients
            if adv_s.grad is not None:
                adv_s.grad.zero_()
            # compute  and clip gradients
            loss_opt = loss_opt.sum()
            loss_opt.backward()
            torch.nn.utils.clip_grad_value_(adv_s, clip)

            # apply gradients
            with torch.no_grad():
                # polynomial decay
                global_step = min(global_step, decay_steps)

                decayed_learning_rate = (learning_rate_init - end_learning_rate) *  torch.pow(1 - (global_step / decay_steps),power)  +  end_learning_rate
                adv_s = adv_s - (adv_s.grad * decayed_learning_rate)
                global_step += 1

                # add L1 term to overall loss; this is not the loss that will be directly optimized
                l1_l2, _ = compute_l1_l2_loss(delta, delta_s, beta, shape)

                for batch_idx, (dist, proba, adv_idx) in enumerate(zip(l1_l2, pred_proba, adv)):
                    Y_class = torch.argmax(Y[batch_idx])
                    adv_class = torch.argmax(proba)

                    adv_idx = torch.unsqueeze(adv_idx, dim=0)

                    # calculate trust score
                    if threshold > 0.:
                        if enc_model is not None:
                            score = get_score(enc_model,adv_idx, torch.argmax(
                            pred_proba), Y_class,class_proto=class_proto)
                        else:
                            score = get_score(model,adv_idx, torch.argmax(
                                pred_proba), Y_class,class_proto=class_proto)
                        above_threshold = score > threshold
                    else:
                        above_threshold = True


                    # current step
                    if (dist < current_best_dist[batch_idx] and compare(proba, Y_class, kappa)
                            and above_threshold and adv_class in target_class):
                        current_best_dist[batch_idx] = dist
                        # type: ignore
                        current_best_proba[batch_idx] = adv_class


                    # global
                    if (dist < overall_best_dist[batch_idx] and compare(proba, Y_class,kappa)
                            and above_threshold and adv_class in target_class):
                        if verbose:
                            print('\nNew best counterfactual found!')
                        overall_best_dist[batch_idx] = dist
                        overall_best_attack[batch_idx] = adv_idx[0]
                        best_attack = True

                # update values of adv, adv_s, delta and delta_s
                assign_adv = update_adv(adv_s, orig, beta, feature_range)

                assign_adv_s = update_adv_s(
                 assign_adv, adv, global_step, feature_range)
                adv = assign_adv
                adv_s = assign_adv_s.clone()
                adv_s.requires_grad = True
                delta = orig - adv
                delta_s = orig - adv_s

        #adjust the 'c' constant for the first loss term
        for batch_idx in range(batch_size):
            if (compare(current_best_proba[batch_idx], torch.argmax(Y[batch_idx]),kappa) and
                    current_best_proba[batch_idx] != -1):
                # want to refine the current best solution by putting more emphasis on the
                # regularization terms of the loss by reducing 'c'; aiming to find a
                # perturbation closer to the original instance
                const_ub[batch_idx] = min(
                    const_ub[batch_idx], const[batch_idx])
                if const_ub[batch_idx] < 1e9:
                    const[batch_idx] = (
                        const_lb[batch_idx] + const_ub[batch_idx]) / 2
            else:
                # no valid current solution; put more weight on the first loss term to try and
                # meet the prediction constraint before finetuning the solution with the
                # regularization terms update lower bound to constant
                const_lb[batch_idx] = max(
                    const_lb[batch_idx], const[batch_idx])
                if const_ub[batch_idx] < 1e9:
                    const[batch_idx] = (
                        const_lb[batch_idx] + const_ub[batch_idx]) / 2
                else:
                    const[batch_idx] *= 10

    best_attack = overall_best_attack
    return best_attack
