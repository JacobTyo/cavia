"""
Regression experiment using CVA
"""
import copy
import os
import time

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
import tasks_sine, tasks_celebA
from gameopt_model import GameOptModel
from gametask_model import GameTaskModel
from logger import LoggerGame

from torch.utils.tensorboard import SummaryWriter

def run(args, log_interval=5000, rerun=False):
    assert not args.maml

    writer = SummaryWriter('testing')

    # see if we already ran this experiment
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/tpg_{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/tpg_{}_result_files/'.format(code_root, args.task))
    path = '{}/tpg_{}_result_files/'.format(code_root, args.task) + args.id  # utils.get_path_from_args(args)

    if os.path.exists(path + '.pkl') and not rerun:
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(args.seed)

    # --- initialise everything ---

    # get the task family
    if args.task == 'sine':
        task_family_train = tasks_sine.RegressionTasksSinusoidal()  # args.num_fns)
        task_family_valid = tasks_sine.RegressionTasksSinusoidal()  # 10000)
        task_family_test = tasks_sine.RegressionTasksSinusoidal()  # 10000)
    elif args.task == 'celeba':
        task_family_train = tasks_celebA.CelebADataset('train', device=args.device)
        task_family_valid = tasks_celebA.CelebADataset('valid', device=args.device)
        task_family_test = tasks_celebA.CelebADataset('test', device=args.device)
    else:
        raise NotImplementedError

    # initialise network
    apmodel = GameTaskModel(n_in=3, n_out=2, n_hidden=[40, 40], device=args.device).to(args.device)
    model = GameOptModel(n_in=3, n_out=1, n_hidden=args.num_hidden_layers, device=args.device).to(args.device)

    # intitialise optimiser for each network
    ap_optimizer = optim.Adam(apmodel.parameters(), args.lr_meta)
    optimizer = optim.Adam(model.parameters(), args.lr_meta)

    # initialise loggers
    logger = LoggerGame()
    logger.best_valid_opt_model = copy.deepcopy(model)
    logger.best_valid_task_model = copy.deepcopy(apmodel)

    for i_iter in range(args.n_iter):
        # select task using the ap_optim network
        # last step loss diff can probably be big.  How do we account for this? clip to a range?
        apinput = np.asarray([[[1, 2.5, np.pi/2]]]) if i_iter == 0 else np.clip(last_step_loss_diff, 0.00001, 1)
        apinput = torch.from_numpy(apinput).to(args.device).float()
        ampphase = apmodel(apinput)
        # writer.add_graph(apmodel, apinput)
        ramp, rphase = ampphase[0][0][0], ampphase[0][0][1]
        amp, phase = rawtotrueap(ramp, rphase)
        # amp, phase = amp.to(args.device), phase.to(args.device)

        # now get target function
        target_function = task_family_train.get_target_function(amp, phase)

        # get the training points
        train_inputs = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
        # append phase amp
        train_inputs_orig = train_inputs.clone()
        print(train_inputs.shape)
        train_inputs = torch.cat((train_inputs, (amp * torch.ones((train_inputs.shape[0], 1))).to(args.device)),
                                 dim=1)
        train_inputs = torch.cat((train_inputs, (phase * torch.ones((train_inputs.shape[0], 1))).to(args.device)),
                                 dim=1)
        print(train_inputs.shape)

        # now train main network on this task for x, steps
        for niu in range(args.num_inner_updates):
            # forward through model
            train_outputs = model(train_inputs)
            writer.add_graph(model, train_inputs)

            # get targets
            train_targets = target_function(train_inputs_orig).to(args.device)

            # ------------ update on current task ------------

            # compute loss for current task
            print(train_outputs.device)
            print(train_targets.device)
            print(amp.device)
            print(phase.device)
            print(ramp.device)
            print(rphase.device)
            task_loss = F.mse_loss(train_outputs, train_targets)

            # keep track of the first step loss
            if niu == 0:
                first_step_loss = task_loss.item()

            # compute gradient for model
            optimizer.zero_grad()
            task_loss.backward()  # does this calculate the loss for the ap stuff too?

            # update model
            optimizer.step()

        # keep track of the difference between the first and last step loss
        last_step_loss_diff = first_step_loss - task_loss.item()

        # now update the task selector network
        if i_iter != 0:
            for name, param in apmodel.named_parameters():
                print(name, param.grad)
            #ap_optimizer.step()

        # ------------ logging ------------

        if i_iter % log_interval == 0:
            exit(0)

            # keep track of time spent evaluating
            logger.time_eval()

            # evaluate on training set
            loss_mean, loss_conf = eval_cva(args, copy.deepcopy(model), task_family_train,
                                            args.num_eval_updates)
            logger.train_loss.append(loss_mean)
            logger.train_conf.append(loss_conf)

            # evaluate on test set
            loss_mean, loss_conf = eval_cva(args, copy.deepcopy(model), task_family_valid,
                                            args.num_eval_updates, offset=args.num_fns)
            logger.valid_loss.append(loss_mean)
            logger.valid_conf.append(loss_conf)

            # evaluate on validation set
            loss_mean, loss_conf = eval_cva(args, copy.deepcopy(model), task_family_test,
                                            args.num_eval_updates, offset=args.num_fns+10000)
            logger.test_loss.append(loss_mean)
            logger.test_conf.append(loss_conf)

            # keep track of elapsed time
            logger.track_time()

            # save logging results
            utils.save_obj(logger, path)

            # save best model
            if logger.valid_loss[-1] == np.min(logger.valid_loss):
                print('saving best model at iter', i_iter)
                logger.best_valid_model = copy.deepcopy(model)

            # visualise results
            if args.task == 'celeba':
                task_family_train.visualise(task_family_train, task_family_test, copy.deepcopy(logger.best_valid_model),
                                            args, i_iter)

            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    return logger


def rawtotrueap(ramp, rphase):
    # ramp and rphase are between 0, 1 - return them between 0.1 and 5/pi respectively
    amp = 5*torch.clamp(ramp+0.02, 0.02, 1)
    phase = np.pi*torch.clamp(rphase+0.1/np.pi, 0.1/np.pi, 1)
    return amp, phase


def eval_game(args, model, task_family, num_updates, n_tasks=100, return_gradnorm=False, offset=0):
    # logging
    losses = []
    gradnorms = []

    opt = optim.Adam(model.parameters(), lr=args.lr_inner)

    # --- inner loop ---
    # need to update this to where it learns the embedding parameters, then test
    # but CAVIA is bette set up for the few shot scheme.  Not sure we are.

    for t in range(n_tasks):
        # get the task family
        input_range = task_family.get_input_range().to(args.device)

        # sample a task
        target_function, ids = task_family.sample_task()

        # reset context parameters
        # model.reset_context_params()

        # get data for current task
        curr_inputs = task_family.sample_inputs(args.k_shot_eval, args.use_ordered_pixels).to(args.device)
        # append id's
        curr_inputs_orig = curr_inputs.clone()
        curr_inputs = torch.cat((curr_inputs, ((ids+offset)*torch.ones((curr_inputs.shape[0], 1))).to(args.device)), dim=1)
        curr_targets = target_function(curr_inputs_orig)

        # ------------ update on current task ------------
        # how do I know how many iterations to update for here?
        for _ in range(1, num_updates + 1):
            opt.zero_grad()
            # forward pass
            curr_outputs = model(curr_inputs)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # compute gradient wrt context params
            # can add the create_graph=not args.first_order here, but this makes it cva
            task_loss.backward()
            # the above stores the gradients in each tensors .grad parameter, keep track of those

            opt.step()
            #
            # # compute gradient wrt context params
            # task_gradients = \
            #     torch.autograd.grad(task_loss, model.context_params, create_graph=not args.first_order)[0]
            #
            # # update context params
            # if args.first_order:
            #     model.context_params = model.context_params - args.lr_inner * task_gradients.detach()
            # else:
            #     model.context_params = model.context_params - args.lr_inner * task_gradients
            #
            # # keep track of gradient norms
            # gradnorms.append(task_gradients[0].norm().item())

        # ------------ logging ------------

        # compute true loss on entire input range
        model.eval()
        input_range_orig = input_range.clone()
        input_range = torch.cat((input_range, ((ids+offset)*torch.ones((input_range.shape[0], 1))).to(args.device)), dim=1)
        losses.append(F.mse_loss(model(input_range), target_function(input_range_orig)).detach().item())
        model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)
