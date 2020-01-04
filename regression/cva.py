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
from cva_model import CvaModel
from logger import Logger


def run(args, log_interval=5000, rerun=False):
    assert not args.maml

    # see if we already ran this experiment
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/cva_{}_result_files/'.format(code_root, args.task)):
        os.mkdir('{}/cva_{}_result_files/'.format(code_root, args.task))
    path = '{}/cva_{}_result_files/'.format(code_root, args.task) + args.id  # utils.get_path_from_args(args)

    if os.path.exists(path + '.pkl') and not rerun:
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(args.seed)

    # --- initialise everything ---

    # get the task family
    if args.task == 'sine':
        task_family_train = tasks_sine.RegressionTasksSinusoidal(args.num_fns)
        task_family_valid = tasks_sine.RegressionTasksSinusoidal(10000)
        task_family_test = tasks_sine.RegressionTasksSinusoidal(10000)
    elif args.task == 'celeba':
        task_family_train = tasks_celebA.CelebADataset('train', device=args.device)
        task_family_valid = tasks_celebA.CelebADataset('valid', device=args.device)
        task_family_test = tasks_celebA.CelebADataset('test', device=args.device)
    else:
        raise NotImplementedError

    # initialise network
    model = CvaModel(n_in=task_family_train.num_inputs,
                     n_out=task_family_train.num_outputs,
                     num_context_params=args.num_context_params,
                     num_tasks=args.num_fns + 20000,
                     n_hidden=args.num_hidden_layers,
                     device=args.device
                     ).to(args.device)

    # intitialise meta-optimiser
    # (only on shared params - context parameters are *not* registered parameters of the model)
    meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)
    embedding_optimizer = optim.Adam(model.embedding.parameters(), args.lr_inner)

    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    # --- main training loop ---

    for i_iter in range(args.n_iter):

        # initialise meta-gradient
        meta_gradient = [0 for _ in range(len(model.state_dict()))]

        # sample tasks
        target_functions, idxs = task_family_train.sample_tasks(args.tasks_per_metaupdate)

        # if flag is set, reinitialize embedding
        if args.reinit_emb:
            model.zero_embeddings()

        # --- inner loop ---

        for t in range(args.tasks_per_metaupdate):

            # We want to be able to test 2 things:
            #   1) update the network once for every task (so the tasks_per_metaupdate = 1
            #   2) update the embeddings for every task, but then update the shared parameters wrt all tasks

            # reset private network weights
            # model.reset_context_params()  # not needed for cva

            # get data for current task
            train_inputs = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)

            # append id's
            train_inputs_orig = train_inputs.clone()
            train_inputs = torch.cat((train_inputs, (idxs[t]*torch.ones((train_inputs.shape[0], 1))).to(args.device)), dim=1)


            for _ in range(args.num_inner_updates):
                # forward through model
                train_outputs = model(train_inputs)

                # get targets
                train_targets = target_functions[t](train_inputs_orig)

                # ------------ update on current task ------------

                # compute loss for current task
                task_loss = F.mse_loss(train_outputs, train_targets)

                # compute gradient wrt context params
                # can add the create_graph=not args.first_order here, but this makes it cva
                embedding_optimizer.zero_grad()
                task_loss.backward()
                # the above stores the gradients in each tensors .grad parameter, keep track of those
                for i, param in enumerate(model.parameters()):
                    meta_gradient[i] += param.grad

                embedding_optimizer.step()

                # task_gradients = torch.autograd.grad(task_loss, model.embedding.parameters())[0]

                # update embeddings (remember that the tensors keep track of the gradients)

            # ------------ compute meta-gradient on test loss of current task ------------

            # get test data
            test_inputs = task_family_train.sample_inputs(args.k_meta_test, args.use_ordered_pixels).to(args.device)

            test_inputs_orig = test_inputs.clone()
            test_inputs = torch.cat((test_inputs, (idxs[t]*torch.ones((test_inputs.shape[0], 1))).to(args.device)), dim=1)

            # get outputs after update
            test_outputs = model(test_inputs)

            # get the correct targets
            test_targets = target_functions[t](test_inputs_orig)

            # compute loss after updating context (will backprop through inner loop)
            loss_meta = F.mse_loss(test_outputs, test_targets)

            # now the parameters hold the proper gradients in param.grad

            # compute gradient + save for current task
            task_grad = torch.autograd.grad(loss_meta, model.parameters())

            for i in range(len(task_grad)):
                # clip the gradient
                meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)

        # ------------ meta update ------------

        # assign meta-gradient
        for i, param in enumerate(model.parameters()):
            param.grad = meta_gradient[i] / args.tasks_per_metaupdate

        # dont update the embedding gradients if we updated them in inner loop, probably
        if args.num_inner_updates > 0:
            for name, param in model.named_parameters():
                if 'embedding' in name:
                    param.grad = None

        # do update step on shared model
        meta_optimiser.step()

        # ------------ logging ------------

        if i_iter % log_interval == 0:

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


def eval_cva(args, model, task_family, num_updates, n_tasks=100, return_gradnorm=False, offset=0):
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
