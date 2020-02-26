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
from logger import Logger
from task_pred import TaskPredModel, CvaModelVanilla


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

    tmp_embedding = torch.nn.Embedding(args.num_fns + 20000, 2)


    # First, we are going to train a network to predict which task we are doing right now
    task_predictor = TaskPredModel(2*args.k_meta_train, args.tp_hidden_layers, args.device).to(args.device)
    tp_optimizer = optim.Adam(task_predictor.parameters(), args.lr_inner)
    for i in range(args.n_iter_tp):
        target_functions, idxs = task_family_train.sample_tasks(args.tasks_per_metaupdate)
        for tf, tfid in zip(target_functions, idxs):
            # zero gradient
            tp_optimizer.zero_grad()

            train_inputs = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
            # get targets
            train_targets = tf(train_inputs)

            # concatenate x and y values
            inputs = torch.cat((train_inputs, train_targets), dim=1).to(args.device).flatten().to(args.device)

            # this doesn't happen in cavia
            train_outputs = task_predictor(inputs.to(args.device)).to(args.device)

            # compute loss for current task
            task_loss = F.mse_loss(train_outputs.flatten(), inputs.to(args.device)/5)

            # update
            task_loss.backward()
            tp_optimizer.step()

            # print info
            if i % 1000 == 0:
                print(f'itr: {i}, tp loss: {task_loss.item()}')  #, eval loss: {eval_tp(args, copy.deepcopy(task_predictor), task_family_test, 1000)}')

    print('Finished training the task predictor')
    # initialise network
    model = CvaModelVanilla(n_in=task_family_train.num_inputs,
                            n_out=task_family_train.num_outputs,
                            num_context_params=args.num_context_params,
                            n_hidden=args.num_hidden_layers,
                            device=args.device,
                            ).to(args.device)

    meta_optimiser = optim.Adam(model.parameters(), args.lr_meta)

    # initialise loggers
    logger = Logger(args=args)
    logger.best_valid_model = copy.deepcopy(model)

    # --- main training loop ---
    for i_iter in range(args.n_iter):

        target_functions, idxs = task_family_train.sample_tasks(args.tasks_per_metaupdate)

        # --- inner loop ---
        for t in range(args.tasks_per_metaupdate):

            # get data for current task
            train_inputs = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)

            # zero gradient
            tp_optimizer.zero_grad()

            train_inputs = task_family_train.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
            # get targets
            train_targets = target_functions[t](train_inputs)

            # concatenate x and y values
            xy = torch.cat((train_inputs, train_targets), dim=1).to(args.device)

            # get the context from task predictor
            context = task_predictor.get_embedding(xy.flatten())

            # Concatenate the context with the input
            x = torch.cat((train_inputs, torch.from_numpy(np.tile(context.cpu().numpy(), (train_inputs.shape[0], 1) )).to(args.device)), dim=1).to(args.device)

            # forward pass through model
            train_outputs = model(x)

            # compute loss for current task
            task_loss = F.mse_loss(train_outputs, train_targets)

            # update
            task_loss.backward()
            meta_optimiser.step()

            # may need this to do batch updates instead of update on every
            # # compute gradient + save for current task
            # task_grad = torch.autograd.grad(loss_meta, model.parameters())
            #
            # for i in range(len(task_grad)):
            #     # clip the gradient
            #     meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)

        # ------------ meta update ------------

        # only need this if I change to updating only after x tasks
        # # assign meta-gradient
        # for i, param in enumerate(model.parameters()):
        #     param.grad = meta_gradient[i] / args.tasks_per_metaupdate
        #
        # # dont update the embedding gradients if we updated them in inner loop, probably
        # if args.num_inner_updates > 0:
        #     for name, param in model.named_parameters():
        #         if 'embedding' in name:
        #             param.grad = None
        #
        # # do update step on shared model and embeddings
        # meta_optimiser.step()
        # embedding_optimizer.step()
        #
        # # take a step on the learning rate scheduler
        # model_scheduler.step()
        # emb_scheduler.step()

        # ------------ logging ------------

        if i_iter % log_interval == 0:

            # keep track of time spent evaluating
            logger.time_eval()

            # evaluate on training set
            loss_mean, loss_conf, _ = eval_cva(args, copy.deepcopy(model), task_predictor, task_family_train)
            logger.train_loss.append(loss_mean)
            logger.train_conf.append(loss_conf)

            # evaluate on test set
            loss_mean, loss_conf, _ = eval_cva(args, copy.deepcopy(model), task_predictor, task_family_valid)
            logger.valid_loss.append(loss_mean)
            logger.valid_conf.append(loss_conf)

            # evaluate on validation set
            loss_mean, loss_conf, ft_model = eval_cva(args, copy.deepcopy(model), task_predictor, task_family_test)
            logger.test_loss.append(loss_mean)
            logger.test_conf.append(loss_conf)

            # keep track of elapsed time
            logger.track_time()

            # save logging results
            utils.save_obj(logger, path)

            # save best model, and the finetuned model
            if logger.valid_loss[-1] == np.min(logger.valid_loss):
                print('saving best model at iter', i_iter)
                logger.best_valid_model = copy.deepcopy(model)
                logger.best_ft_model = copy.deepcopy(ft_model)

            # visualise results
            if args.task == 'celeba':
                task_family_train.visualise(task_family_train, task_family_test, copy.deepcopy(logger.best_valid_model),
                                            args, i_iter)

            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    return logger


def eval_cva(args, model, task_predictor, task_family, n_tasks=100, return_gradnorm=False, offset=0):

    losses = []

    target_functions, idxs = task_family.sample_tasks(n_tasks)

    # --- inner loop ---
    for t in range(n_tasks):
        # get data for current task
        train_inputs = task_family.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
        # get targets
        train_targets = target_functions[t](train_inputs)

        # concatenate x and y values
        xy = torch.cat((train_inputs, train_targets), dim=1).to(args.device)

        # get the context from task predictor
        context = task_predictor.get_embedding(xy.flatten())

        # Concatenate the context with the input
        x = torch.cat((train_inputs,
                       torch.from_numpy(np.tile(context.cpu().numpy(), (train_inputs.shape[0], 1))).to(args.device)),
                      dim=1).to(args.device)

        # forward pass through model
        train_outputs = model(x)

        # compute loss for current task
        task_loss = F.mse_loss(train_outputs, train_targets)

        losses.append(task_loss.item())

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), copy.deepcopy(model)
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), []


def eval_tp(args, model, task_family, n_tasks=100):
    # First, we are going to train a network to predict which task we are doing right now
    losses = []
    task_predictor = TaskPredModel(2 * args.k_meta_train, 1, args.tp_hidden_layers, args.device)
    with torch.no_grad():
        target_functions, idxs = task_family.sample_tasks(n_tasks)
        for tf, tfid in zip(target_functions, idxs):
            train_inputs = task_family.sample_inputs(args.k_meta_train, args.use_ordered_pixels).to(args.device)
            train_targets = tf(train_inputs)

            # concatenate x and y values
            inputs = torch.cat((train_inputs, train_targets), dim=1).flatten().to(args.device)

            # this doesn't happen in cavia
            train_outputs = model(inputs)

            # compute loss for current task
            task_loss = F.mse_loss(train_outputs, (tfid * torch.ones((train_outputs.shape[0], 1))).to(args.device))
            losses.append(task_loss.item())
    return np.mean(np.asarray(losses)), np.std(np.asarray(losses))