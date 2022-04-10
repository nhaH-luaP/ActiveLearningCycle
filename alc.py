import torch
import math
import random
import torch.nn as nn
from models.engine import train_one_epoch, evaluate
from models.uncertainty import monte_carlo_dropout_pass
from models.temperature_scaling import ModelWithTemperature
from alq import EntropyQuery, SoftmaxQuery
from alq import RandomQuery
from alq import MixedQuery


class ActiveLearningCycle:
    def __init__(self, params, dataset, test_dataset, args):
        #Dataset represents all available Data
        self.dataset = dataset

        #Testset is for evaluating model-performance only
        self.testset = test_dataset

        #Known and labeld data-pool to train the model on / unknown data-pool
        self.train_pool_idx = random.sample(range(args.num_samples), args.num_labeld_samples)
        self.unlabeld_pool_idx = [x for x in range(args.num_samples) if x not in self.train_pool_idx]

        #model parameters to start every cycle with
        self.params = params

        #Choose a query strategy
        if args.query_strategy == "entropy":
            self.ALQ = EntropyQuery(self)

        elif args.query_strategy == "random":
            self.ALQ = RandomQuery(self)

        elif args.query_strategy == "mixed":
            self.ALQ = MixedQuery(self)

        elif args.query_strategy == "softmax":
            self.ALQ = SoftmaxQuery(self)

        else:
            self.ALQ = RandomQuery(self, args)

    def do_one_cycle(self, args, model, device, logger, cycle):
        #Initialize model with standard parameters and initialize data loaders
        model.load_state_dict(self.params)
        train_data_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.Subset(self.dataset, self.train_pool_idx) , batch_size = args.num_purchases, shuffle = False)
        unlabeld_data_loader = torch.utils.data.DataLoader(dataset = torch.utils.data.Subset(self.dataset, self.unlabeld_pool_idx) , batch_size = args.num_purchases, shuffle = False)
        test_data_loader = torch.utils.data.DataLoader(dataset = self.testset, batch_size = args.test_batch_size, shuffle = False)


        #Train the network on the labeld pool and tracking loss
        train_loss = 0
        for i in range(args.num_epochs):
            train_loss += train_one_epoch(model, args, train_data_loader, device)
        avg_train_loss = train_loss.item()/args.num_epochs
        

        #Apply temperature-scaling to model if args.calibrate is set
        if args.t_scaling == True:
            model = ModelWithTemperature(model)


        #Evaluating performance on test data
        if args.evaluation_mode == "single":
            test_accuracy, avg_test_loss, ECE, MCE, underconfidence, overconfidence = evaluate(model, args, test_data_loader, device)
        else:
            test_accuracy, avg_test_loss, ECE, MCE, underconfidence, overconfidence = evaluate(model, args, test_data_loader, device)


        #Logging
        logger.log_metric("avg_train_loss", avg_train_loss, step=cycle)
        logger.log_metric("test_accuracy", test_accuracy, step=cycle)
        logger.log_metric("avg_test_loss", avg_test_loss, step=cycle)
        logger.log_metric("ece", ECE, step=cycle)
        logger.log_metric("mce", MCE, step=cycle)
        logger.log_metric("underconfidence", underconfidence, step=cycle)
        logger.log_metric("overconfidence", overconfidence, step=cycle)


        #Update pools
        self.ALQ.query(args, model, unlabeld_data_loader, device)