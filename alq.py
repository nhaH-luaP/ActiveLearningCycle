import torch
import random
from models.uncertainty import monte_carlo_dropout_pass


class ActiveLearningQuery():
    def __init__(self, ALC):
        self.ALC = ALC

    def query(self, args, model, unlabeld_data_loader, device):
        raise ValueError("Query not defined! use as abstract class.")


class EntropyQuery(ActiveLearningQuery):
    def __init__(self, ALC):
        super().__init__(ALC)

    def query(self, args, model, unlabeld_data_loader, device):
        #Calculating Entropy on unlabeld data pool
        unlabeld_mean, _ = monte_carlo_dropout_pass(model, args, unlabeld_data_loader, device)
        unlabeld_entropy = -torch.sum(unlabeld_mean*torch.log(unlabeld_mean), dim=1)
        unlabeld_entropy = torch.tensor([0 if torch.isnan(x) else x for x in unlabeld_entropy])

        for x in range(args.num_purchases):
            idx = torch.argmax(unlabeld_entropy)
            self.ALC.train_pool_idx.append(self.ALC.unlabeld_pool_idx.pop(idx))
            unlabeld_entropy = torch.cat([unlabeld_entropy[0:idx], unlabeld_entropy[idx+1:]])


class RandomQuery(ActiveLearningQuery):
    def __init__(self, ALC):
        super().__init__(ALC)

    def query(self, args, model, unlabeld_data_loader, device):
        for x in range(args.num_purchases):
            idx = random.sample(range(len(self.ALC.unlabeld_pool_idx)), 1)[0]
            self.ALC.train_pool_idx.append(self.ALC.unlabeld_pool_idx.pop(idx))


class MixedQuery(ActiveLearningQuery):
    def __init__(self, ALC):
        super().__init__(ALC)
        self.rquery = RandomQuery(ALC)
        self.equery = EntropyQuery(ALC)
        self.method = "A"

    def query(self, args, model, unlabeld_data_loader, device):
        if self.method == "A":
            self.rquery.query(args, model, unlabeld_data_loader, device)
            self.method = "B"
        else:
            self.equery.query(args, model, unlabeld_data_loader, device)
            self.method = "A"


class SoftmaxQuery(ActiveLearningQuery):
    def __init__(self, ALC):
        super().__init__(ALC)

    def query(self, args, model, unlabeld_data_loader, device):
        #Create a tensor containing all predictions and their softmaxvalue
        L = []
        model.to(device)
        for data, _ in unlabeld_data_loader:
            data = data.to(device)
            L.append(torch.max(model(data).softmax(dim=-1), dim=-1)[0])
        prediction_certaintys = torch.cat(L)

        for x in range(args.num_purchases):
            idx = torch.argmin(prediction_certaintys)
            self.ALC.train_pool_idx.append(self.ALC.unlabeld_pool_idx.pop(idx))
            prediction_certaintys = torch.cat([prediction_certaintys[0:idx], prediction_certaintys[idx+1:]])