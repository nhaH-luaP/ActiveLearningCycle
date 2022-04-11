from comet_ml import Experiment
import random
import torch
import tqdm as tqdm
from os.path import exists
from models.simple import NeuralNetwork
from utils.misc import setup_seed
from utils.parser import get_args_parser
from alc import ActiveLearningCycle
from utils.misc import build_dataset


def main(args):
    if torch.cuda.is_available():
        device =  torch.device("cuda")
    else:
        device = torch.device("cpu")
    #For Debugging on Laptop
    device = torch.device("cpu")
    ####Delete before transfer####

    print("Device used is",device)


    print("Setting up random seed...")
    setup_seed(args.random_seed)


    print("Creating/Loading Modelparameters...")
    model = NeuralNetwork(args)
    if exists(args.path_to_model):
        params = torch.load(args.path_to_model)
    else:
        params = model.state_dict()
        torch.save(params, args.path_to_model)
    

    print("Loading and building data...")
    train_data, test_data = build_dataset(args)


    print("Creating an Active Learning Cycle...")
    ALC = ActiveLearningCycle(params, train_data, test_data, args)


    print("Setting up logging on comet-ml...")
    logger = Experiment(
        api_key="yTvP56YV3OlvUFHrLDDRTMfhw",
        project_name="activelearningcycle",
        workspace="nhah-luap",
    )
    #Save Hyperparameters used
    logger.log_parameters(vars(args))


    print("Learning...")
    for i in tqdm.tqdm(range(args.num_alc)):
        ALC.do_one_cycle(args, model, device, logger, i)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
