import numpy
import random
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def setup_seed(seed):
    random.seed(seed)                          
    numpy.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True     

def flatten_img(x):
    return x.flatten()

def build_dataset(args):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(flatten_img)])
    train_dataset = dsets.MNIST(root = args.path_to_data, train = True,transform = transform,download = True)
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [args.num_samples,60000-args.num_samples])
    test_dataset = dsets.MNIST(root = args.path_to_data, train = False ,transform = transform,download = True)
    test_dataset, _ = torch.utils.data.random_split(test_dataset, [args.num_test_samples,10000-args.num_test_samples])
    return train_dataset, test_dataset
