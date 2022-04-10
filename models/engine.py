import torch
from calibration.utils import get_ece
from models.uncertainty import monte_carlo_dropout_pass
from models.calibration import calibration_error

def train_one_epoch(model, args, data_loader, device):
    #Put model into training mode
    model.train()

    #Put model on device selected for training
    model = model.to(device)

    #Choose an optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)

    #Choose a loss function
    if args.criterion == "cel":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    #Initialize variables to track loss
    train_loss = 0
    n_samples = 0

    #Train the model for one epoch on the data in the data_loader given
    for i,(images,targets) in enumerate(data_loader):
        #Put data onto the same device as the model
        images, targets = images.to(device), targets.to(device)
        #Calculate model outputs of given data
        output = model(images)
        #Calculate the loss of outputs compared to true targets
        loss = criterion(output, targets)
        #Reset gradient history of optimizer
        optimizer.zero_grad()
        #Count loss and number of samples
        n_samples += len(images)
        train_loss += loss*len(images)
        #Calculate gradients
        loss.backward()
        #Update model parameters
        optimizer.step()
    
    return train_loss/n_samples


@torch.no_grad()
def evaluate(model, args, data_loader, device):
    #Put model into evaluation mode and onto the chosen device
    model.eval()
    model = model.to(device)

    #Initialize counters to calculate accuracy, loss, confidences etc. ...
    num_false_predictions = 0
    num_correct_predictions = 0
    num_samples = 0
    total_overconfidence = 0
    total_underconfidence = 0
    loss = 0

    #Define loss function (criterion)
    if args.criterion == "cel":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    #Initialize Container for data predictions to calculate ECE later
    Bins = [[] for i in range(args.num_bins)]

    #Calculate Accuracy
    for (data, targets) in data_loader:
        #Get predictions and confidences of the net
        data, targets = data.to(device), targets.to(device)
        output = model(data).softmax(dim=1)
        loss += criterion(output, targets)
        confidences, predictions = torch.max(output, dim=1)
        num_samples += data.shape[0]

        #Create a dictionary for each sample and sort them by confidence intervalls (bins)
        for i in range(data.shape[0]):
            if predictions[i] == targets[i]:
                num_correct_predictions += 1
                total_underconfidence += (1-confidences[i])
            else:
                num_false_predictions += 1
                total_overconfidence += confidences[i]
            d = {}
            d["pred"] = predictions[i].item()
            d["conf"] = confidences[i].item()
            d["target"] = targets[i].item()
            Bins[int(args.num_bins*confidences[i].item()-0.00001)].append(d)

    ECE, MCE = calibration_error(args, Bins)
    test_loss = loss.item()/num_samples
    accuracy = num_correct_predictions/num_samples
    underconfidence = total_underconfidence.item()/num_correct_predictions
    overconfidence = total_overconfidence.item()/num_false_predictions

    return accuracy, test_loss, ECE, MCE, underconfidence, overconfidence