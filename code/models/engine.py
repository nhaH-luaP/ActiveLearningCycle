import torch
from calibration.utils import get_ece

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
def evaluate(model, args, data_loader, device, calibrator=None):
    #Put model into evaluation mode and onto the chosen device
    model.eval()
    model = model.to(device)

    #Initialize counters to calculate accuracy, loss, confidences etc. ...
    num_false_predictions = 0
    num_correct_predictions = 0
    total_overconfidence = 0
    total_underconfidence = 0
    loss = 0

    #Define loss function (criterion)
    if args.criterion == "cel":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    #Initialize Container for data predictions to calculate ECE later
    p = []
    l = []

    #Calculate Accuracy
    for (data, targets) in data_loader:
        #Get predictions and confidences of the net
        data, targets = data.to(device), targets.to(device)
        output = model(data).softmax(dim=-1)
        if calibrator != None:
            output = torch.tensor(calibrator.calibrate(output))
        p.append(output)
        l.append(targets)
        loss += criterion(output, targets)
        confidences, predictions = torch.max(output, dim=-1)

        #Create a dictionary for each sample and sort them by confidence intervalls (bins)
        for i in range(data.shape[0]):
            if predictions[i] == targets[i]:
                num_correct_predictions += 1
                total_underconfidence += (1-confidences[i])
            else:
                num_false_predictions += 1
                total_overconfidence += confidences[i]

    num_samples = num_false_predictions+num_correct_predictions
    probs, labels = torch.cat(p), torch.cat(l)
    ECE, TCE = get_ece(probs, labels, mode="marginal"), get_ece(probs, labels, mode="top-label") 
    test_loss = loss.item()/num_samples
    accuracy = num_correct_predictions/num_samples
    underconfidence = total_underconfidence.item()/num_correct_predictions
    overconfidence = total_overconfidence.item()/num_false_predictions

    return accuracy, test_loss, ECE, TCE, underconfidence, overconfidence