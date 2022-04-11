import torch

@torch.no_grad()
def monte_carlo_dropout_pass(model, args, data_loader, device, calibrator):
    model = model.train().to(device)
    predictions = []
    for i in range(args.num_mc_passes):
        model.update_dropout_masks()
        p = []
        for images, _ in data_loader:
            images = images.to(device)
            output = model(images).softmax(dim=-1) # shape (n_samples, n_classes)
            if calibrator != None:
                output = torch.tensor(calibrator.calibrate(output.detach().numpy()))
            p.append(output)
        predictions.append(torch.cat(p))
    predictions = torch.stack(predictions, 0) # shape (n_mc_passes, n_samples, n_classes)
    return torch.mean(predictions, dim=0), torch.std(predictions, dim=0)