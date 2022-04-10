import torch

def calibration_error(args, Bins):
    ECE = 0
    MCE = 0
    num_samples = 0
    for Bin in Bins:
        #Check if its an empty Bin
        if len(Bin) > 0:
            #Initialize some variables
            sum_confidences = 0
            num_bin_samples = 0
            num_correct = 0

            #Analyze the Bin
            for sample in Bin:
                num_bin_samples += 1
                sum_confidences+= sample["conf"]
                if sample["pred"] == sample["target"]:
                    num_correct += 1

            #Calculating confidence, accuracy and resulting calibration error
            confidence = sum_confidences/len(Bin)
            accuracy = num_correct/num_bin_samples
            calibration_error = abs(accuracy-confidence)
            num_samples += num_bin_samples

            #Update ECE and MCE
            if calibration_error > MCE:
                MCE = calibration_error
            ECE += len(Bins)*calibration_error

    return ECE*(1/num_samples), MCE