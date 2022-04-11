def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Active Learning Cycle Research", add_help=add_help)

    parser.add_argument("--activation_function", default="relu", type=str)

    parser.add_argument("--batch_size", default=200, type=int)
    
    parser.add_argument("--cal_data_factor", default=0.3, type=float)

    parser.add_argument("--calibrator", default="histogram", type=str)

    parser.add_argument("--criterion", default="cel", type=str)

    parser.add_argument("--dropout_rate", default=0.3, type=float)

    parser.add_argument("--evaluation_mode", default="single", type=str)

    parser.add_argument("--hidden_layer_size", default=150, type=int)

    parser.add_argument("--input_size", default=784, type=int)

    parser.add_argument("--learning_rate", default=1e-3, type=float)

    parser.add_argument("--num_alc", default=18, type=int)

    parser.add_argument("--num_bins", default=15, type=int)
    
    parser.add_argument("--num_classes", default=10, type=int)

    parser.add_argument("--num_epochs", default=100, type=int)

    parser.add_argument("--num_labeld_samples", default=20, type=int)

    parser.add_argument("--num_mc_passes", default=10, type=int)

    parser.add_argument("--num_purchases", default=10, type=int)

    parser.add_argument("--num_samples", default=1000, type=int)

    parser.add_argument("--num_test_samples", default=1000, type=int)

    parser.add_argument("--optimizer", default="Adam", type=str)

    parser.add_argument("--output_size", default=10, type=int)

    parser.add_argument("--path_to_data", default='./data', type=str)

    parser.add_argument("--path_to_model", default="./model_params.txt", type=str)

    parser.add_argument("--query_strategy", default="entropy", type=str)

    parser.add_argument("--random_seed", default=38, type=int)

    parser.add_argument("--test_batch_size", default=64, type=int)

    parser.add_argument("--train_batch_size", default=64, type=int)

    return parser
