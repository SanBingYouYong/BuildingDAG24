import os
import datetime
import shutil

import torch
import torch.optim as optim

from nn_models import *
from nn_dataset import *
from nn_training import train, test, load_metadata
from nn_visualize import visualize_loss
from nn_acc import acc_discrete
from performance import calculate_performance



def pipeline(dataset_name: str="DAGDataset100_100_5", 
             single_decoder: str=None,
             epochs: int=10,
             batch_size: int=32,
             lr: float=0.001,
             lx_regularizor: int=2,
             seed: int=0, 
             results_num: int=5,
             additional_notes: str=None):
    '''
    Train, test with best weights, visualize curves, print out results pairs, calcuulate acc for discrete.
    '''
    if not os.path.exists(f"./datasets/{dataset_name}"):
        raise FileNotFoundError(f"Dataset {dataset_name} not found")
    
    if seed != -1:
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    print(f"Dataset: {dataset_name}")
    dataset = DAGDatasetSingleDecoder(single_decoder, dataset_name) if single_decoder else DAGDataset(dataset_name)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.8, 0.1, 0.1)
    train_loader, val_loader, test_loader = create_dataloaders_of(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
    print(f"Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    encoder = Encoder()
    print(f"Loaded Encoder")
    ranges, parameter_output_mapping, decoders, switches, batch_cam_angles = load_metadata(dataset_name, single_decoder=single_decoder)
    print(f"Loaded {len(decoders)} decoders")
    model = EncoderDecoderModel(encoder, decoders)

    criterion = EncDecsLoss(decoders, switches, lx_regularizor=lx_regularizor)
    # criterion = custom_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    tag = "ovf_dropout_l1"
    results_name = f"results_{tag}.yml"

    os.makedirs("./models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model_name = f"./models/model_{dataset_name}_{timestamp}.pth"
    loss_name = f"./models/model_{dataset_name}_{timestamp}_loss.yml"
    train(model, criterion, optimizer, train_loader, val_loader, epochs=epochs, seed=-1, model_save_path=model_name, loss_save_path=loss_name)
    test_res = test(model, model_name, test_loader, criterion, ranges, results_save_path=results_name)
    # copy the meta.yml from dataset to models
    os.system(f"cp ./datasets/{dataset_name}/meta.yml ./models/model_{dataset_name}_{timestamp}_meta.yml")

    # visualize the loss curve
    visualize_loss(loss_name)

    # print out some results
    for param_name in test_res:
        retrieve = test_res[param_name][:results_num]
        print(f"Parameter: {param_name}")
        for i, (pred, target) in enumerate(retrieve):
            print(f" - Prediction: {pred}, Target: {target}")

    # calculate acc for discrete variables
    acc_discrete(results_name)
    # back up performance.yml
    shutil.copyfile(results_name, f"./performance_{tag}.yml")

    # calculate performance
    calculate_performance()
    # backup pdf
    shutil.copyfile("performance.pdf", f"./performance_{tag}.pdf")

    # record some optional notes
    if additional_notes:
        notes_path = model_name.replace(".pth", "_notes.txt")
        with open(notes_path, "w") as file:
            file.write(additional_notes)
        
    return model_name, loss_name, f"results.yml"



if __name__ == "__main__":
    # avoid unnecessary OOM
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    dataset_name = "DAGDataset100_100_5"
    # single_decoder = "Building Mass Decoder"
    single_decoder = None

    '''
    change tag name
    change arch if needed
    change lx params
    change notes
    '''
    epochs = 5
    batch_size = 32
    lx = 1
    notes = f"batch size: {batch_size}; dataset: {dataset_name}; 1+[1, 1] model with 0.5 dropout and l1 regularization on 10k"
    pipeline(dataset_name, single_decoder, epochs=epochs, batch_size=batch_size, lx_regularizor=lx, additional_notes=notes)

    # unset the env var
    # os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
