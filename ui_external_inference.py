import torch
from PIL import Image
import yaml
from torchvision import transforms


from nn_models import Encoder, ParamAwareMultiTailDecoder, EncoderDecoderModel
from nn_training import load_metadata_for_inference
from nn_dataset import denormalize

import os
from tqdm import tqdm


img_path = "./inference/sketch.png"
model_path = "./models/EncDecModel.pth"
meta_path = "./models/meta.yml"
output_path = "./inference/output.yml"

transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )


def inference():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata
    ranges, parameter_output_mapping, decoders = load_metadata_for_inference(meta_path)

    # Load the model
    encoder = Encoder()
    model = EncoderDecoderModel(encoder, decoders)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load the image
    img = Image.open(img_path).convert("L")
    img = transform(img)
    img = img.to(device)

    # print shapes
    print(img.shape)

    # Inference
    with torch.no_grad():
        output = model(img.unsqueeze(0))

    parsed_outputs = {}
    for decoder_name, decoder_outputs in output.items():
        print(f"Decoder: {decoder_name}")
        classification_outputs, regression_output = decoder_outputs
        for param_name, pred in classification_outputs.items():
            print(f"Classification: {param_name}")
            print(pred)
            param_type = ranges[param_name]["type"]
            print(f"Type: {param_type}")
            pred = denormalize(pred[0], ranges[param_name])
            parsed_outputs[param_name] = pred
        for param_name, pred in regression_output.items():
            print(f"Regression: {param_name}")
            print(pred)
            param_type = ranges[param_name]["type"]
            print(f"Type: {param_type}")
            pred = denormalize(pred[0], ranges[param_name])
            parsed_outputs[param_name] = pred

    # Save the output
    with open(output_path, "w") as f:
        yaml.dump(parsed_outputs, f)

def batch_inference(imgs: list, image_folder: str, pred_folder: str) -> list:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata
    ranges, parameter_output_mapping, decoders = load_metadata_for_inference(meta_path)

    # Load the model
    encoder = Encoder()
    model = EncoderDecoderModel(encoder, decoders)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    inference_outs = []

    # Load the images and perform inference
    progress_bar = tqdm(imgs)
    for i, s_image_path in enumerate(progress_bar):
        if not s_image_path.endswith(".png"):
            continue
        image_path = os.path.join(image_folder, s_image_path)
        # Load the image
        img = Image.open(image_path).convert("L")
        img = transform(img)
        img = img.to(device)
        # Inference
        with torch.no_grad():
            output = model(img.unsqueeze(0))

        parsed_outputs = {}
        for decoder_name, decoder_outputs in output.items():
            print(f"Decoder: {decoder_name}")
            classification_outputs, regression_output = decoder_outputs
            for param_name, pred in classification_outputs.items():
                print(f"Classification: {param_name}")
                print(pred)
                param_type = ranges[param_name]["type"]
                print(f"Type: {param_type}")
                pred = denormalize(pred[0], ranges[param_name])
                parsed_outputs[param_name] = pred
            for param_name, pred in regression_output.items():
                print(f"Regression: {param_name}")
                print(pred)
                param_type = ranges[param_name]["type"]
                print(f"Type: {param_type}")
                pred = denormalize(pred[0], ranges[param_name])
                parsed_outputs[param_name] = pred

        output_name = s_image_path.replace(".png", ".yml")
        output_path = os.path.join(pred_folder, output_name)
        # Save the output
        with open(output_path, "w") as f:
            yaml.dump(parsed_outputs, f)
        
        inference_outs.append(
            (image_path, output_path)
        )

        progress_bar.set_description(f"Processed {i+1} images")
    
    return inference_outs



