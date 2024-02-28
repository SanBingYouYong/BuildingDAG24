import torch
from PIL import Image
import yaml
from torchvision import transforms


from nn_models import Encoder, ParamAwareMultiTailDecoder, EncoderDecoderModel
from nn_training import load_metadata_for_inference
from nn_dataset import denormalize


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
            if param_type == "bool":
                pred = bool(torch.argmax(pred, dim=1) == 0)
            elif param_type == "states":
                pred = int(torch.argmax(pred, dim=1))
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

