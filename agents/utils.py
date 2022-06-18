import os

import torch

SAVE_MODEL_LABEL = "model"
SAVE_OPTIMIZER_LABEL = "optimizer"
SAVE_FILE_EXTENSION = ".pt"
SAVED_MODELS_FOLDER = "saved_models/"


def get_current_directory():
    return os.path.dirname(os.path.abspath(__file__)) + "/"


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_torch():
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        cuda_id = torch.cuda.current_device()
        print(f"CUDA current device id: {cuda_id}")
        print(f"CUDA device name: {torch.cuda.get_device_name(cuda_id)}")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")


def generate_filepath(filename: str):
    current_dir = get_current_directory()
    filepath = f"{current_dir}{SAVED_MODELS_FOLDER}{filename}{SAVE_FILE_EXTENSION}"
    return filepath


def save(model: torch.nn.Module, optimizer: torch.optim.Optimizer, name: str):
    """
    Save a model and its optimizer to a file
    :param model: The model to save
    :param optimizer: The optimizer to save
    :param name: The name for the file
    :return: None
    """
    to_save = {
        SAVE_MODEL_LABEL: model.state_dict(),
        SAVE_OPTIMIZER_LABEL: optimizer.state_dict(),
    }
    filepath = generate_filepath(name)
    torch.save(to_save, filepath)


def load(model: torch.nn.Module, optimizer: torch.optim.Optimizer, name: str):
    """
    Load a model and its optimizer from a file
    :param model: The model to load into
    :param optimizer: The optimizer to load into
    :param name: The name of the file
    :return: None
    """
    filepath = generate_filepath(name)
    try:
        loaded = torch.load(filepath, map_location=get_torch_device())
        model.load_state_dict(loaded[SAVE_MODEL_LABEL])
        optimizer.load_state_dict(loaded[SAVE_OPTIMIZER_LABEL])
    except FileNotFoundError:
        print(f"Could not load model, file not found: \n{filepath}")
