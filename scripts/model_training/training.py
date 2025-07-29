#Here we train custom CNN without using pretrained models and any optimization -> baseline
import torch
from pathlib import Path
import os
import sys
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss



# Check if the code is running in Colab
IN_COLAB = 'google.colab' in sys.modules

'''
if not IN_COLAB:
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    project_root = os.getenv('PROJECT_ROOT_PATH')
else:
    from google.colab import userdata
    # Set the project root path for Colab
    project_root = userdata.get("project_root_path")

# Check if the project root path is set correctly
if project_root is None:
    raise ValueError("PROJECT_ROOT_PATH environment variable is not set.")

'''
#PROJECT_ROOT_PATH="/home/jovyan/Documents/DSProject"

# Add the project root path to the system path
#sys.path.append(PROJECT_ROOT_PATH)

#from trainers.ImgClassification import ImgClassification
#from datasets.dataset import Subset
#from datasets.preprocessing import CustomDatasetPreprocessor
#from datasets.AIArtBench import AIArtbench


import importlib.util

def load_module(name, path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

flood_dataset = load_module("flood_dataset", "scripts/data_model/flood_dataset.py")
metrics = load_module("metrics", "scripts/model_training/metrics.py")
trainer = load_module("trainer", "scripts/model_training/trainer.py")

FloodDataset = flood_dataset.FloodDataset
SegmentationMetrics = metrics.SegmentationMetrics
ImgSegmentation = trainer.ImgSegmentation 


def debug_print(message, debug_mode):
    if debug_mode:
        print(message)


def get_device(debug_mode):
    # Checking device (CPU vs GPU)
    if torch.cuda.is_available():
        debug_print("CUDA (GPU) is available.", debug_mode)
    else:
        debug_print("CUDA (GPU) is not available. Training on CPU.", debug_mode)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_datasets(config):
    """
    Prepares the datasets for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary with dataset paths and preprocessing details.

    Returns:
        dict: Dictionary containing train, validation, and test datasets.
    """
    debug_mode = config['debug_mode']
    debug_print("Preprocessing dataset...", debug_mode)

    # Preprocess dataset and split

    train_dataset = FloodDataset(split_file=config['train_split'], transform=config['train_transform'], subset_size=config.get('subset_size'))
    val_dataset = FloodDataset(split_file=config['val_split'], transform=config['val_transform'])#, subset_size=config.get('subset_size'))
    test_dataset = FloodDataset(split_file=config['test_split'], transform=config['test_transform'])#, subset_size=config.get('subset_size'))

    debug_print(f"Train dataset length: {len(train_dataset)}", debug_mode)
    debug_print(f"Validation dataset length: {len(val_dataset)}", debug_mode)
    debug_print(f"Test dataset length: {len(test_dataset)}", debug_mode)

    return {
        "train_data": train_dataset,
        "val_data": val_dataset,
        "test_data": test_dataset,
        "train_classes": train_dataset.get_num_classes(),
        "val_classes": val_dataset.get_num_classes(),
        "test_classes": test_dataset.get_num_classes()
    }


def initialize_model_and_optimizer(config, classes, device):
    """
    Initializes the model, optimizer, scheduler, and loss function.

    Args:
        config (dict): Configuration dictionary with model and training settings.
        classes (list): Class labels for the dataset.
        device (torch.device): Device for training (CPU or GPU).

    Returns:
        dict: Dictionary containing the model, optimizer, scheduler, loss, and metrics.
    """
    debug_mode = config['debug_mode']
    model = config['model']
    model = model.to(device)

    #optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, nesterov=True)
    
    #trying adam as optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

    #trying another scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=2, verbose=True, min_lr=1e-5
    )
        
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['scheduler_gamma'])

    #loss_fn = torch.nn.CrossEntropyLoss()

    #trying more complicated loss to tackle class imbalance
    dice_loss = DiceLoss(mode="multiclass", from_logits=True)

    class_weights = config.get("class_weights", None)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    def loss_fn(logits, targets):
        ce = F.cross_entropy(logits, targets, weight=class_weights)
        dice = dice_loss(logits, targets)
        return 0.5 * ce + 0.5 * dice #0.7 * ce + 0.3 * dice - prioritize accuracy

    debug_print(f"Model: {model}", debug_mode)

    metric = SegmentationMetrics(num_classes=classes, focus_class=1)
    
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
        "loss_fn": loss_fn,
        "metric": metric
    }


def initialize_trainer(config, datasets, model_components, device):
    """
    Initializes the trainer class with model, datasets, and training settings.

    Args:
        config (dict): Configuration dictionary with training parameters.
        datasets (dict): Dictionary of prepared datasets.
        model_components (dict): Initialized model, optimizer, scheduler, and metrics.
        device (torch.device): Device for training/testing.

    Returns:
        ImgClassification: Instance of the ImgClassification trainer.
    """
    return ImgSegmentation(
        model=model_components['model'],
        optimizer=model_components['optimizer'],
        loss_fn=model_components['loss_fn'],
        lr_scheduler=model_components['scheduler'],
        train_metric=model_components['metric'],
        val_metric=SegmentationMetrics(num_classes=datasets["val_classes"], focus_class = 1),
        test_metric=SegmentationMetrics(num_classes=datasets["test_classes"], focus_class = 1),
        train_data=datasets["train_data"],
        val_data=datasets["val_data"],
        test_data=datasets["test_data"],
        device=device,
        num_epochs=config['epochs'],
        training_save_dir=Path(config['model_save_dir']),
        batch_size=config['batch_size'],
        val_frequency=config['val_frequency'],
        debug_mode=config['debug_mode'],
        patience=config['patience'],
        model_name=config['model_name']
    )


def train_model(config):
    """
    Trains the model using the ImgClassification trainer.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        ImgClassification: Trained ImgClassification trainer.
    """
    debug_mode = config['debug_mode']
    device = get_device(debug_mode)

    datasets = prepare_datasets(config)
    model_components = initialize_model_and_optimizer(config, classes=datasets["train_classes"], device=device)
    
    trainer = initialize_trainer(config, datasets, model_components, device)
    trainer.train()

    torch.save(trainer.model.state_dict(), Path(config['model_save_dir']) / f"{config['model_name']}_train.pth")
    trainer.dispose()

    return trainer


def test_model(config, model_name, trainer=None):
    """
    Tests the model on the test dataset.

    Args:
        config (dict): Configuration dictionary.
        trainer (ImgClassification, optional): Existing trainer object. If not provided, initializes a new one.

    Returns:
        None
    """
    debug_mode = config['debug_mode']
    device = get_device(debug_mode)

    # Use the existing trainer if provided, otherwise initialize a new one
    if not trainer:
        datasets = prepare_datasets(config)
        model_components = initialize_model_and_optimizer(config, classes=datasets["test_classes"], device=device)
        trainer = initialize_trainer(config, datasets, model_components, device)
        trainer.model.load_state_dict(torch.load(Path(config['model_save_dir']) / f"{model_name}.pth", map_location=device))

    test_loss, test_metric = trainer.test() #, all_water_probs
    return test_loss, test_metric #, all_water_probs
