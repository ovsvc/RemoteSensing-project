import torch, os
from typing import Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
from torch.cuda.amp import autocast, GradScaler



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

import importlib.util

def load_module(name, path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wandb_logger = load_module("wandb_logger", "scripts/model_training/wandb_logger.py")
WandBLogger = wandb_logger.WandBLogger


class ImgSegmentation:
    """
    Class that stores the logic for training and testing a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        test_metric,
        train_data,
        val_data,
        test_data,
        device,
        num_epochs, 
        training_save_dir,
        model_name,
        debug_mode = False,
        batch_size= 4,
        val_frequency = 5,
        patience= 3
    ) -> None:
        """
        Initializes the trainer with model, data, metrics, etc.
        """

        self.model = model
        self.debug_mode = debug_mode
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.train_metric = train_metric
        self.val_frequency = val_frequency
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.batch_size = batch_size    
        self.num_train_data = len(train_data)
        self.num_val_data = len(val_data)
        self.num_test_data = len(test_data)
        self.training_save_dir = training_save_dir
        self.patience = patience
        self.model_name = model_name
        self.scaler = GradScaler() 

        #DataLoaders
        self.train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_data_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        self.test_data_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self.num_train = len(train_data)
        self.num_val = len(val_data)
        self.num_test = len(test_data)

        #WanDB Logger
        self.wandb_logger = WandBLogger(enabled=True, model=model, run_name=model_name)
        
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        if self.debug_mode:
            print(f"--- Training epoch {epoch_idx} ---")


        self.model.train() 
        self.train_metric.reset()

        epoch_loss = 0

        for inputs, masks in tqdm(self.train_data_loader, desc="Train"):
            
            inputs, masks = inputs.to(self.device), masks.to(self.device).squeeze(1)

            self.optimizer.zero_grad()

            with autocast():  # Mixed precision enabled here
                outputs = self.model(inputs)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
    
                loss = self.loss_fn(outputs, masks)

            # Scaled backprop
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Gather metrics
            epoch_loss += (loss.item() * inputs.size(0))
            preds = torch.argmax(outputs.detach().cpu(), dim=1)
            for p, t in zip(preds, masks.cpu()):
                self.train_metric.update(p, t)

            #if self.debug_mode and i % 500 == 0:  # Print debug info every 10 batches
                #print(f"Batch {i}, Loss: {loss.item()}")
      

        # Update learning rate scheduler
        
        avg_loss = epoch_loss / self.num_train
        self.lr_scheduler.step(avg_loss)

        if self.debug_mode:
            print(f"Epoch {epoch_idx} Training Loss: {epoch_loss}")
            print(f"Training Metrics: {self.train_metric}")

        return (avg_loss, 
                self.train_metric.mean_iou().item(), 
                self.train_metric.overall_accuracy().item(),
                self.train_metric.iou(1).item(), #iou for class 1
                self.train_metric.dice(1).item(), #dice for class 1
                self.train_metric.precision(1).item(), #precision for class 1
                self.train_metric.recall(1).item() #recall for class 1
               )
        

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """

        if self.debug_mode:
            print(f"--- Validating epoch {epoch_idx} ---")

        self.model.eval()
        self.val_metric.reset()
        epoch_loss = 0.
        

        with torch.no_grad():
             for inputs, masks in tqdm(self.val_data_loader, desc="Evaluate"):
            
                inputs, masks = inputs.to(self.device), masks.to(self.device).squeeze(1)

                # Forward pass
                outputs = self.model(inputs)
                if isinstance(outputs, dict):  # handle torchvision segmentation models
                    outputs = outputs["out"]

                # Compute loss
                loss = self.loss_fn(outputs, masks)

                # Update metrics
                epoch_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs.cpu(), dim=1)
                for p, t in zip(preds, masks.cpu()):
                    self.val_metric.update(p, t)
             #if self.debug_mode and i % 500 == 0:  # Print debug info every 10 batches
                 #print(f"Batch {i}, Validation Loss: {loss.item()}")

        avg_loss = epoch_loss / self.num_val

        if self.debug_mode:
            print(f"Epoch {epoch_idx} Validation Loss: {epoch_loss}")
            print(f"Validation Metrics: {self.val_metric}")

        return (
            avg_loss, 
            self.val_metric.mean_iou().item(), 
            self.val_metric.overall_accuracy().item(),
            self.val_metric.iou(1).item(), #iou for class 1
            self.val_metric.dice(1).item(), #dice for class 1
            self.val_metric.precision(1).item(), #precision for class 1
            self.val_metric.recall(1).item() #recall for class 1
        )

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """

        best_miou = -np.inf
        best_loss = np.inf
        early_stopping_counter = 0 

        print(f"Training with batch size: {self.batch_size}")

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}/{self.num_epochs}:")

            train_loss, train_miou, train_acc, train_iou, train_dice, train_precision, train_recall = self._train_epoch(epoch)
           
            wandb_log = {"epoch": epoch, "train/loss": train_loss, "train/miou": train_miou, "train/acc": train_acc,
                         "train/iou_1": train_iou, "train/dice_1": train_dice, "train/precision_1": train_precision,
                         "train/recall_1": train_recall}

            if epoch % self.val_frequency == 0:
                val_loss, val_miou, val_acc, val_iou, val_dice, val_precision, val_recall = self._val_epoch(epoch)
                wandb_log.update({"val/loss": val_loss, "val/miou": val_miou, "val/acc": val_acc,
                                  "val/iou_1": val_iou, "val/dice_1": val_dice, 
                                  "val/precision_1": val_precision, "val/recall_1": val_recall})

                if val_miou >= best_miou and val_loss <= best_loss :
                    print(f"#### Best val_miou {val_miou} at epoch {epoch}")
                    print(f"#### Saving model to {self.training_save_dir}")
                    self.model.save(Path(self.training_save_dir), suffix=self.model_name + "best")
                    best_miou = val_miou
                    best_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    print(f"Early stopping counter: {early_stopping_counter}/{self.patience}")


                if epoch == self.num_epochs-1:
                    self.model.save(Path(self.training_save_dir), suffix=self.model_name + "last")
                
                # Check if early stopping condition is met
                if early_stopping_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

            self.wandb_logger.log(wandb_log)

    def test(self) -> Tuple[float, float, float]:
        """
        Tests the model on a given test dataset.
        Prints the metrics and returns loss, accuracy, and per-class accuracy.

        Args:
            test_data (torch.utils.data.Dataset): Test dataset to evaluate on.
            batch_size (int): Batch size for the test DataLoader.

        Returns:
            Tuple[float, float, float]: Test loss, mean accuracy, and mean per-class accuracy.
        """

        print("Testing the model...")

        self.wandb_logger = WandBLogger(enabled=True, model=self.model, run_name=self.model_name)
        

        self.model.eval()  # Set model to evaluation mode

        print("Model name...", self.model_name)

        all_water_probs = []

        test_loss = 0.0
        self.test_metric.reset()

        #with torch.no_grad():
        for inputs, masks in tqdm(self.test_data_loader, desc="Test"):
            inputs, masks = inputs.to(self.device), masks.to(self.device).squeeze(1)

            
            outputs_dict = self.model(inputs, return_water_prob=False)

            if isinstance(outputs_dict, dict):
                outputs = outputs_dict["out"]


            
            # Compute loss
            loss = self.loss_fn(outputs, masks)
            test_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs.cpu(), dim=1)
            for p, t in zip(preds, masks.cpu()):
                    self.test_metric.update(p, t)

        # Compute average loss
        test_loss /= self.num_test


        if self.debug_mode:
            print(f"Test Metrics: {self.test_metric}")
        # Log metrics to WandB
        self.wandb_logger.log({"test/loss": test_loss, "test/miou": self.test_metric.mean_iou().item(),
                               "test/acc": self.test_metric.overall_accuracy().item(),
                               "test/iou_1": self.test_metric.iou(1).item(), 
                               "test/dice_1": self.test_metric.dice(1).item(), 
                               "test/precision_1": self.test_metric.precision(1).item(), 
                               "test/recall_1": self.test_metric.recall(1).item()})


        return test_loss, self.test_metric


    def dispose(self) -> None:
        """
        Finish logging.
        """
        self.wandb_logger.finish()
