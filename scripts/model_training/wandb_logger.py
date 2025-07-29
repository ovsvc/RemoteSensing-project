import torch
import wandb
from typing import Dict, Any
import os, sys

# Check if the code is running in Colab
IN_COLAB = 'google.colab' in sys.modules

# Get API key for logging
'''
if IN_COLAB:
    # In Colab, read API key from user data
    from google.colab import userdata
    api_key = userdata.get("api_key")
    if api_key is None:
        raise ValueError("API_KEY is not set in Colab user data.")
else:
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('API_KEY')
    if api_key is None:
        raise ValueError("API_KEY is not set in the .env file.")

'''


api_key = os.getenv("api_key")

# Login to wandb
wandb.login(key=api_key)

class WandBLogger:
    def __init__(
        self,
        enabled: bool = True,
        model: torch.nn.Module = None,
        run_name: str = None,
        config: Dict[str, Any] = None,
        project: str = "ds project" 
    ) -> None:
        """
        Initialize WandB Logger.
        
        :param enabled: Whether logging is enabled.
        :param model: The PyTorch model to watch.
        :param run_name: Custom run name for WandB.
        :param config: Configuration dictionary.
        :param project: Name of the WandB project.
        :param entity: Name of the WandB entity (organization or user).
        :param group: Group for organizing runs.
        """
        self.enabled = enabled
        self.config = config

        if self.enabled:
            # Initialize WandB
            wandb.init(
                project=project,
                config=self.config,
            )
            wandb.run.name = run_name or wandb.run.id

            # Watch the model if provided
            if model is not None and isinstance(model, torch.nn.Module):
                self.watch(model)

    def watch(self, model: torch.nn.Module, log_freq: int = 100):
        """
        Start monitoring the model in WandB.
        
        :param model: PyTorch model to monitor.
        :param log_freq: Frequency of logging gradients/weights.
        """
        if isinstance(model, torch.nn.Module):
            wandb.watch(model, log="all", log_freq=log_freq)
        else:
            raise ValueError("Model must be an instance of torch.nn.Module.")

    def log(self, log_dict: Dict[str, Any], commit: bool = True, step: int = None):
        """
        Log metrics to WandB.
        
        :param log_dict: Dictionary of metrics to log.
        :param commit: Whether to commit the log as a step.
        :param step: Optional step number.
        """
        if self.enabled:
            try:
                if step is not None:
                    wandb.log(log_dict, commit=commit, step=step)
                else:
                    wandb.log(log_dict, commit=commit)
            except Exception as e:
                print(f"Failed to log data: {e}")

    def finish(self):
        """Finish the WandB run."""
        if self.enabled:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Failed to finish WandB run: {e}")
