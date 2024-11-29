import argparse
import logging
import os
import sys
import torch
from utils.streamlit_utils import StreamlitInterface


def main():
    """Main entry point of the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create an ArgumentParser instance
    # parser = argparse.ArgumentParser(description="Run the Streamlit interface.")

    # # Add your custom arguments
    # parser.add_argument(
    #     "unet_path",
    #     type=str,
    #     help="Path to the UNet model",
    #     default=os.path.join("models", "lc_models", "unet_model.pth"),
    # )
    # parser.add_argument(
    #     "--detectron_path",
    #     type=str,
    #     help="Path to the Detectron2 model configuration",
    #     default=os.path.join("models", "lc_models", "d2_model.pth"),
    # )
    # parser.add_argument(
    #     "sam_path",
    #     type=str,
    #     help="Path to the SAM model",
    #     default=os.path.join("models", "sc_models", "sam_model.pth"),
    # )
    # parser.add_argument(
    #     "cellpose_path",
    #     type=str,
    #     help="Path to the Cellpose model",
    #     default=os.path.join("models", "sc_models", "cellpose_model.pth"),
    # )

    # Parse the custom arguments

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the model paths
    unet_path = os.path.join("models", "final", "unet_model.pth")
    detectron_path = os.path.join("models", "final", "d2_model.pth")
    sam_path = os.path.join("models", "final", "sam_model.pth")
    cellpose_path = os.path.join("models", "final", "cellpose_model.pth")

    # Initialize the Streamlit interface with custom model paths
    interface = StreamlitInterface(
        unet_path=unet_path,
        detectron_path=detectron_path,
        sam_path=sam_path,
        cellpose_path=cellpose_path,
        device=device,
    )
    interface.run()


if __name__ == "__main__":
    main()
