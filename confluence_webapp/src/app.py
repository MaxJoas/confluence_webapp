import argparse
import logging
import os

import torch
from utils.streamlit_utils import StreamlitInterface


def main():
    """Main entry point of the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the Streamlit interface.")
    parser.add_argument(
        "unet_path",
        type=str,
        help="Path to the UNet model",
        default=os.path.join("models", "lc_models", "unet_model_final.pth"),
    )
    parser.add_argument(
        "detectron_path",
        type=str,
        help="Path to the Detectron2 model configuration",
        default=os.path.join("models", "lc_models", "d2_model_final.pth"),
    )
    parser.add_argument(
        "sam_path",
        type=str,
        help="Path to the SAM model",
        default=os.path.join("models", "lc_models", "sam_model_final.pth"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interface = StreamlitInterface(
        unet_path=args.unet_path,
        detectron_path=args.detectron_path,
        sam_path=args.sam_path,
        device=device,
    )
    interface.run()


if __name__ == "__main__":
    main()
