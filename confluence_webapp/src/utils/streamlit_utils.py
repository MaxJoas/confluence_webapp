import sys
from io import StringIO
from typing import Any, List

import logging
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from models.sam import SAMModel
from models.unet_model import UNet
from services.prediction_service import PredictionManager
from utils.sam_utils import load_datasets
from utils.detectron_utils import setup_cfg
from utils.utils import visualize_contours

# from utils.cellpose_utils import get_total_cellpose_mask
from cellpose import models as cellpose_models


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StreamlitInterface:
    """Manages Streamlit UI and interaction."""

    def __init__(
        self, unet_path: str, detectron_path: str, sam_path: str, device: torch.device
    ):
        """
        Args:
            unet_path (str): Path to UNet weights
            detectron_path (str): Path to Detectron2 weights
            sam_path (str): Path to SAM weights
            device (torch.device): Computation device
        """
        self.unet_path = unet_path
        self.detectron_path = detectron_path
        self.sam_path = sam_path
        self.device = device
        self.prediction_manager = PredictionManager(device)

    def mask_to_image(self, mask: torch.Tensor) -> Image.Image:
        """
        Convert a tensor mask to a PIL image.

        Args:
            mask (torch.Tensor): Mask tensor
        Returns:
            Image.Image: Mask image

        """

        mask_np = mask.cpu().detach().numpy().squeeze()
        # only multiply by 255 if the mask is not binary
        logger.info(f"range of mask in mask_to_img: {mask_np.min()} - {mask_np.max()}")
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        return mask_img

    def process_unet_mask(self, mask: np.ndarray) -> Image.Image:
        """
        Process the UNet mask.
        Args:
            mask (np.ndarray): Mask array
        Returns:
            Image.Image: Processed mask image

        """
        if mask.ndim == 2:
            return Image.fromarray((mask * 255).astype(np.uint8))
        elif mask.ndim == 3:
            return Image.fromarray(
                (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
            )

    def process_unet(self, files: List[Any]) -> str:
        """Process files using UNet model."""
        cols = st.columns(3)
        confluence_dict = {"unet": []}
        file_list = []

        try:
            net = UNet(n_channels=1, n_classes=2)
            net.to(device=self.device)
            net.load_state_dict(torch.load(self.unet_path, map_location=self.device))
            net.eval()

            for f in files:
                if f is not None:
                    img = Image.open(f).convert("L")
                    numpy_img = np.array(img)
                    cols[0].markdown("##### Preview \n Original file ")
                    cols[0].image(img)
                    file_list.append(f)

                    res = cols[1].markdown("##### Please wait...")

                    # Get prediction
                    mask, probs = self.prediction_manager.predict_unet(
                        net=net, image=img, scale_factor=0.35, threshold=0.1
                    )

                    # Process results
                    result = self.process_unet_mask(mask)
                    logger.info(f"UNet probs: {probs}")
                    result = Image.fromarray(255 - np.array(result) * 2)
                    pred_mask = np.array(result) // 255

                    # Calculate confluence
                    confluence = self.prediction_manager.calc_confluence(pred_mask)
                    confluence_str = f"{round(confluence * 100, 2):.2f}%"
                    confluence_dict["unet"].append(confluence)
                    cells_pred = visualize_contours(img=numpy_img, pred_mask=pred_mask)

                    cols[1].image(cells_pred)
                    res.markdown(f"##### Results UNet \n Confluence: {confluence_str}")

                    cols[2].markdown("##### Prediction Probabilities \n as probmap")
                    cols[2].image(self.process_unet_mask(probs.cpu().detach().numpy()))

        except Exception as e:
            st.error(f"Error processing UNet predictions: {str(e)}")
            return ""

        return pd.DataFrame(index=file_list, data=confluence_dict).to_csv()

    def process_detectron(self, files: List[Any]) -> str:
        """
        Process files using Detectron2 model.

        Args:
            files (List[Any]): List of uploaded files
        Returns:
            str: CSV data containing confluence values

        """
        cols = st.columns(2)
        confluence_dict = {"detectron2": []}
        file_list = []

        try:
            cfg = setup_cfg()

            for image_file in files:
                if image_file is not None:
                    img = Image.open(image_file).convert("L")
                    cols[0].markdown("##### Preview \n Original file ")
                    cols[0].image(img)
                    res = cols[1].markdown("#### Please wait...")
                    file_list.append(image_file)

                    # Get prediction
                    img_out, confluence = self.prediction_manager.predict_detectron(
                        cfg=cfg, image_path=image_file, weights_path=self.detectron_path
                    )

                    confluence_dict["detectron2"].append(confluence)
                    cols[1].image(img_out)

                    # Display results
                    conf_str = (
                        f"{round(confluence * 100, 2):.2f}%"
                        if confluence != -9999
                        else "Image dimension not supported"
                    )
                    res.markdown(f"##### Results Detectron2 \n Confluence: {conf_str}")

        except Exception as e:
            st.error(f"Error processing Detectron2 predictions: {str(e)}")
            return ""

        return pd.DataFrame(index=file_list, data=confluence_dict).to_csv()

    def process_sam(self, files: List[Any]) -> str:
        """
        Process files using SAM model

        Args:
            files (List[Any]): List of uploaded files
        Returns:
            str: CSV data containing confluence values

        """
        cols = st.columns(3)
        confluence_dict = {"sam": []}
        file_list = []

        try:
            sample_img = Image.open(files[0]).convert("L")
            original_size = sample_img.size
            inference_loader = load_datasets(file_list=files)

            for (_, img), filepath in zip(enumerate(inference_loader), files):
                display_img = Image.open(filepath).convert("L")
                numpy_img = np.array(display_img)
                img = img.to(self.device)

                cols[0].markdown("##### Preview \n Original file ")
                cols[0].image(display_img)
                res = cols[1].markdown("##### Please wait...")
                file_list.append(filepath)
                mask_resized, confluence, pred = self.prediction_manager.predict_sam(
                    image=img, sam_path=self.sam_path, original_size=original_size
                )

                confluence_str = f"{round(confluence * 100, 2):.2f}%"
                confluence_dict["sam"].append(confluence_str)
                cells_pred = visualize_contours(img=numpy_img, pred_mask=mask_resized)
                cols[1].image(cells_pred)
                res.markdown(f"##### Results SAM \n Confluence: {confluence_str}")

                cols[2].markdown("##### Prediction Probabilities \n as probmap")
                cols[2].image(self.mask_to_image(pred))

        except Exception as e:
            st.error(f"Error processing SAM predictions: {str(e)}")
            return ""

        return pd.DataFrame(index=file_list, data=confluence_dict).to_csv()

    def process_cellpose(
        self, files: List[Any], cellprob_threshold: float, flow_threshold: float
    ) -> str:
        """
        Uses Cellpose to process files and predict confluence.

        Args:
            files (List[Any]): List of uploaded files
        Returns:
            str: CSV data containing confluence values

        """
        # make suitable to plot with matplotlib by reordering channels
        use_gpu = True if self.device.type == "cuda" else False
        model = cellpose_models.CellposeModel(gpu=use_gpu, model_type="cyto")
        confl_dict = {"cellpose": []}
        cols = st.columns(3)
        for f in files:
            img = Image.open(f).convert("RGB")
            img = np.array(img)
            display_img = Image.open(f).convert("L")
            cols[0].markdown("##### Preview \n Original file ")
            cols[0].image(display_img)
            res = cols[1].markdown("#### Please wait...")

            imgs = [img]
            logger.info(f"Cellpose images type: {type(imgs[0])}")
            logger.info(f"Cellpose images shape: {imgs[0].shape}")
            masks, flows, *_ = model.eval(
                imgs,
                channels=[0, 0],
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )
            probs = flows[0][2]
            # make display image
            probs_sig = torch.sigmoid(torch.from_numpy(probs))

            prediction_mask = masks[0]
            prediction_mask = prediction_mask.astype(np.uint8)
            # make binary mask
            prediction_mask = (prediction_mask > 0).astype(np.uint8)
            logger.info(
                f"range of mask in cellpose: {prediction_mask.min()} - {prediction_mask.max()}"
            )
 
            postive_class = (prediction_mask == 1).sum()
            negative_class = (prediction_mask == 0).sum()
            logger.info(f"Positive class: {postive_class}")
            logger.info(f"Negative class: {negative_class}")
            cells_pred = visualize_contours(img=img, pred_mask=prediction_mask)
            confluence = self.prediction_manager.calc_confluence(prediction_mask, mask_class=1)
            confluence_str = f"{round(confluence * 100, 2):.2f}%"
            confl_dict["cellpose"].append(confluence_str)
            cols[1].image(cells_pred)
            res.markdown(f"##### Results Cellpose \n Confluence: {confluence_str}")

            cols[2].markdown("##### Prediction Probabilities \n as probmap")
            cols[2].image(self.mask_to_image(probs_sig))

        pass

    def run(self):
        """Main interface for running the Streamlit application."""
        st.image(
            "https://scads.ai/wp-content/themes/scads2023/assets/images/logo.png",
            width=200,
            use_container_width=False,
        )
        st.header("SaxoCell Confluence Detection")

        # Dropdown for selecting model
        model_choice = st.selectbox(
            "Choose model for cell detection", ("SAM", "UNet", "Detectron2", "Cellpose")
        )
        # if the Cellpose is selected show a swithcer for adjusting the flow threshold and cellprob threshold
        if model_choice == "Cellpose":
            flow_threshold = st.slider("Flow Threshold", 0.0, 1.0, 0.4)
            cellprob_threshold = st.slider("Cell Probability Threshold", -6.0, 6.0, 0.0)

        files = st.file_uploader(label="Upload Cell Image", accept_multiple_files=True)

        if files:
            if st.button("Predict with Selected Model"):
                if model_choice == "UNet":
                    st.write("Processing with UNet...")
                    csv_data = self.process_unet(files)
                elif model_choice == "Detectron2":
                    st.write("Processing with Detectron2...")
                    csv_data = self.process_detectron(files)
                elif model_choice == "SAM":
                    st.write("Processing with SAM...")
                    csv_data = self.process_sam(files)
                elif model_choice == "Cellpose":
                    st.write("Processing with Cellpose...")
                    csv_data = self.process_cellpose(
                        files=files,
                        cellprob_threshold=cellprob_threshold,
                        flow_threshold=flow_threshold,
                    )
                else:
                    st.error(f"Model {model_choice} not supported.")

                # Downloadable CSV
                if csv_data:
                    st.download_button(
                        "Download Results",
                        data=csv_data,
                        file_name="confluence_results.csv",
                    )
        else:
            st.info("Please upload images to start processing.")


class StreamlitOutputCapture:
    """Context manager for capturing and redirecting stdout to Streamlit."""

    def __init__(self, output_func: callable):
        """
        Args:
            output_func (callable): Function to handle captured output
        """
        self.output_func = output_func

    def __enter__(self):
        self.string_io = StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.string_io
        return self

    def __exit__(self, *args):
        sys.stdout = self.old_stdout
