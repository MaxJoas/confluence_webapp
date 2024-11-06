import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from PIL import Image, ImageDraw


def setup_cfg(iter=10):
    """
    Set up the Detectron2 configuration
    Args:
        iter (int): Number of iterations
    Returns:
        cfg (CfgNode): Detectron2 configuration
    """
    
    cfg = get_cfg()
    config_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
    cfg.SOLVER.IMS_PER_BATCH = 1
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.WARMUP_ITERS = 5
    cfg.SOLVER.MAX_ITER = iter
    cfg.SOLVER.STEPS = []
    # Small value=Frequent save  need a lot of storage.
    cfg.SOLVER.CHECKPOINT_PERIOD = 100000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    return cfg


def visualize_detectron_results(data_dict, anns):
    """
    Visualize the results of the Detectron2 model.

    Args:
        data_dict (dict): Dictionary containing image data
        anns (list): List of annotations
    Returns:
        img_out (Image): Output image
    """

    img_out = Image.new(
        "L",
        [
            data_dict["width"],
            data_dict["height"],
        ],
    )
    print(f"img_out size: {img_out.size}")
    print(f' with form data: {data_dict["width"]} and {data_dict["height"]} ')

    file_name = data_dict["file_name"]
    org_image = Image.open(file_name)

    img_out1 = ImageDraw.Draw(img_out)
    overlay_out = ImageDraw.Draw(org_image, "RGBA")
    for i in range(len(anns)):
        if len(anns[i]) == 0:
            continue
        try:
            segs = anns[i][0]
            img_out1.polygon(segs, fill="white", outline="white")
            overlay_out.polygon(segs, fill=(21, 239, 116, 27), outline="#B7FF00")
        except Exception as e:
            print(f"Error: {e}")
            raise e
    return org_image
