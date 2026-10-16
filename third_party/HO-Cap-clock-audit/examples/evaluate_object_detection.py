import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from hocap_toolkit.utils import *


def evaluate_object_detection_results(gt_file, pred_file):
    coco_gt = COCO(str(gt_file))
    coco_dt = coco_gt.loadRes(str(pred_file))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap_metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APs": coco_eval.stats[3],
        "APm": coco_eval.stats[4],
        "APl": coco_eval.stats[5],
        "AR1": coco_eval.stats[6],
        "AR10": coco_eval.stats[7],
        "AR100": coco_eval.stats[8],
        "ARs": coco_eval.stats[9],
        "ARm": coco_eval.stats[10],
        "ARl": coco_eval.stats[11],
    }
    print(
        f"AP: {ap_metrics['AP']:.3f} | AP_50: {ap_metrics['AP50']:.3f} | AP_75: {ap_metrics['AP75']:.3f} | AP_s: {ap_metrics['APs']:.3f} | AP_m: {ap_metrics['APm']:.3f} | AP_l: {ap_metrics['APl']:.3f}"
    )

    # Save to csv
    df = pd.DataFrame([ap_metrics])
    save_csv_file = Path(pred_file).parent / f"{Path(pred_file).stem}_ap.csv"
    df.to_csv(save_csv_file, index=False)

    # Save to txt
    str_metrics = [
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {coco_eval.stats[0]*100:.2f}",
        f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {coco_eval.stats[1]*100:.2f}",
        f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {coco_eval.stats[2]*100:.2f}",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {coco_eval.stats[3]*100:.2f}",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {coco_eval.stats[4]*100:.2f}",
        f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {coco_eval.stats[5]*100:.2f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {coco_eval.stats[6]*100:.2f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {coco_eval.stats[7]*100:.2f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {coco_eval.stats[8]*100:.2f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {coco_eval.stats[9]*100:.2f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {coco_eval.stats[10]*100:.2f}",
        f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {coco_eval.stats[11]*100:.2f}",
    ]
    str_metrics = "\n".join(str_metrics)
    save_txt_file = Path(pred_file).parent / f"{Path(pred_file).stem}_ap.txt"
    save_txt_file.write_text(str_metrics)
    print(f"AP metrics saved to '{save_csv_file}' and '{save_txt_file}'")


if __name__ == "__main__":
    gt_file = "config/benchmarks/odet_gt.json"
    pred_file = "results/odet_demo.json"

    tqdm.write("- Evaluating Object Detection results...")
    evaluate_object_detection_results(gt_file, pred_file)
    tqdm.write("- Evaluation Done...")
