import config
import torch
import torch.optim as optim
from Yolov3 import YOLOv3
from tqdm import tqdm
from utils import(
    mean_average_precision,
    non_max_suppression,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    plot_couple_examples,
    get_evaluation_bboxes,
    cells_to_bboxes,
)
from loss import YoloLoss

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaler_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaler_anchors[0])
                + loss_fn(out[1], y1, scaler_anchors[1])
                + loss_fn(out[2], y2, scaler_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv",
        test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.Anchors)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)



    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        model.train()
        train_fn(train_loader, model, optimizer, YoloLoss(), scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)

        model.eval()
        pred_boxes, target_boxes = get_evaluation_bboxes(
            test_loader, model, iou_threshold=config.MAP_IOU_THRESH, anchors=scaled_anchors
        )
        mapval = mean_average_precision(
            pred_boxes,
            target_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"MAP: {mapval.item()}")
        model.train()

if __name__ == "__main__":
    main()