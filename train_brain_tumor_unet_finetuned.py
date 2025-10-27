import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
import segmentation_models_pytorch as smp
from tqdm.notebook import tqdm
from matplotlib.patches import Patch


"""
Custom Dataset
"""
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, split="train", transforms=None):
        self.root_dir = os.path.join(root_dir, split)
        self.coco = COCO(os.path.join(self.root_dir, "_annotations.coco.json"))
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")  # Since imagenet has 3 channels (RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

        for ann in anns:
            cat_id = ann["category_id"]
            rle = self.coco.annToMask(ann)
            mask[rle == 1] = cat_id - 1

        image = F.to_tensor(image)
        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask, img_info['file_name']


"""
Metrics
    - dice_score: Dice Similarity Coefficient (DSC)
    - iou_score: Intersection over Union
    - accuracy_score: accuracy
    note: loss calculated in loop
"""
def dice_score(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth)
    return dice.mean().item()


def iou_score(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1).float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2))
    union = (pred + target).sum(dim=(1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def accuracy_score(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float()
    return correct.mean().item()


"""
One epoch iteration
"""
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=True):
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = dice_score(outputs.detach(), masks)
            iou = iou_score(outputs.detach(), masks)
            acc = accuracy_score(outputs.detach(), masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        running_acc += acc

    n = len(dataloader)
    return running_loss/n, running_dice/n, running_iou/n, running_acc/n


"""
One validate iteration
"""
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    val_acc = 0.0

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        for images, masks in tqdm(dataloader, desc="Validating", leave=False):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = accuracy_score(outputs, masks)

            val_loss += loss.item()
            val_dice += dice
            val_iou += iou
            val_acc += acc

    n = len(dataloader)
    return val_loss/n, val_dice/n, val_iou/n, val_acc/n


"""
Function for testing model on test-set and save results
"""
def test_model(model, dataloader, device, save_results=True, save_dir="test_results"):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_dice, test_iou, test_acc = 0.0, 0.0, 0.0, 0.0

    if save_results:
        os.makedirs(save_dir, exist_ok=True)

    # Disable gradient computation for testing
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        for batch_idx, (images, masks, file_names) in enumerate(tqdm(dataloader, desc="Testing", leave=False)):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)

            # Upsample predictions to match target size
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, 
                                        size=masks.shape[-2:],
                                        mode="bilinear",
                                        align_corners=False)

            # Compute metrics 
            loss = criterion(outputs, masks)
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = accuracy_score(outputs, masks)

            test_loss += loss.item()
            test_dice += dice
            test_iou += iou
            test_acc += acc

            # Save example images/predictions (first few batches)
            if save_results and batch_idx < 5:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                imgs = images.cpu().permute(0, 2, 3, 1).numpy()
                masks_gt = masks.cpu().numpy()

                for i in range(len(preds)):
                    img = imgs[i]
                    gt = masks_gt[i]
                    pred = preds[i]
                    file_name = os.path.splitext(file_names[i])[0]

                    # Normalize for image
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(img, cmap="gray")

                    # Overlay ground truth (green)
                    gt_overlay = (gt>0).astype(float)
                    ax.imshow(np.ma.masked_where(gt_overlay == 0, gt_overlay), cmap="Greens", alpha=0.4)

                    # Overlay prediction (red)
                    pred_overlay = (pred>0).astype(float)
                    ax.imshow(np.ma.masked_where(pred_overlay == 0, pred_overlay), cmap="Reds", alpha=0.4)

                    ax.set_title(f"Ground Truth (green) vs Predicted (red)")
                    ax.axis("off")

                    legend_elements = [
                        Patch(facecolor='green', edgecolor='none', alpha=0.4, label='Ground Truth'),
                        Patch(facecolor='red', edgecolor='none', alpha=0.4, label='Prediction')
                    ]
                    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"{file_name}_overlay.png"), dpi=150)
                    plt.close(fig)

    # Average metrics
    n = len(dataloader)
    test_loss /= n
    test_dice /= n
    test_iou  /= n
    test_acc  /= n

    print("\n=== Test Results ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Dice Score: {test_dice:.4f}")
    print(f"IoU Score: {test_iou:.4f}")
    print(f"Accuracy: {test_acc:.4f}")

    return test_loss, test_dice, test_iou, test_acc


def main():
    dataset_root = "./dataset"
    batch_size = 8
    num_classes = 2
    fine_tune_epochs = 10
    num_workers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = BrainTumorDataset(root_dir=dataset_root, split="train")
    val_dataset   = BrainTumorDataset(root_dir=dataset_root, split="valid")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Model setup
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(device)

    # Unfreeze last encoder layers for fine-tuning
    for name, param in model.encoder.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    # Keep batchnorm layers trainable
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # History
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
        "train_acc": [], "val_acc": []
    }

    print("\n Fine-tuning encoder")
    for epoch in range(fine_tune_epochs):
        print(f"\nEpoch [{epoch+1}/{fine_tune_epochs}]")
        t_loss, t_dice, t_iou, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        v_loss, v_dice, v_iou, v_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {t_loss:.4f}, Dice: {t_dice:.4f}, IoU: {t_iou:.4f}, Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}, Dice: {v_dice:.4f}, IoU: {v_iou:.4f}, Acc: {v_acc:.4f}")

        # Save metrics
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_dice"].append(t_dice)
        history["val_dice"].append(v_dice)
        history["train_iou"].append(t_iou)
        history["val_iou"].append(v_iou)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

    # Save fine-tuned model
    torch.save(model.state_dict(), "unet_finetuned.pth")
    print("\n Fine-tuning complete. Model saved as 'unet_finetuned.pth'")

    # Test
    test_dataset = BrainTumorDataset(root_dir=dataset_root, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print("\n Evaluating on Test Set")
    model.load_state_dict(torch.load("unet_finetuned.pth", map_location=device))
    test_model(model, test_loader, device)

    epochs = range(1, fine_tune_epochs + 1)

    # ----------------- PLOTS -----------------
    metrics = [
        ("Loss", "train_loss", "val_loss", "loss.png"),
        ("Dice Score", "train_dice", "val_dice", "dice.png"),
        ("IoU Score", "train_iou", "val_iou", "iou.png"),
        ("Accuracy", "train_acc", "val_acc", "accuracy.png")
    ]

    for title, train_key, val_key, fname in metrics:
        plt.figure()
        plt.plot(epochs, history[train_key], label=f"Train {title}")
        plt.plot(epochs, history[val_key], label=f"Val {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"{title} over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot: {fname}")

if __name__ == "__main__":
    main()
