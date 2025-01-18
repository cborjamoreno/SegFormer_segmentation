from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from transformers import AdamW
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import os
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, TFSegformerForSemanticSegmentation
import pandas as pd
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt


WIDTH = 512
HEIGHT = 512

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, transforms=None, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train
        self.transforms = transforms

        sub_path = "train" if self.train else "test"
        self.img_dir = os.path.join(self.root_dir, sub_path, "images")
        self.ann_dir = os.path.join(self.root_dir, sub_path, "labels")

        print(self.img_dir)
        print(self.ann_dir)
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)
        
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), f"There must be as many images as there are segmentation maps. Number of images: {len(self.images)}, Number of annotations: {len(self.annotations)}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Construct paths to the image and annotation
        image_path = os.path.join(self.img_dir, self.images[idx])
        annotation_path = os.path.join(self.ann_dir, self.annotations[idx])

        # Read the image and annotation using OpenCV
        image = cv2.imread(image_path)
        # Check if the image was loaded successfully
        if image is None:
            print(f"Failed to load image from {image_path}")
        else:
            try:
                # Attempt to convert the image color space from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                print(f"Failed to convert image color space: {e}")

        segmentation_map = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert the OpenCV image to a PIL Image for the transformation
        image = Image.fromarray(image)
        segmentation_map = Image.fromarray(segmentation_map)

        # Resize the image and segmentation map to a fixed size, if needed
        # image = TF.resize(image, (HEIGHT, WIDTH))
        # segmentation_map = TF.resize(segmentation_map, (HEIGHT, WIDTH), interpolation=TF.InterpolationMode.NEAREST)

        # Apply the transformations if any
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            # Convert the PIL Image to a tensor (this also permutes dimensions to C x H x W)
            image = TF.to_tensor(image)

        # Convert the PIL Image back to a NumPy array if your processing pipeline requires it
        segmentation_map = np.array(segmentation_map)

        # Convert the segmentation map to a tensor
        segmentation_map = torch.tensor(segmentation_map, dtype=torch.long)


        # Prepare the return dictionary
        return_dict = {'pixel_values': image, 'labels': segmentation_map}

        # Include the filename in the return dictionary if not in training mode
        # Inside your method, after checking if not in training mode
        if not self.train:
            with Image.open(image_path) as img:
                width, height = img.size
            return_dict['filename'] = self.images[idx]
            return_dict['dim'] = (width, height)

        return return_dict


def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer Segmentation")
    parser.add_argument('--dataset_path', type=str, default='MosaicsUCSD_sam_100', help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default='test_predictions_mixed_100', help='Path to save the output predictions')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path
    output_path = args.output_path
    WIDTH = 512
    HEIGHT = 512

    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomVerticalFlip(0.05),
        # transforms.RandomRotation(20),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
    ])

    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
    train_dataset = ImageSegmentationDataset(root_dir=dataset_path, feature_extractor=feature_extractor, transforms=transform)
    test_dataset = ImageSegmentationDataset(root_dir=dataset_path, feature_extractor=feature_extractor, transforms=None, train=False)


    from torch.utils.data import DataLoader, random_split

    total_size = len(train_dataset)
    train_size = int(total_size * 0.8)  # 80% for training
    validation_size = total_size - train_size  # 20% for validation

    # Splitting the dataset
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

    # Creating DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4)

    id2label = {**{i: str(i) for i in range(34)}, 34: "background"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)

    model_checkpoint = "nvidia/mit-b5"
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
        reshape_last_stage=True,
    )

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_val_loss = float("inf")  # Initialize a high value

    # Training
    num_epochs = 15
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")

        running_loss, running_acc = 0.0, 0.0
        model.train()

        for idx, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            upsampled_logits = nn.functional.interpolate(
                outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)

            # Accuracy calculation (excluding background)
            mask = labels != 34
            if mask.any():
                pred_labels = predicted[mask].detach().cpu().numpy()
                true_labels = labels[mask].detach().cpu().numpy()
                accuracy = accuracy_score(pred_labels, true_labels)
            else:
                accuracy = 0

            loss = outputs.loss
            # Optionally mask background pixels
            labels_masked = labels.clone()
            labels_masked[labels_masked == 34] = -100  # Set ignored pixels to -100
            loss = criterion(upsampled_logits, labels_masked)
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss = (running_loss * idx + loss.item()) / (idx + 1)
            running_acc = (running_acc * idx + accuracy) / (idx + 1)
            pbar.set_postfix({"Loss": running_loss, "Pixel-wise Acc": running_acc})

        # Validation
        model.eval()
        val_losses, val_accuracies = [], []
        with torch.no_grad():
            for batch in valid_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                upsampled_logits = nn.functional.interpolate(
                    outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                predicted = upsampled_logits.argmax(dim=1)

                mask = labels != 34
                if mask.any():
                    pred_labels = predicted[mask].detach().cpu().numpy()
                    true_labels = labels[mask].detach().cpu().numpy()
                    accuracy = accuracy_score(pred_labels, true_labels)
                else:
                    accuracy = 0

                val_losses.append(outputs.loss.item())
                val_accuracies.append(accuracy)

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = sum(val_accuracies) / len(val_accuracies)

        print(
            f"Train Loss: {running_loss:.4f}, Train Acc: {running_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
            
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), dataset_path + ".pth")
            print(f"Model saved with validation loss: {best_val_loss:.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_train_losses, label='Training Loss')
    plt.plot(epoch_val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(dataset_path+"_losses.png")

    # Testing

    model.eval()

    def rescale_image(image, new_shape):
        return cv2.resize(image, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)

    def get_predictions(predictions_numpy, batch):
        # Get the image sizes
        original_shape = batch['pixel_values'].shape[-2:]
        
        # Rescale the predictions to the original image size
        rescaled_predictions = [rescale_image(prediction, original_shape) for prediction in predictions_numpy]
        
        # Convert list of rescaled predictions to a NumPy array
        rescaled_predictions = np.array(rescaled_predictions)
        
        # Get the original images and labels
        images = batch['pixel_values'].cpu().numpy()
        images = images.transpose(0, 2, 3, 1)
        labels = batch['labels'].cpu().numpy()

        return images, labels, rescaled_predictions

    # Plot the images, labels, and predictions
    def save_predictions(images, labels, predictions, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
            image_path = os.path.join(output_dir, f"image_{i}.png")
            label_path = os.path.join(output_dir, f"label_{i}.png")
            prediction_path = os.path.join(output_dir, f"prediction_{i}.png")
            
            # Convert tensors to numpy arrays and save them
            image = image.permute(1, 2, 0).cpu().numpy()  # Convert from CxHxW to HxWxC
            label = label.cpu().numpy()
            prediction = prediction.cpu().numpy()
            
            plt.imsave(image_path, image)
            plt.imsave(label_path, label, cmap='gray')
            plt.imsave(prediction_path, prediction, cmap='gray')

    def label_to_rgb(label, palette):
        # Create an empty image with 3 channels for RGB
        rgb_image = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        
        # Map each label to its corresponding color
        for i, color in enumerate(palette):
            mask = (label == i)
            rgb_image[mask] = color
        
        return rgb_image

    def convert_predictions_and_labels_to_rgb(labels, predictions, palette):
        labels_rgb = np.array([label_to_rgb(label, palette) for label in labels])
        predictions_rgb = np.array([label_to_rgb(prediction, palette) for prediction in predictions])
        return labels_rgb, predictions_rgb

    all_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch['pixel_values'].to(device)
            filenames = batch['filename']
            dims = batch['dim']

            widths, heights = dims
            widths_list = widths.tolist()
            heights_list = heights.tolist()
            dims_tuples = list(zip(widths_list, heights_list))
            
            # Get model predictions
            outputs = model(inputs)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)
            
            # Move predictions to CPU and convert to numpy for further processing if necessary
            predictions_numpy = predicted_labels.cpu().numpy()

            # Process and plot predictions for the current batch
            images, labels, rescaled_predictions = get_predictions(predictions_numpy, batch)

            for filename, prediction_rgb, dim in zip(filenames, rescaled_predictions, dims_tuples):
                all_predictions.append((filename, prediction_rgb, dim))

    # Reshape all predicitons to the original image size
    os.makedirs(output_path, exist_ok=True)

    for filename, prediction, dim in all_predictions:  # Adjusted to unpack filenames with predictions
        # Reshape prediction to the original image size
        # prediction = cv2.resize(prediction, dim, interpolation=cv2.INTER_NEAREST)
        output_dir = os.path.join(output_path, os.path.splitext(filename)[0] + '.png')  # Use original filename
        cv2.imwrite(output_dir, prediction)