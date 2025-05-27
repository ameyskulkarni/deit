import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
from timm.models.vision_transformer import VisionTransformer
import copy
import random
from typing import Tuple


class DenseProjectionHead(nn.Module):
    """Projection head for dense contrastive learning"""

    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)


class DeiTWithDenseCL(nn.Module):
    """DeiT model with added dense contrastive learning"""

    def __init__(
            self,
            model_name='deit_small_patch16_224',
            num_classes=1000,
            dense_dim=128,
            queue_size=65536,
            momentum=0.999,
            temperature=0.2,
            dense_weight=0.5
    ):
        super().__init__()

        # Load base DeiT model
        self.base_model = timm.create_model(model_name, pretrained=False)
        print(self.base_model)
        print(f"The embedding dim of base model is {self.base_model.embed_dim}")
        embed_dim = self.base_model.embed_dim

        # Create query and key encoders
        self.encoder_q = self.base_model
        self.encoder_k = copy.deepcopy(self.encoder_q)

        # Create projection heads
        self.dense_proj_q = DenseProjectionHead(embed_dim, out_dim=dense_dim)
        self.dense_proj_k = DenseProjectionHead(embed_dim, out_dim=dense_dim)

        # Create classification head (reuse the one from base model)
        self.cls_head = self.encoder_q.head

        # Disable grad for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.dense_proj_k.parameters():
            param.requires_grad = False

        # Initialize queue for negative samples
        self.register_buffer("queue", torch.randn(dense_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Hyperparameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.dense_weight = dense_weight

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update key encoder using momentum"""
        # Update base encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

        # Update projection head
        for param_q, param_k in zip(self.dense_proj_q.parameters(), self.dense_proj_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue of negative samples"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace the keys at ptr
        self.queue[:, ptr:ptr + batch_size] = keys.T

        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def _get_patch_tokens(self, x, cls_token=False):
        """Extract patch tokens from vision transformer output"""
        if isinstance(x, torch.Tensor):
            # If x is already the output tensor
            if cls_token:
                return x
            else:
                return x[:, 1:, :]  # Remove cls token
        elif hasattr(x, 'tokens'):
            # Some implementations might store tokens in this attribute
            tokens = x.tokens
            if cls_token:
                return tokens
            else:
                return tokens[:, 1:, :]
        else:
            raise ValueError("Cannot extract patch tokens from the model output")

    def forward_encoder_q(self, x):
        """Forward pass through query encoder"""
        # Forward through the transformer
        print(f"Shape of x before forward_features: {x.shape}")
        x = self.encoder_q.forward_features(x)
        print(f"Shape of x after forward_features: {x.shape}")

        # Get patch tokens (excluding cls token)
        patch_tokens = self._get_patch_tokens(x, cls_token=False)

        # Get cls token for classification
        cls_token = self._get_patch_tokens(x, cls_token=True)[:, 0]

        # Project patch tokens for dense contrastive learning
        dense_q = self.dense_proj_q(patch_tokens)

        return cls_token, patch_tokens, dense_q

    @torch.no_grad()
    def forward_encoder_k(self, x):
        """Forward pass through key encoder (no grad)"""
        # Forward through the transformer
        x = self.encoder_k.forward_features(x)

        # Get patch tokens (excluding cls token)
        patch_tokens = self._get_patch_tokens(x, cls_token=False)

        # Project patch tokens for dense contrastive learning
        dense_k = self.dense_proj_k(patch_tokens)

        return patch_tokens, dense_k

    def dense_contrastive_loss(self, q_feat, k_feat):
        """Compute dense contrastive loss"""
        batch_size = q_feat.shape[0]
        patch_size = q_feat.shape[1]

        # Reshape to (batch_size * patch_size, dim)
        q_feat = q_feat.reshape(batch_size * patch_size, -1)
        k_feat = k_feat.reshape(batch_size * patch_size, -1)

        # Compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_feat, k_feat]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_feat, self.queue.clone().detach()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.temperature

        # Labels: positives are the first elements
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q_feat.device)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, im_q, im_k=None, is_train=True):
        """
        Forward pass with both classification and dense contrastive loss

        Args:
            im_q: query image
            im_k: key image (optional, for contrastive loss)
            is_train: whether in training mode

        Returns:
            dict containing losses and/or predictions
        """
        # Forward query through encoder
        cls_token, _, dense_q = self.forward_encoder_q(im_q)

        # Classification prediction
        logits = self.cls_head(cls_token)

        result = {"logits": logits}

        # If training with contrastive loss
        if is_train and im_k is not None:
            # Update key encoder
            self._momentum_update_key_encoder()

            # Forward key through encoder
            _, dense_k = self.forward_encoder_k(im_k)

            # Compute dense contrastive loss
            densecl_loss = self.dense_contrastive_loss(dense_q, dense_k)
            result["densecl_loss"] = densecl_loss

            # Update queue
            self._dequeue_and_enqueue(dense_k.reshape(-1, dense_k.shape[-1]))

        return result


class DeiTDenseCLTrainer:
    """Trainer for DeiT with DenseCL"""

    def __init__(
            self,
            model,
            optimizer,
            train_loader,
            val_loader=None,
            device='cuda',
            cls_weight=1.0,
            dense_weight=0.5,
            num_epochs=100,
            lr_scheduler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cls_weight = cls_weight
        self.dense_weight = dense_weight
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        cls_losses = 0
        densecl_losses = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Create two augmented views
            inputs_q = inputs.to(self.device)
            targets = targets.to(self.device)

            # Create second view for contrastive learning
            inputs_k = self._apply_second_augmentation(inputs).to(self.device)

            # Forward pass
            output = self.model(inputs_q, inputs_k, is_train=True)
            print(f"Output from the models keys : {output.keys()}")
            logits = output["logits"]

            # Classification loss (supervised)
            cls_loss = F.cross_entropy(logits, targets)

            # Dense contrastive loss (self-supervised)
            densecl_loss = output["densecl_loss"]

            # Combined loss
            loss = self.cls_weight * cls_loss + self.dense_weight * densecl_loss

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track statistics
            total_loss += loss.item()
            cls_losses += cls_loss.item()
            densecl_losses += densecl_loss.item()

            # Compute accuracy
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Cls Loss: {cls_loss.item():.4f}, '
                      f'DenseCL Loss: {densecl_loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')

        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = cls_losses / len(self.train_loader)
        avg_densecl_loss = densecl_losses / len(self.train_loader)
        accuracy = 100. * correct / total

        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'densecl_loss': avg_densecl_loss,
            'accuracy': accuracy
        }

    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return None

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass (only classification)
                output = self.model(inputs, im_k=None, is_train=False)
                logits = output["logits"]

                # Compute accuracy
                _, predicted = logits.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        accuracy = 100. * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        return accuracy

    def train(self):
        """Train the model for multiple epochs"""
        best_acc = 0

        for epoch in range(self.num_epochs):
            print(f'\nEpoch: {epoch + 1}/{self.num_epochs}')

            # Train
            train_metrics = self.train_epoch()
            print(f'Train Loss: {train_metrics["loss"]:.4f}, '
                  f'Train Cls Loss: {train_metrics["cls_loss"]:.4f}, '
                  f'Train DenseCL Loss: {train_metrics["densecl_loss"]:.4f}, '
                  f'Train Acc: {train_metrics["accuracy"]:.2f}%')

            # Validate
            val_acc = self.validate()

            # Schedule learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Save best model
            if val_acc and val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'deit_densecl_best.pth')
                print(f'Best model saved with accuracy: {best_acc:.2f}%')

    def _apply_second_augmentation(self, images):
        """Apply a different set of augmentations for the second view"""
        # This function implements the augmentations for the second view
        # Similar to DenseCL's augmentation strategy
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Apply transformations independently to each image
        batch_size = images.size(0)
        transformed_images = torch.zeros_like(images)

        # For each image in the batch
        for i in range(batch_size):
            img = images[i].cpu()
            transformed_images[i] = transform(img)

        return transformed_images


# Example usage
def create_imagenet_dataloaders(path, batch_size=256):
    """Create ImageNet dataloaders"""
    # Define transformations for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transformations for validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root=f'{path}/train',
        transform=train_transform
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=f'{path}/val',
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    return train_loader, val_loader


def main():
    # Create DeiT model with DenseCL
    model = DeiTWithDenseCL(
        model_name='deit_tiny_patch16_224',
        num_classes=1000,
        dense_dim=128,
        queue_size=65536,
        momentum=0.999,
        temperature=0.2,
        dense_weight=0.5
    )

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dataloaders
    train_loader, val_loader = create_imagenet_dataloaders(
        path='/home/cognition/datasets/IMAGENET/imagenet_ILSVRC-2012_ImageNet-1K',
        batch_size=64
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.05
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )

    # Create trainer
    trainer = DeiTDenseCLTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cls_weight=1.0,
        dense_weight=0.5,
        num_epochs=100,
        lr_scheduler=scheduler
    )

    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()