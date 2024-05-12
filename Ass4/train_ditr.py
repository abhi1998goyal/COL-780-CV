import torchvision
import os
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from torch.utils.data import DataLoader
from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
import torch
import torch.optim as optim
import CocoDetection

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = CocoDetection(img_folder='/kaggle/input/cv-a4-data/coco_1k/coco_1k/train2017', processor=processor, train=True)
val_dataset = CocoDetection(img_folder='/kaggle/input/cv-a4-data/coco_1k/coco_1k/val2017', processor=processor, train=False)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=4)

model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

num_epochs = 2  
best_val_loss = float('inf')
patience = 5
no_improvement_count = 0

def train():
    for epoch in range(num_epochs):

        avg_train_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            avg_train_loss+=loss
            loss.backward()
            optimizer.step()
    
        avg_train_loss /= len(train_dataloader)    

        # Validation step
        model.eval() 
        with torch.no_grad():
            avg_val_loss = 0.0
            for batch_idx, batch in enumerate(val_dataloader):
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                avg_val_loss += outputs.loss.item()

            avg_val_loss /= len(val_dataloader)
            
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        scheduler.step(avg_val_loss)
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_count = 0
            torch.save({'model_state_dict': model.state_dict()}, "model.pt")
        else:
            no_improvement_count += 1
            if no_improvement_count == patience and epoch > 5:
                print("Validation loss did not improve. Stopping early.")
                break

if __name__ == "__main__":
    train()