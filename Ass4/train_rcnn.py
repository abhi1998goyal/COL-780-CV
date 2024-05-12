import ImageDataset
import torch 
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def train_model(model, optimizer, train_data, val_data, num_epochs,patience=5):
    print("training the model")
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        model.train()  
        avg_train_loss = 0.0
        num_batches = 0.0
        for batch in create_batches(train_data, batch_size=4):
            images = [x[0].to(device) for x in batch]
            targets = [{'boxes': x[1]['boxes'].to(device), 'labels': x[1]['labels'].to(device)} for x in batch]
          #  images = [x[0] for x in batch]
           # targets = [x[1] for x in batch]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            avg_train_loss += losses.item()
            num_batches += 1
            losses.backward()  
            optimizer.step() 
            
        avg_train_loss /= num_batches

      #  np.random.shuffle(val_data)
        avg_val_loss = 0.0
      #  model.eval()
        with torch.no_grad():
            for batch in create_batches(val_data, batch_size=4):
                images = [x[0].to(device) for x in batch]
                targets = [{'boxes': x[1]['boxes'].to(device), 'labels': x[1]['labels'].to(device)} for x in batch]
              #  images = [x[0] for x in batch]
               # targets = [x[1] for x in batch]
                val_loss_dict = model(images, targets)
                val_losses = sum(loss for loss in val_loss_dict.values())
                avg_val_loss += val_losses.item()
                num_batches += 1
            avg_val_loss /= num_batches   
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_count = 0
            # Save the best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, "rcnn_model_best.pt")
        else:
            no_improvement_count += 1

        # If no improvement for 'patience' epochs, stop training
        if no_improvement_count >= patience:
            print(f"No improvement for {patience} epochs. Stopping training.")
            break

def train():
    train_folder = "/kaggle/input/cv-a4-data/yolo_1k/train"
    val_folder = "/kaggle/input/cv-a4-data/yolo_1k/val"

    train_data = ImageDataset(train_folder).get_all_images_and_boxes()
    val_data = ImageDataset(val_folder).get_all_images_and_boxes()
    print("train and val data created")

    num_classes = 2  # tumor (class 0) + no Tumor (class 1)
    learning_rate = 0.0001
    num_epochs = 100

    model = fasterrcnn_resnet50_fpn(weights = None, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, optimizer, train_data, val_data, num_epochs)

        # Save the trained model
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'mean': mean,
        #     'var': var
        # }, "model.pt")  



if __name__ == "__main__":
    train()