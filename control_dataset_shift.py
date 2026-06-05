from torch.utils.data import Dataset, DataLoader
import scipy
import torch

class ControlShiftDataset(Dataset):
    def __init__(self, data: Dataset, direction: str = 'right',shift: int = 1, edge_correction: bool = True):
        self.original_dataset = data
        self.shift_by = shift
        self.edge_correction = edge_correction
        self.shift_direction = direction
        self.direction_mapping = { # found all by visualising it using the code below (trial and error)
            'up': (-1, 0),
            'down': (1, 0),
            'right': (0, 1),
            'left': (0, -1),
            'up-left': (-1, -1),
            'down-left': (1, -1),
            'up-right': (-1, 1),
            'down-right': (1, 1)
            
        }

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        np_img = img.numpy()
        
        shift_y = self.shift_by * self.direction_mapping[self.shift_direction][0]
        shift_x = self.shift_by * self.direction_mapping[self.shift_direction][1]
        shift = (0, shift_y, shift_x) # shift height and width 
        
        correction = 'reflect' if self.edge_correction else 'constant'
        shifted_img = scipy.ndimage.shift(np_img, shift=shift, mode=correction, cval=0.0)
        shifted_tensor = torch.from_numpy(shifted_img).float()
        
        return shifted_tensor, label
    


def test_shift_invariance(model, dataloader: DataLoader, device: str = 'cpu'):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # outputs logits for all classes 
            # for cifar [64,10] -> [batch, classes hot encoded]
            #print(outputs.shape)
            
            _, predicted = torch.max(outputs.data, 1) # highest score class
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy
