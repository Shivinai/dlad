import torch
import torch.nn as nn
from tqdm import tqdm

class SpectrogramAE(nn.Module):
    def __init__(self):
        super(SpectrogramAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        original_size = x.shape[2:] 
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        cropped_output = decoded[:, :, :original_size[0], :original_size[1]]
        
        return cropped_output

def train_model(model, train_loader, optimizer, criterion, device, epochs=20):
    model.train()
    
    with tqdm(range(epochs), desc="Training: ", unit="epoch", ncols=100) as epoch_pbar:
        for epoch in epoch_pbar:
            train_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False, ncols=100) as batch_pbar:
                for data, _ in batch_pbar:
                    data = data.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(data)
                    
                    loss = criterion(outputs, data)
                    loss.backward()
                    optimizer.step()
                    
                    current_loss = loss.item()
                    train_loss += current_loss * data.size(0)
                    
                    batch_pbar.set_postfix({"batch_loss": f"{current_loss:.4f}"})
                    
            epoch_loss = train_loss / len(train_loader.dataset)
            
            epoch_pbar.set_postfix({"epoch_loss": f"{epoch_loss:.6f}"})

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader)