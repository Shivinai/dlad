import torch
import torch.nn as nn

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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model(model, train_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for data, _ in train_loader: 
            data = data.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader.dataset):.6f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader)