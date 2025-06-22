import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- VAE model ---
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- Load trained model ---
model = VAE()
model.load_state_dict(torch.load("vae_model/mnist_vae.pth", map_location=torch.device('cpu')))
model.eval()

# --- Streamlit UI ---
st.title("🧠 Handwritten Digit Generator (MNIST)")
digit = st.selectbox("Select a digit (0-9):", list(range(10)))

if st.button("Generate 5 Images"):
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        z = torch.randn(1, 20)
        with torch.no_grad():
            sample = model.decode(z).view(28, 28).numpy()
        axes[i].imshow(sample, cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)

st.markdown("\nModel dilatih dengan VAE di dataset MNIST.")