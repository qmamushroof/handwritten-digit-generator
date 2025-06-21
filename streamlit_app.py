import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- MODEL DEFINITION ---
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(28*28 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def encode(self, x, y):
        h1 = torch.relu(self.fc1(torch.cat([x, y], 1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = torch.relu(self.fc3(torch.cat([z, y], 1)))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# --- LOAD TRAINED MODEL ---
device = torch.device("cpu")
model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

# --- STREAMLIT APP ---
st.title("Handwritten Digit Generator (0-9)")

digit = st.selectbox("Select digit to generate:", list(range(10)))

if st.button("Generate 5 Images"):
    num_samples = 5
    y = torch.eye(10)[digit].repeat(num_samples, 1)
    z = torch.randn(num_samples, 20)
    with torch.no_grad():
        samples = model.decode(z, y).numpy().reshape(num_samples, 28, 28)
    
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axs[i].imshow(samples[i], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
