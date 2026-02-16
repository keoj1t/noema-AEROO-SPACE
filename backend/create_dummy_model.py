import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import os

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_dummy_model():
    """Создаем простую модель для тестирования"""
    
    # Создаем папку models если её нет
    os.makedirs("models", exist_ok=True)
    
    # Создаем модель
    model = SimpleAutoencoder()
    model.eval()
    
    # Сохраняем в ONNX
    dummy_input = torch.randn(1, 4)
    onnx_path = "models/model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    
    # Проверяем модель
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✓ Модель создана: {onnx_path}")
    print(f"✓ Размер модели: {os.path.getsize(onnx_path) / 1024:.2f} KB")
    
    # Тестируем загрузку
    try:
        session = ort.InferenceSession(onnx_path)
        print("✓ ONNX Runtime успешно загрузил модель")
        return True
    except Exception as e:
        print(f"✗ Ошибка загрузки: {e}")
        return False

if __name__ == "__main__":
    create_dummy_model()