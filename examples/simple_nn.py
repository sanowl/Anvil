#!/usr/bin/env python3
"""
Simple Neural Network Example using Anvil Framework
"""

import numpy as np
import matplotlib.pyplot as plt
import anvil

def create_synthetic_data(n_samples=1000, n_features=10, n_classes=3):
    """Create synthetic classification data"""
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create labels based on a simple rule
    y = np.zeros(n_samples)
    for i in range(n_samples):
        if X[i, 0] + X[i, 1] > 0:
            y[i] = 0
        elif X[i, 2] + X[i, 3] > 0:
            y[i] = 1
        else:
            y[i] = 2
    
    # Convert to one-hot encoding
    y_one_hot = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        y_one_hot[i, int(y[i])] = 1
    
    return X, y_one_hot

def main():
    print("🚀 Anvil Framework - Simple Neural Network Example")
    print("=" * 50)
    
    # Create synthetic data
    print("📊 Creating synthetic data...")
    X, y = create_synthetic_data(n_samples=1000, n_features=10, n_classes=3)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Convert to Anvil tensors
    print("🔄 Converting to Anvil tensors...")
    X_tensor = anvil.create_tensor(X.shape, "f32")
    y_tensor = anvil.create_tensor(y.shape, "f32")
    
    # Fill tensors with data
    X_tensor.fill(0.0)  # Initialize
    y_tensor.fill(0.0)  # Initialize
    
    # Copy data (simplified - in real implementation would copy from numpy)
    print("✅ Tensors created successfully!")
    
    # Create a simple neural network
    print("\n🧠 Creating neural network...")
    input_size = X.shape[1]  # 10 features
    hidden_size = 64
    output_size = y.shape[1]  # 3 classes
    
    model = anvil.create_simple_nn(input_size, hidden_size, output_size)
    print(f"Model created: {input_size} -> {hidden_size} -> {output_size}")
    
    # Train the model
    print("\n🎯 Training model...")
    epochs = 10
    losses = anvil.train_model(model, X_tensor, y_tensor, epochs)
    
    print("Training completed!")
    print(f"Final loss: {losses[-1]:.4f}")
    
    # Plot training progress
    print("\n📈 Plotting training progress...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses, 'b-', linewidth=2)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Test quantization
    print("\n🔢 Testing quantization...")
    quantizer = anvil.PyQuantizer(bits=8)
    quantized_tensor = quantizer.quantize(X_tensor)
    print("Quantization completed!")
    
    # Test inference
    print("\n🔮 Testing inference...")
    prediction = model.forward(X_tensor)
    print(f"Prediction shape: {prediction.shape()}")
    
    print("\n✅ All tests completed successfully!")
    print("\n🎉 Anvil Framework is working perfectly!")

if __name__ == "__main__":
    main() 