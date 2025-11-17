from bunny_vision_120 import BunnyVision120
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('bunny-vision-120 Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('bunny-vision-120 Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    print("🐰 Initializing bunny-vision-120...")
    
    # Create model
    bunny = BunnyVision120(img_size=120)
    
    # Build architecture
    model = bunny.build_model()
    print(f"📊 Model Parameters: {model.count_params():,}")
    print("\n🏗️  Architecture:")
    model.summary()
    
    # Train the model
    print("\n🚀 Starting training...")
    history = bunny.train('data', epochs=30, batch_size=32)
    
    # Plot results
    plot_training_history(history)
    
    # Save model
    bunny.save_model('bunny_vision_120.h5')
    
    print("\n✅ bunny-vision-120 training complete!")
    print("📁 Model saved as 'bunny_vision_120.h5'")
    print("📈 Training history saved as 'training_history.png'")
    
    # Quick test
    print("\n🧪 Testing model...")
    test_img = input("Enter path to test image (or press Enter to skip): ")
    if test_img and test_img.strip():
        pred_class, confidence, probs = bunny.predict_with_confidence(test_img)
        result = "With Mask" if pred_class == 1 else "Without Mask"
        print(f"Prediction: {result} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()