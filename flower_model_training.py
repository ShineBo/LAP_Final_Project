import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import psutil
import platform
from pathlib import Path
import pandas as pd
from PIL import Image
import cv2

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class SystemInfo:
    """Utility class to gather system information for performance analysis"""
    
    @staticmethod
    def get_system_specs():
        """Get system specifications"""
        specs = {
            'os': platform.system(),
            'python_version': platform.python_version(),
            'tensorflow_version': tf.__version__,
            'total_ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'cpu_cores': psutil.cpu_count(logical=True),
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0
        }
        return specs
    
    @staticmethod
    def monitor_resources():
        """Monitor system resources during training"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        }

class DatasetAnalyzer:
    """Class to analyze dataset characteristics and prepare data insights"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.class_distribution = {}
        self.total_images = 0
        
    def analyze_dataset(self):
        """Analyze dataset structure and class distribution"""
        print("Analyzing Dataset Structure...")
        
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                image_count = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                self.class_distribution[class_name] = image_count
                self.total_images += image_count
        
        return self.class_distribution
    
    def plot_class_distribution(self):
        """Plot class distribution"""
        plt.figure(figsize=(10, 6))
        classes = list(self.class_distribution.keys())
        counts = list(self.class_distribution.values())
        
        plt.subplot(1, 2, 1)
        plt.bar(classes, counts, color='skyblue')
        plt.title('Class Distribution')
        plt.xlabel('Flower Types')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution Percentage')
        
        plt.tight_layout()
        plt.show()
        
        return self.class_distribution
    
    def get_sample_images(self, num_samples=5):
        """Display sample images from each class"""
        fig, axes = plt.subplots(len(self.class_distribution), num_samples, 
                                figsize=(15, 3 * len(self.class_distribution)))
        
        for i, class_name in enumerate(self.class_distribution.keys()):
            class_path = os.path.join(self.data_dir, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for j in range(min(num_samples, len(image_files))):
                img_path = os.path.join(class_path, image_files[j])
                img = Image.open(img_path)
                
                if len(self.class_distribution) == 1:
                    axes[j].imshow(img)
                    axes[j].set_title(f'{class_name}')
                    axes[j].axis('off')
                else:
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f'{class_name}')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()

class ModelArchitectures:
    """Different model architectures for comparison"""
    
    @staticmethod
    def create_custom_cnn(input_shape, num_classes):
        """Create custom CNN from scratch"""
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    @staticmethod
    def create_mobilenet_model(input_shape, num_classes):
        """Create MobileNetV2 based model"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model, base_model
    
    @staticmethod
    def create_vgg16_model(input_shape, num_classes):
        """Create VGG16 based model"""
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model, base_model

class TrainingCallbacks:
    """Custom callbacks for training monitoring"""
    
    @staticmethod
    def get_callbacks(model_name="flower_model"):
        """Get training callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                f'{model_name}_training_log.csv',
                append=True
            )
        ]
        return callbacks

class PerformanceAnalyzer:
    """Comprehensive performance analysis"""
    
    def __init__(self):
        self.metrics_history = {}
        self.training_time = 0
        self.inference_times = []
    
    def analyze_training_performance(self, history):
        """Analyze training performance metrics"""
        metrics = {
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_accuracy': max(history.history['val_accuracy']),
            'epochs_trained': len(history.history['accuracy']),
            'overfitting_score': history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
        }
        return metrics
    
    def plot_detailed_training_history(self, history, fine_tune_history=None):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Accuracy plots
        epochs1 = range(1, len(history.history['accuracy']) + 1)
        axes[0, 0].plot(epochs1, history.history['accuracy'], 'b-', label='Training Accuracy')
        axes[0, 0].plot(epochs1, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plots
        axes[0, 1].plot(epochs1, history.history['loss'], 'b-', label='Training Loss')
        axes[0, 1].plot(epochs1, history.history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[0, 2].plot(epochs1, history.history['lr'], 'g-')
            axes[0, 2].set_title('Learning Rate')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True)
        
        # Fine-tuning history (if available)
        if fine_tune_history:
            epochs2 = range(len(epochs1) + 1, len(epochs1) + len(fine_tune_history.history['accuracy']) + 1)
            axes[1, 0].plot(list(epochs1) + list(epochs2), 
                           history.history['accuracy'] + fine_tune_history.history['accuracy'], 
                           'b-', label='Training Accuracy')
            axes[1, 0].plot(list(epochs1) + list(epochs2), 
                           history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], 
                           'r-', label='Validation Accuracy')
            axes[1, 0].axvline(x=len(epochs1), color='k', linestyle='--', label='Fine-tuning Start')
            axes[1, 0].set_title('Complete Training History')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Validation accuracy improvement
        val_acc_diff = np.diff(history.history['val_accuracy'])
        axes[1, 1].plot(range(2, len(history.history['val_accuracy']) + 1), val_acc_diff, 'g-')
        axes[1, 1].set_title('Validation Accuracy Improvement per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Change')
        axes[1, 1].grid(True)
        
        # Training vs Validation Loss Difference
        loss_diff = np.array(history.history['loss']) - np.array(history.history['val_loss'])
        axes[1, 2].plot(epochs1, loss_diff, 'purple')
        axes[1, 2].set_title('Training vs Validation Loss Difference')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss Difference')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def benchmark_inference_speed(self, model, test_generator, num_samples=100):
        """Benchmark inference speed"""
        print("Benchmarking inference speed...")
        times = []
        
        # Get a batch of test images
        test_batch = next(test_generator)
        test_images = test_batch[0]
        
        # Fix: Use the minimum of requested samples and available images
        actual_samples = min(num_samples, len(test_images))
        
        for i in range(actual_samples):
            start_time = time.time()
            _ = model.predict(np.expand_dims(test_images[i], axis=0), verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        inference_stats = {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'images_per_second': 1.0 / np.mean(times)
        }
        
        return inference_stats

class FlowerClassifierAdvanced:
    """Advanced Flower Classifier with comprehensive features"""
    
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.base_model = None
        self.class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        self.training_history = None
        self.fine_tune_history = None
        self.performance_analyzer = PerformanceAnalyzer()
        self.system_info = SystemInfo()
        
        # Initialize experiment tracking
        self.experiment_log = {
            'start_time': datetime.now().isoformat(),
            'system_specs': self.system_info.get_system_specs(),
            'hyperparameters': {
                'img_height': img_height,
                'img_width': img_width,
                'batch_size': batch_size
            }
        }
    
    def setup_data_pipeline(self, data_dir, validation_split=0.2, test_split=0.1):
        """Setup comprehensive data pipeline"""
        print("Setting up data pipeline...")
        
        # Analyze dataset first
        analyzer = DatasetAnalyzer(data_dir)
        class_dist = analyzer.analyze_dataset()
        analyzer.plot_class_distribution()
        analyzer.get_sample_images()
        
        # Create data generators with advanced augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split + test_split
        )
        
        val_test_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split + test_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        # Validation generator
        validation_generator = val_test_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        # Update class names from generator
        self.class_names = list(train_generator.class_indices.keys())
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        print(f"Classes found: {self.class_names}")
        
        return train_generator, validation_generator, class_dist
    
    def build_model(self, architecture='mobilenet', compile_model=True):
        """Build model with specified architecture"""
        print(f"Building model with {architecture} architecture...")
        
        input_shape = (self.img_height, self.img_width, 3)
        num_classes = len(self.class_names)
        
        if architecture == 'mobilenet':
            self.model, self.base_model = ModelArchitectures.create_mobilenet_model(input_shape, num_classes)
        elif architecture == 'vgg16':
            self.model, self.base_model = ModelArchitectures.create_vgg16_model(input_shape, num_classes)
        elif architecture == 'custom':
            self.model = ModelArchitectures.create_custom_cnn(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        if compile_model:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        # Log model architecture
        self.experiment_log['architecture'] = architecture
        self.experiment_log['total_params'] = self.model.count_params()
        
        return self.model
    
    def train_model_comprehensive(self, train_generator, validation_generator, 
                                 initial_epochs=20, fine_tune_epochs=10):
        """Comprehensive training with monitoring"""
        print("Starting comprehensive training...")
        
        # Get callbacks
        callbacks = TrainingCallbacks.get_callbacks("flower_classifier")
        
        # Monitor system resources
        start_resources = self.system_info.monitor_resources()
        start_time = time.time()
        
        # Initial training
        print("Phase 1: Initial training with frozen base model...")
        self.training_history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            epochs=initial_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning (if base model exists)
        if self.base_model is not None:
            print("Phase 2: Fine-tuning with unfrozen layers...")
            self.base_model.trainable = True
            
            # Fine-tune from this layer onwards
            fine_tune_at = len(self.base_model.layers) // 2
            
            for layer in self.base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            # Use lower learning rate for fine-tuning
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Continue training
            self.fine_tune_history = self.model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // self.batch_size,
                epochs=fine_tune_epochs,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // self.batch_size,
                verbose=1
            )
        
        # Calculate training time and resource usage
        end_time = time.time()
        end_resources = self.system_info.monitor_resources()
        
        self.performance_analyzer.training_time = end_time - start_time
        
        # Log training results
        self.experiment_log['training_time'] = self.performance_analyzer.training_time
        self.experiment_log['resource_usage'] = {
            'start': start_resources,
            'end': end_resources
        }
        
        print(f"Training completed in {self.performance_analyzer.training_time:.2f} seconds")
        
        return self.training_history, self.fine_tune_history
    
    def comprehensive_evaluation(self, validation_generator):
        """Comprehensive model evaluation"""
        print("Performing comprehensive evaluation...")
        
        # Plot training history
        self.performance_analyzer.plot_detailed_training_history(
            self.training_history, self.fine_tune_history
        )
        
        # Get predictions
        validation_generator.reset()
        predictions = self.model.predict(validation_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = validation_generator.classes
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_classes, predicted_classes, average=None
        )
        
        # Per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        
        # Classification report
        report = classification_report(true_classes, predicted_classes, 
                                     target_names=self.class_names)
        print("Detailed Classification Report:")
        print(report)
        
        # Enhanced confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix (Counts)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.subplot(2, 2, 2)
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Per-class performance
        plt.subplot(2, 2, 3)
        metrics_df = pd.DataFrame(class_metrics).T
        metrics_df[['precision', 'recall', 'f1_score']].plot(kind='bar')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Prediction confidence distribution
        plt.subplot(2, 2, 4)
        max_probs = np.max(predictions, axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, color='skyblue')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Maximum Probability')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Benchmark inference speed
        inference_stats = self.performance_analyzer.benchmark_inference_speed(
            self.model, validation_generator
        )
        
        print("\nInference Performance:")
        print(f"Average inference time: {inference_stats['mean_inference_time']:.4f} seconds")
        print(f"Images per second: {inference_stats['images_per_second']:.2f}")
        
        # Log evaluation results
        self.experiment_log['evaluation'] = {
            'accuracy': float(accuracy),
            'class_metrics': {k: {kk: float(vv) for kk, vv in v.items()} 
                            for k, v in class_metrics.items()},
            'inference_stats': inference_stats
        }
        
        return {
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': cm,
            'inference_stats': inference_stats
        }
    
    def predict_single_image(self, image_path, show_confidence=True):
        """Predict single image with confidence scores"""
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        result = {
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'all_probabilities': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }
        
        if show_confidence:
            print(f"Predicted: {result['predicted_class']} (Confidence: {confidence:.2%})")
            print("All probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name}: {prob:.2%}")
        
        return result
    
    def save_model_and_logs(self, model_path='flower_classifier_final.h5', 
                           log_path='experiment_log.json'):
        """Save model and experiment logs"""
        # Save model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save experiment log
        self.experiment_log['end_time'] = datetime.now().isoformat()
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
        print(f"Experiment log saved to {log_path}")
        
        return model_path, log_path

# Example usage and workflow
def main():
    """Main workflow demonstrating comprehensive flower classification"""
    
    print("="*80)
    print("COMPREHENSIVE FLOWER CLASSIFICATION SYSTEM")
    print("="*80)
    
    # Step 1: Initialize system
    classifier = FlowerClassifierAdvanced(img_height=224, img_width=224, batch_size=32)
    
    # Step 2: Get dataset path
    data_dir = input("Enter the path to your flower dataset directory: ").strip()
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist!")
        return
    
    # Step 3: Setup data pipeline
    train_gen, val_gen, class_dist = classifier.setup_data_pipeline(
        data_dir, validation_split=0.2
    )
    
    # Step 4: Build model
    model = classifier.build_model(architecture='mobilenet')
    print(f"\nModel Summary:")
    model.summary()
    
    # Step 5: Train model
    train_hist, fine_tune_hist = classifier.train_model_comprehensive(
        train_gen, val_gen, initial_epochs=15, fine_tune_epochs=8
    )
    
    # Step 6: Comprehensive evaluation
    evaluation_results = classifier.comprehensive_evaluation(val_gen)
    
    # Step 7: Save results
    model_path, log_path = classifier.save_model_and_logs()
    
    # Step 8: Test single image prediction (optional)
    test_image = input("\nEnter path to test image (or press Enter to skip): ").strip()
    if test_image and os.path.exists(test_image):
        result = classifier.predict_single_image(test_image)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Final Validation Accuracy: {evaluation_results['accuracy']:.2%}")
    print(f"Model saved to: {model_path}")
    print(f"Experiment log saved to: {log_path}")

if __name__ == "__main__":
    main()