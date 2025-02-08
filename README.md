# Cats vs Dogs Classification using CNN Python Keras

## Project Overview
This project aims to develop a Convolutional Neural Network (CNN) to classify images as either containing a cat or a dog. The trained model is deployed using a Graphical User Interface (GUI) to facilitate user interaction and prediction visualization.

## Dataset
- **Name**: Cats vs Dogs
- **Source**: [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/overview)
- **Description**: Contains images of cats and dogs in various backgrounds with RGB color space.
- **Data Split**:
  - Training: 16,000 images
  - Validation: 4,000 images
  - Testing: 5,000 images

## System Requirements
### Hardware
- RAM: 15GB
- GPU RAM: 15GB
- Hard Disk: 4GB
- Operating Systems: Windows, MacOS, Linux
- Processor: Intel, Apple Silicon

### Software
- **Programming Language**: Python
- **Tools**: Jupyter Notebook, Google Colab
- **Libraries**: Numpy, Keras, TensorFlow, Scikit-Learn, Tkinter

## Usage
1. **Run the CNN Model:**
   Open `CNNModel.ipynb` in Jupyter Notebook and execute the cells to train the model.

2. **Deploy the GUI:**
   Open `GUI.ipynb` to launch the graphical interface. Use the GUI to upload images and get classification results.

## Project Architecture

![Proposed Architecture](https://github.com/praneethravirala/Cats-vs-Dogs-Classification-using-CNN-Python-Keras/blob/main/ProposedArchitecture.png)

## Model Architecture
- **Input Layer**: 128x128 image size
- **Convolutional Layers**: 2 layers with 32 filters and 2x2 window size
- **Pooling Layers**: 2 max-pooling layers with 2x2 window size
- **Flatten Layer**: Converts input into a 1D vector
- **Dropout Layer**: Dropout factor of 0.5
- **Dense Layer**: 256 neurons

### Training Parameters
- **Optimizer**: Stochastic Gradient Descent
- **Learning Rate**: 0.001
- **Loss Function**: Cross-Entropy
- **Epochs**: 20

## Results
- **Accuracy**: ~80% on training and validation datasets
- **Performance Metrics**:
  - Precision: >75%
  - Recall: >75%
  - F1 Score: >75%
- **Visualizations**: Accuracy and loss graphs over epochs, confusion matrix, and classification report.

## Challenges Faced
- Limited GPU resources and long CPU training times
- Optimizing learning rate, dropout, and model architecture
- Handling overfitting and underfitting issues

## Key Learnings
- Mastered data preprocessing techniques like augmentation and normalization
- Gained experience in CNN model building with Keras and TensorFlow
- Learned GUI integration with deep learning models using Tkinter

## Future Enhancements
- Implement pre-trained models like VGG16, VGG19, and ResNet50 for better performance
- Add more convolutional layers and filters to improve feature extraction
- Use ensemble techniques to combine multiple models
- Enhance GUI with more graphical elements and features
- Package the project as a standalone executable using PyInstaller

## Contributors
- **Praneeth Ravirala** (G01448129)
- **Shalvi Sanjay Lale** (G01419005)

