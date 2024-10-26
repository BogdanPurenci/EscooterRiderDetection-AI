
# Escooter Detection Using Multilayer Perceptron

This project implements a neural network (Multilayer Perceptron) to detect electric scooter riders in traffic from images. The network is trained using labeled image data of both scooter and non-scooter riders, and it uses a simple forward and backward propagation process with sigmoid activation to classify images.

## Project Structure

- `main.py`: Contains the main script to load the dataset, define and train the neural network, and evaluate its accuracy.
- `no_escooter/`: Folder containing images without e-scooter riders (used for training).
- `with_escooter/`: Folder containing images with e-scooter riders (used for training).
  
## Requirements

To run this project, you need the following Python libraries:
- `numpy`
- `Pillow`
- `scikit-learn`
- `matplotlib`

You can install the dependencies with:
```bash
pip install numpy Pillow scikit-learn matplotlib
```

## How It Works

1. **Data Preparation**:
   - The `incarca_imaginile()` function loads images from two folders: `no_escooter` (negative examples) and `with_escooter` (positive examples).
   - The images are converted to grayscale, resized to 50x50 pixels, and flattened into 1D arrays for input into the neural network.
  
2. **Neural Network Architecture**:
   - Input Layer: 2500 nodes (corresponding to the 50x50 image size).
   - Hidden Layer: 250 nodes.
   - Output Layer: 1 node (binary classification: 0 for no e-scooter, 1 for e-scooter).
   - The network uses the sigmoid activation function for both layers.

3. **Training**:
   - The `antrenare()` method performs forward and backward propagation to minimize error. The error is visualized over epochs to monitor training progress.
   - The model is trained for 100 epochs with a learning rate of 0.1.

4. **Testing**:
   - After training, the model is tested on unseen images. The `test()` method calculates the classification accuracy of the network on the test dataset.

## Usage

1. **Prepare Data**:
   Place images in the appropriate folders (`no_escooter/` and `with_escooter/`) for training.

2. **Run the Script**:
   After setting up the dataset, run the main script to train the neural network and evaluate its accuracy.
   ```bash
   python main.py
   ```

3. **Results**:
   After training, the script will print the accuracy of the model on the test set:
   ```
   Acuratețea pe setul de testare: 85.00%
   ```

## Future Improvements

- Experiment with different architectures (e.g., more hidden layers, varying number of neurons).
- Enhance the dataset with more varied images.
- Apply data augmentation techniques to improve generalization.

## License

This project is licensed under the MIT License.
