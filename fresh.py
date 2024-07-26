import os
import time
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

file_path = 'Bestmodel.pickle'
image_directory = 'proj1_test'


def load_and_verify_model(file_path):
    start_time = time.time()
    # Load the pickle file from the specified path
    with open(file_path, 'rb') as file:
        best_svm = pickle.load(file)

    # Verify the model
    print(best_svm)
    print("--- %s seconds ---" % (time.time() - start_time))
    return best_svm


def load_and_normalize_images(image_directory):
    start_time = time.time()
    image_arrays = []

    # List all files in the directory
    for filename in os.listdir(image_directory):
        # Construct the full file path
        img_path = os.path.join(image_directory, filename)

        # Open the image
        img = Image.open(img_path)

        # Convert to array
        img_array = np.array(img)

        # Normalize the image array
        img_array = img_array / 255.0

        # Append the flattened array to the list
        image_arrays.append(img_array.flatten())
    print("--- %s seconds ---" % (time.time() - start_time))
    return np.array(image_arrays)


def plot_images_in_table(image_arrays, num_images=6):
    # Ensure num_images does not exceed the number of available images
    num_images = min(num_images, len(image_arrays))

    # Create a figure with 1 row and num_images columns
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Get the image array
        img_array = image_arrays[i]

        # Reshape the array back to the original image shape
        img_shape = int(np.sqrt(img_array.shape[0]))
        img_array = img_array.reshape(img_shape, img_shape)

        # Display the image
        axes[i].imshow(img_array, cmap='gray')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_images_with_predictions(image_arrays, predictions, num_cols=6):
    # Map labels to animal names
    animal_names = {0: 'Panda', 1: 'Dog', 2: 'Cow'}

    num_rows = len(image_arrays) // num_cols + int(len(image_arrays) % num_cols != 0)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))

    start_time = time.time()
    for idx, ax in enumerate(axs.flat):
        if idx < len(image_arrays):
            # Reshape array back to original image shape if needed, here assuming 150x150
            img_shape = int(np.sqrt(image_arrays[idx].shape[0]))
            ax.imshow(image_arrays[idx].reshape(img_shape, img_shape), cmap='gray')
            ax.set_title(f'Predicted: {animal_names[predictions[idx]]}')
            ax.axis('off')
        else:
            ax.axis('off')
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.tight_layout()
    plt.show()


# Load the model and images
best_svm = load_and_verify_model(file_path)
image_arrays = load_and_normalize_images(image_directory)

plot_images_in_table(image_arrays, num_images=6)

# Make predictions using the loaded SVM model
predictions = best_svm.predict(image_arrays)

# Plot images with predictions as titles
plot_images_with_predictions(image_arrays, predictions, num_cols=6)
