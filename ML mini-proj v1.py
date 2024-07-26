import os
import time
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

COW = "CowHead_gray"
DOG = "DogHead_gray"
PANDA = "PandaHead_gray"


def get_image_list(directory):
    return [(directory, filename) for filename in os.listdir(directory)]


def plot_random_images(panda, dog, cow):
    start_time = time.time()
    categories = [("Panda", panda), ("Dog", dog), ("Cow", cow)]

    plt.figure(figsize=(30, 5))

    for i, (label, category) in enumerate(categories):
        for j in range(2):
            random_index = random.randint(0, len(category) - 1)
            img_path = os.path.join(category[random_index][0], category[random_index][1])
            img = Image.open(img_path)
            plt.subplot(1, 6, i * 2 + j + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f'{label} {j + 1}')
            plt.axis('off')
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()


def normalize_images(image_list):
    start_time = time.time()
    image_arrays = []
    for path, filename in image_list:
        img_path = os.path.join(path, filename)
        img = Image.open(img_path)
        img_array = np.array(img)
        img_array = img_array.reshape(150 * 150)  # Reshape to 1D array
        img_array = img_array / 255.0  # Normalize pixel values
        image_arrays.append(img_array)
    print("--- %s seconds ---" % (time.time() - start_time))
    return np.array(image_arrays)


def train_svm_classifier(panda_arrays, dog_arrays, cow_arrays):
    start_time = time.time()
    # Combine the arrays and create labels
    X = np.concatenate((panda_arrays, dog_arrays, cow_arrays))
    y = np.concatenate((np.zeros(len(panda_arrays)), np.ones(len(dog_arrays)), np.full(len(cow_arrays), 2)))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    # Initialize the SVM classifier with an RBF kernel
    svm = SVC(kernel='rbf', gamma=0.01, C=1)

    # Train the SVM classifier
    svm.fit(X_train, y_train)
    print("--- %s seconds ---" % (time.time() - start_time))
    return svm, X_test, y_test, X_train, y_train


def evaluate_model(svm, X_test, y_test):
    # Make predictions on the test set
    y_pred = svm.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy: {:.2%}\n".format(accuracy))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Panda', 'Dog', 'Cow'],
                yticklabels=['Panda', 'Dog', 'Cow'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Generate classification report with zero_division parameter
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Convert classification report to DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Format columns
    report_df['support'] = report_df['support'].astype(int)  # Convert support to integers
    report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].round(2)

    print(report_df)
    return cm, report_df



def make_random_predictions(svm, panda_arrays, dog_arrays, cow_arrays):
    start_time = time.time()
    # Select random images
    random_panda_index = random.randint(0, len(panda_arrays) - 1)
    random_dog_index = random.randint(0, len(dog_arrays) - 1)
    random_cow_index = random.randint(0, len(cow_arrays) - 1)

    random_panda = panda_arrays[random_panda_index]
    random_dog = dog_arrays[random_dog_index]
    random_cow = cow_arrays[random_cow_index]

    # Reshape for prediction
    random_panda_reshaped = random_panda.reshape(1, -1)
    random_dog_reshaped = random_dog.reshape(1, -1)
    random_cow_reshaped = random_cow.reshape(1, -1)

    # Make predictions
    panda_prediction = svm.predict(random_panda_reshaped)[0]
    dog_prediction = svm.predict(random_dog_reshaped)[0]
    cow_prediction = svm.predict(random_cow_reshaped)[0]

    # Map labels to animal names
    animal_names = {0: 'Panda', 1: 'Dog', 2: 'Cow'}

    predictions = [
        (random_panda, 'Panda', animal_names[panda_prediction]),
        (random_dog, 'Dog', animal_names[dog_prediction]),
        (random_cow, 'Cow', animal_names[cow_prediction])
    ]

    print("--- %s seconds ---" % (time.time() - start_time))
    return predictions


def plot_predictions(predictions):
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    for i, (img, actual, predicted) in enumerate(predictions):
        # Actual images
        axes[i, 0].imshow(img.reshape(150, 150), cmap='gray')
        axes[i, 0].set_title(f'Actual: {actual}')

        # Predicted images
        axes[i, 1].imshow(img.reshape(150, 150), cmap='gray')
        axes[i, 1].set_title(f'Predicted: {predicted}')

    plt.tight_layout()
    plt.show()


def perform_grid_search(X_train, y_train):
    start_time = time.time()
    # Define the parameter grid for GridSearch
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

    # Initialize GridSearchCV with an RBF kernel
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, refit=True, verbose=5)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)

    # Get the best estimator (model)
    best_svm = grid_search.best_estimator_
    print("--- %s seconds ---" % (time.time() - start_time))
    return best_params, best_svm, grid_search


def save_best_model(grid_search, filename):
    # Get the best estimator (model)
    best_svm = grid_search.best_estimator_

    # Save model
    with open(filename, "wb") as file:
        pickle.dump(best_svm, file)
    print(f"Best model saved to {filename}")


panda = get_image_list(PANDA)
dog = get_image_list(DOG)
cow = get_image_list(COW)

plot_random_images(panda, dog, cow)

# Normalize the images
panda_arrays = normalize_images(panda)
dog_arrays = normalize_images(dog)
cow_arrays = normalize_images(cow)

# Train the SVM classifier
svm, X_test, y_test, X_train, y_train = train_svm_classifier(panda_arrays, dog_arrays, cow_arrays)

# Evaluate the model
conf_matrix, class_report_df = evaluate_model(svm, X_test, y_test)

# Predict with initial SVM
predictions = make_random_predictions(svm, panda_arrays, dog_arrays, cow_arrays)
plot_predictions(predictions)

# Perform grid search to find the best SVM parameters
best_params, best_svm, grid_search = perform_grid_search(X_train, y_train)

# Evaluate the best model
conf_matrix_best, class_report_df_best = evaluate_model(best_svm, X_test, y_test)

# Prediction after grid search
predictions_best = make_random_predictions(best_svm, panda_arrays, dog_arrays, cow_arrays)
plot_predictions(predictions_best)

save_best_model(grid_search, "Bestmodel.pickle")
