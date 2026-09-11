# Image Classifier API

This project is a small Python-based image classification API built with FastAPI and PyTorch. It loads a trained Convolutional Neural Network (CNN) model for CIFAR-10 image classes and exposes an HTTP endpoint that accepts an uploaded image and returns the predicted class label.

## Project overview

The app was created to:

- train a simple CNN on the CIFAR-10 dataset
- save the trained model weights to a file
- serve the model through a FastAPI API
- accept uploaded images and return prediction results
- provide a simple structure for future testing and deployment

## What this project contains

### Main application

- `app/main.py` - FastAPI app and prediction endpoint
- `app/model.py` - model loading and inference logic
- `app/utils.py` - image preprocessing pipeline
- `app/schemas.py` - API response schema
- `frontend/index.html` - simple upload UI, served by FastAPI at the root URL

### Model and training

- `model/train.py` - CNN model definition and training loop
- `model/evaluate.py` - measures accuracy on the CIFAR-10 test set
- `model/classifier.pth` - saved trained model weights

### Data

- `data/cifar-10-batches-py/` - CIFAR-10 dataset files used for training

### Testing

- `tests/test_predict.py` - endpoint test that uploads a sample image and checks the response
- `tests/sample.jpg` - sample image used by the endpoint test
- `tests/__init__.py` - marks the tests directory as a Python package

### Deployment setup files

- `Dockerfile` - container configuration for running the FastAPI service
- `requirements.txt` - pinned runtime and testing dependencies
- `.gitignore` - excludes the virtual environment, dataset directory, Python caches, environment files, and model weights

---

## Tech stack

- Python 3
- FastAPI
- PyTorch
- TorchVision
- PIL (Python Imaging Library)
- CIFAR-10 dataset

---

## Model architecture

The training logic in `model/train.py` defines a simple CNN:

- `Conv2d` layer 1: input 3 channels, output 16 features
- `Conv2d` layer 2: input 16 channels, output 32 features
- `MaxPool2d` after each convolution
- Flatten layer to connect to fully connected layers
- ReLU activation
- Final output layer with 10 classes

The trained model predicts one of the following CIFAR-10 labels:

- plane
- car
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## How the app works

### 1. Image preprocessing

The app receives an uploaded image in `app/utils.py`.

The preprocessing pipeline does the following:

- opens the uploaded bytes as a PIL image
- converts it to RGB
- resizes to 32x32
- converts to tensor
- normalizes pixel values with mean and std of 0.5
- adds a batch dimension (`unsqueeze(0)`) so it matches the model input

### 2. Model prediction

In `app/model.py`:

- the saved model is loaded from `model/classifier.pth`
- the model is switched to evaluation mode
- the input tensor is passed through the CNN
- the predicted class index is selected with `torch.max()`
- the index is mapped to a human-readable label string

### 3. API endpoint

In `app/main.py`:

- a FastAPI app is created
- a POST endpoint is defined at `/predict`
- the API expects a multipart image file upload
- the file is read, preprocessed, and passed to the model
- the predicted label is returned in JSON format

---

## API endpoint

### POST /predict

Request:

- form-data field named `file`
- uploaded image file, for example JPG or PNG

Example response:

```json
{
  "label": "cat",
  "confidence": null
}
```

Note: the current schema includes a `confidence` field, but the project does not yet compute a real confidence score; it is left as `None`.

---

## Training process

The training code in `model/train.py`:

- loads CIFAR-10 data from `./data`
- creates a DataLoader with batch size 32
- builds the CNN model
- uses CrossEntropyLoss as the loss function
- trains with the Adam optimizer
- runs for 40 epochs
- saves the resulting weights to `model/classifier.pth`

This means the project has a complete training pipeline, even though the API is built to use the saved model weights for inference.

---

## Project run instructions

### 1. Install dependencies

Make sure the required packages are installed:

```bash
pip install -r requirements.txt
```

The dependency versions are recorded in `requirements.txt`, including FastAPI, Uvicorn, PyTorch, TorchVision, Pillow, `python-multipart`, and pytest.

### 2. Train the model

```bash
python model/train.py
```

This trains the model and saves the file `model/classifier.pth`.

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

You can upload an image in the FastAPI Swagger UI and test the `/predict` endpoint.

### 4. Run the tests

```bash
pytest
```

The endpoint test uploads `tests/sample.jpg` and verifies that the API returns HTTP 200 with a `label` field.

### 5. Run with Docker

Build the image from the project root:

```bash
docker build -t image-classifier-api .
```

Start the container:

```bash
docker run --rm -p 8000:8000 image-classifier-api
```

The Dockerfile uses `python:3.11-slim`, installs `requirements.txt`, copies the project into `/app`, and starts Uvicorn on `0.0.0.0:8000`.

---

## Model Performance and Limitations

**Test set accuracy: 66.69%** on the official CIFAR-10 test set (10,000 images the model never saw during training), measured using `model/evaluate.py`. For context, random guessing across 10 classes would score 10%, so the model has clearly learned meaningful patterns, though it falls short of deeper architectures (ResNet-style models can exceed 90% on CIFAR-10).

**Training was iterated in stages**

The model was trained and re-evaluated at 5, 15, and 40 epochs to observe how loss changed over time:

| Epochs | Final Loss (approx.) |
|--------|----------------------|
| 5      | ~1100                |
| 15     | ~1000                |
| 40     | ~90-107               |

Loss dropped significantly between 5 and 40 epochs, but plateaued (with minor fluctuation) toward the end. This, combined with the 66.69% test accuracy, suggests the `SimpleCNN` architecture (2 convolutional layers) is near its practical ceiling on this dataset — further epochs alone are unlikely to meaningfully improve results.

**Accuracy on real-world photos is lower than the test set suggests**

The 66.69% figure reflects performance on CIFAR-10's own images, which are small (32x32), pre-cropped, and centered. Real photos (e.g., from a phone camera) lose significant detail when resized to that resolution and often come from a very different visual distribution, so misclassifications are more common in practice (for example, a real photo of a cat predicted as "deer"). Additionally, CIFAR-10 only covers 10 categories, so any image outside those (like a phone) is forced into the closest-matching class regardless of fit.

**What could improve this further:**

- A deeper CNN architecture (more convolutional layers, batch normalization, dropout) — likely the highest-impact change, since loss has plateaued at the current depth
- Data augmentation during training (random crops, flips) to improve generalization to real-world photos
- Computing and returning real confidence scores (currently the `confidence` field is unused)
- Training on a dataset closer to real-world images, rather than CIFAR-10's small, curated images

---

## Current project status

This project is in a working prototype stage. It includes:

- a trained CNN model for CIFAR-10 classification, evaluated at 66.69% test accuracy
- a FastAPI image prediction service with CORS enabled
- a simple HTML/CSS/JS frontend, served directly by FastAPI at the root URL
- preprocessing and inference logic
- dataset, training, and evaluation pipeline
- a basic API schema and endpoint
- pinned dependencies in `requirements.txt`
- a configured Dockerfile for containerized API startup
- an automated endpoint test using a sample image
- a `.gitignore` for local and generated files

What still needs improvement:

- implement realistic confidence values
- add more automated tests for invalid files and prediction behavior
- validate the API with more real image examples
- review dependency versions for the target deployment environment

---

## Recent project changes

The project has been cleaned up and prepared for repeatable local and container use:

1. Temporary and sensitive files were excluded from version control via `.gitignore`, including the local `.env` file, the downloaded CIFAR-10 archive, and Python `__pycache__` directories.
2. A real prediction endpoint test was added in `tests/test_predict.py`. It sends `tests/sample.jpg` to `/predict` and checks the HTTP response and returned label.
3. `requirements.txt` was populated with pinned application and test dependencies.
4. The Dockerfile was configured to build a Python 3.11 image and start the API with Uvicorn on port 8000.
5. `.gitignore` was configured to keep virtual environments, data, caches, `.env` files, and model weights out of source control.
6. The README was expanded to document the architecture, setup, testing, and deployment workflow.

The `.pytest_cache/` directory may appear after running tests. It is generated by pytest and can be removed when cheaning local test artifacts.

---

## Summary

This project started as an image classification app using CIFAR-10 and ended as a lightweight machine learning API. The main idea is to train a CNN, save its weights, and expose predictions through a FastAPI service. It is a simple but complete example of taking a trained deep learning model and serving it as an HTTP API.

---

## Notes

This README was written based on the actual files currently present in the project. The repository structure and code show that the project is focused on:

- dataset training
- model saving
- image upload prediction
- REST API inference
- beginner-friendly ML deployment workflow