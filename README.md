# Sign Language to Text Translator

This project translates sign language gestures into text using a Convolutional Neural Network (CNN).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    - Settings are located in `config.py`. You can adjust paths, image sizes, and hyperparameters there.

## Usage

### 1. Data Collection
To collect your own data for training:
```bash
python collect-data.py
```
- A window will open showing the camera feed.
- Press keys `0-9` or `A-Z` to save images for the corresponding class.
- Images are saved in `data/train` or `data/test` depending on the mode in the script (default is train).

### 2. Preprocessing
Process the collected images for training:
```bash
python preprocessing.py
```
- This will process images from `data/train` and save them to `data2/train` and `data2/test`.

### 3. Training
Train the model:
```bash
python train.py
```
- The model will be saved to the `model` directory.

### 4. Application
Run the main application:
```bash
python app.py
```
- The application will open a window showing the camera feed and the predicted text.
- **Controls**:
    - **Space**: Add the current word to the sentence.
    - **Backspace**: Delete the last character of the current word.
    - **C**: Clear the sentence.
    - **Esc**: Exit the application.

## Project Structure
- `app.py`: Main application with UI and prediction logic.
- `train.py`: Script to train the CNN model.
- `collect-data.py`: Script to collect training data.
- `preprocessing.py`: Script to preprocess images.
- `image_processing.py`: Helper functions for image processing.
- `config.py`: Configuration file.
- `requirements.txt`: List of dependencies.
