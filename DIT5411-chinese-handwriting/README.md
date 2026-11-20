# DIT5411-YourName: Traditional Chinese Handwriting Recognition

A TensorFlow project that recognises handwritten Traditional Chinese characters by comparing multiple neural-network architectures.

## Dataset

Source: [AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)

The available workspace data is located in:

```text
Traditional-Chinese-Handwriting-Dataset-master/data/cleaned_data(50_50)-20200420T071507Z-001/cleaned_data(50_50)/
```

Each image filename is formatted as `character_sampleNumber.png`. Images are loaded as 50 by 50 pixel, single-channel grayscale images.

## Training and Testing Split

For every character with more than 40 samples:

- Samples numbered `_0` through `_39` are training samples.
- All later-numbered samples are testing samples.
- The training samples are augmented through random rotation, zoom, and translation.
- Augmentation produces at least 200 training samples per character.
- Test images are never augmented or included in training.
- Character labels are generated dynamically from the character folders/classes discovered in the dataset.

The script does not shuffle across character groups when selecting training and testing samples. It therefore follows the assignment's first-40-samples rule instead of using random stratified splitting.

## Models Compared

| Model | Architecture summary | Parameters |
|---|---|---:|
| MLP | Flatten 50x50, Dense 512, Dropout 0.30, Dense 256, Softmax output | 1,699,937 |
| Small CNN | Conv32, Conv64, Conv128, global average pooling, Dropout 0.35 | 237,281 |
| Deep CNN | Conv32 x2, Conv64 x2, Conv128, batch normalisation, Dense 256, Dropout 0.40 | 460,865 |

All output layers use a `Softmax` activation with one neuron per discovered character class.

## Local Results

The local 4,803-character subset run used CPU training with a batch size of 128. Early stopping used validation accuracy and restored the best weights.

| Rank | Model | Test accuracy | Test loss | Epochs run |
|---:|---|---:|---:|---:|
| 1 | Deep CNN | 64.7669% | 1.4405 | 8 |
| 2 | Small CNN | 0.1228% | 7.4095 | 7 |
| 3 | MLP | 0.0934% | 7.0213 | 4 |

The Deep CNN achieved its best validation accuracy during training and was restored through early stopping. The Small CNN and MLP results are substantially lower.

These are the results from the available workspace subset. They are not results for all 13,065 characters in the complete original dataset.

Detailed values are stored in `output/model_results.json` and `output/model_results.csv`.

## Augmentation Method

Training augmentation is implemented in TensorFlow by a geometric augmentation sequence:

- Rotation: up to approximately 8 degrees
- Zoom: from -12% to +12%
- Translation: up to 8% horizontally and vertically

The implementation does not currently apply shear; rotation and zoom provide sufficient geometric variation for the assignment and can be extended with shear if required.

The augmentation is applied only while the training dataset is iterated. Testing data is decoded without augmentation.

## Environment Setup

The project was run with Python 3.11 and TensorFlow 2.21 in a local virtual environment.

```powershell
cd C:\Users\user\Desktop\asm\DIT5411-chinese-handwriting
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU is the default execution device.

- `--device cpu` forces CPU-only execution.
- `--device gpu` requires a visible TensorFlow GPU.
- `--device auto` uses a GPU when available and otherwise uses CPU.

## Run a Smoke Test

A smoke test evaluates the pipeline using only 50 discovered classes:

```powershell
.\.venv\Scripts\python.exe train_models.py --device cpu --smoke-test --epochs 1 --batch-size 32 --data-dir "..\Traditional-Chinese-Handwriting-Dataset-master\data\cleaned_data(50_50)-20200420T071507Z-001\cleaned_data(50_50)"
```

## Run Full Training

```powershell
.\.venv\Scripts\python.exe train_models.py --device cpu --epochs 10 --batch-size 128 --data-dir "..\Traditional-Chinese-Handwriting-Dataset-master\data\cleaned_data(50_50)-20200420T071507Z-001\cleaned_data(50_50)"
```

The script will:

1. Discover character classes from PNG filenames.
2. Create the first-40/remaining split.
3. Augment training samples.
4. Train MLP, Small CNN, and Deep CNN models.
5. Evaluate each model on the original test set.
6. Save models and ranked results.

## Output Files

The `output/` directory contains:

- `mlp.keras`: trained MLP model
- `small_cnn.keras`: trained Small CNN model
- `deep_cnn.keras`: trained Deep CNN model
- `model_results.json`: ranked model results
- `model_results.csv`: results in spreadsheet format
- `characters.json`: character classes used by the run

## Reproducibility

The script fixes NumPy, Python, and TensorFlow random seeds. TensorFlow early stopping restores the model with the highest validation accuracy, and test evaluation uses only the original test images.

## Scaling to the Full 13,065-Class Dataset

To run the assignment's complete 13,065-character target:

1. Download and extract the complete dataset.
2. Pass the complete `cleaned_data` directory with `--data-dir`.
3. Ensure the environment has enough memory and training time.
4. For this class count, the final output layer will contain 13,065 neurons automatically.

A GPU runtime in WSL2 or Google Colab is strongly recommended for the complete dataset.

## References

- [AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)
- [Introduction to OpenCV](https://github.com/darshitajain/Introduction-to-OpenCV)
- [TensorFlow Keras documentation](https://www.tensorflow.org/guide/keras)
