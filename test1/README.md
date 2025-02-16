# Hybrid Depth Estimation Project

This project implements a hybrid depth estimation model that combines three state-of-the-art techniques: MiDaS, AdaBins, and DPT-Large. The goal is to leverage the strengths of each model to produce more accurate depth maps from input images.

## Project Structure

```
hybrid-depth-estimation
├── models
│   ├── __init__.py
│   ├── midas_model.py
│   ├── adabins_model.py
│   └── dpt_large_model.py
├── src
│   ├── hybrid_depth.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Requirements

To run this project, you need to install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

The `requirements.txt` file includes the following libraries:

- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Usage

1. Place your input image in the `src` directory or update the image path in `src/hybrid_depth.py`.
2. Run the main script:

```
python src/hybrid_depth.py
```

3. The script will generate depth maps using MiDaS, AdaBins, and DPT-Large, and display the results.

## Models

- **MiDaS**: A model designed for monocular depth estimation that provides high-quality depth maps.
- **AdaBins**: A model that adapts the binning strategy for depth estimation, improving accuracy in various scenarios.
- **DPT-Large**: A model that utilizes a transformer-based architecture for depth estimation, known for its robustness.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.