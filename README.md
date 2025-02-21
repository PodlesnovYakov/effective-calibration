# Calibration stochastic process with neural networks

Content:
* About project
* Working in conda environment
* Makefile

## About project 
- At the moment, the calibration of the Ornstein-Uhlenbeck stochastic process is presented.
- The best model was trained using a Tesla P40.
- Calibration of the neural network is performed in comparison with classical methods such as the method of moments (MM) and the maximum likelihood method (MLE).

## Working in a conda environment
Creating and activating an environment

```bash
$ make env
$ conda activate calibrate
```

Installing the package

```bash
$ make install
```

## Makefile

- `make env` - creating a conda environment named calibrate and python=3.8.*
  (you need to additionally activate the environment via `conda activate calibrate`)
- `make install` - installing dependencies from requirements.txt
- `make kernel` - creating a kernel for a jupyter notebook

