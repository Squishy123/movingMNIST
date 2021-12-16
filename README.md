# movingMNIST Project

## Installation Instructions
### Install Conda 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html 

## ON WINDOWS:
### 1. Create Conda Environment
```
conda env create -f environment.yml
```

### 2. Activate Conda Environment
```
conda activate moving_mnist
```

### 3. Install Pip Requirements
```
pip install -r requirements.txt
```

## ON LINUX: 
### 1. Create Conda Environment and Install System Dependencies
```
sudo make install_sys
```

### 2. Activate and Install Pip Dependencies
```
conda activate moving_mnist
make install_dep
```

## Running the Project Examples
### Run test
```
python src/test.py
```
