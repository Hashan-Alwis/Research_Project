# Project Setup Instructions

Follow these steps to set up your environment and install the necessary packages for this project.

## Prerequisites
- Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) installed.
- Python 3.12 is required.

## Step 1: Create a Virtual Environment
Run the following command to create a virtual environment using Conda:

```bash
conda create -p venv python==3.12
```

## Step 2: Activate the Virtual Environment
Activate the environment using the command:

```bash
conda activate venv/
```

## Step 3: Install Required Packages
Once the environment is activated, install the necessary packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Additional Notes
- Ensure your `requirements.txt` file is in the project directory before running the installation command.
- If you encounter any issues with package installation, verify that your virtual environment is activated by checking the Python version:

```bash
python --version
```

You are now ready to proceed with the project. Happy coding!

