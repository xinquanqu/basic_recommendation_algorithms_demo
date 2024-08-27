# Project Overview

## **Directory Structure**

- **data/**
  - Contains raw and processed data files.
  - `data_process.py`: Contains the `Dataset` class to load and preprocess the data.
  
- **models/**
  - Contains different recommender model implementations:
    - `content_based_recommender.py`
    - `popularity_recommender.py`
    - `collaborative_filtering_recommender.py`
    - `hybrid_recommender.py`
  
- **evaluation/**
  - `model_evaluator.py`: Contains the `ModelEvaluator` class to evaluate the performance of the models.
  
- **pipelines/**
  - `main.py`: The entry point of the project.

## **Running the Project**

1. **Navigate to the Project Directory:**
   - Open a terminal and navigate to the root of the project directory (where `src/` is located).

2. **Install Dependencies:**
   - Ensure you have a Python environment set up with the necessary dependencies. If you have a `requirements.txt` file, install dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
3. **Run the Main Script:**
   - You can execute the project by running the `main.py` script from the `src` directory using the `-m` flag:
     ```bash
     python -m pipelines.main
     ```

4. **Review the Output:**
   - The `main.py` script loads the dataset, initializes various recommender models, evaluates them using the `ModelEvaluator`, and then generates a report.

## **Key Components**

- **Dataset Class (`data_process.py`):**
  - Responsible for loading and processing the dataset.
  
- **Model Implementations (in `models/` directory):**
  - Implements different recommendation algorithms:
    - Content-Based
    - Popularity-Based
    - Collaborative Filtering
    - Hybrid

- **Model Evaluation (`model_evaluator.py`):**
  - Evaluates the performance of the models and generates a comparative report.
