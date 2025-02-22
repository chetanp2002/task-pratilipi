# LightFM Recommendation System

## Overview
This project implements a recommendation system using the **LightFM** library. It builds personalized content recommendations based on user interactions and metadata, leveraging collaborative filtering and content-based filtering.

## Dataset
The project uses two datasets:
- **User_interaction.csv**: Contains user interactions with content, including read percentage and timestamps.
- **Meta_data.csv**: Contains metadata for each item, such as author, category, and reading time.

## Features
- **User-Item Interaction Filtering**: Uses a read percentage threshold to consider meaningful interactions.
- **Train-Test Split**: 75% of interactions are used for training, and 25% for testing.
- **Item Feature Engineering**: Uses user_id, category, pratilipi_id, reading_time, category_name, author_id as features.
- **LightFM Model Training**: Trains a model using the **logistic** loss function.
- **Recommendation Function**: Provides recommendations for a given user.

## Tech Stack
- **Python**
- **Pandas, NumPy** (Data Processing)
- **LightFM** (Recommendation Model)
- **Streamlit** (Deployment)

## ⚙️ Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/repo-name.git
   cd repo-name
    ```
2. Clone the repository:
    ```sh
   pip install -r requirements.txt

    ```
3. Run the Streamlit app:
    ```sh
   streamlit run app.py

    ```

## Usage
1. **Load Data**: Reads user interactions and metadata from CSV files.
2. **Preprocess Data**: Converts timestamps and filters interactions.
3. **Prepare Dataset**: Fits user and item mappings and builds interaction matrices.
4. **Train Model**: Uses LightFM to train on interactions.
6. **Generate Recommendations**: Retrieves recommendations for a sample user.

## Example Output
```
Recommendations for user 5506791974854999: [5506791979223815, 5506791970045925, 5506791968261668, 5506791991878999, 5506791973354582]
```

## License
This project is open-source and available under the MIT License.
