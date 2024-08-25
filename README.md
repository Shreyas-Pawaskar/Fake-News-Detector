
# 📰 Fake News Detector

## Overview

**Fake News Detector** is a machine learning-powered web application built using Streamlit. This app allows users to input a news article and determine whether the news is real or fake using a logistic regression model trained on a dataset of news articles. The app is designed with a modern and professional dark-themed UI for a seamless user experience.

## Features

- **User Input**: Users can enter a news article into the text input field.
- **Prediction**: The app processes the input and predicts whether the news is fake or real.
- **Machine Learning**: The prediction model is based on logistic regression, trained using `sklearn`.
- **Modern UI**: The application features a dark theme with responsive design elements for a professional look.

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python, Logistic Regression (via `sklearn`)
- **Data Processing**: Pandas, NLTK (for text preprocessing)
- **Machine Learning**: Scikit-learn (TfidfVectorizer, Logistic Regression)

## Setup and Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- Pip (Python package installer)

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scriptsctivate`
   ```

3. **Install Dependencies**

   Install the necessary Python packages using `pip`.

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should include:

   ```txt
   streamlit
   numpy
   pandas
   nltk
   scikit-learn
   ```

4. **Download NLTK Data**

   The app uses NLTK for text processing, so you need to download the stopwords data.

   ```python
   import nltk
   nltk.download('stopwords')
   ```

5. **Run the Application**

   Start the Streamlit app.

   ```bash
   streamlit run app.py
   ```

   This will start a local web server, and you can access the app in your browser at `http://localhost:8501`.

## Usage

1. **Input the News Article**: Type or paste a news article into the input field.
2. **Get the Prediction**: The app will display whether the news is REAL or FAKE based on the trained model.

## Dataset

The app uses a dataset of news articles with labels indicating whether they are real or fake. The dataset should be placed in the project directory as `train.csv`.

**Note**: The dataset is not included in this repository. Ensure that you have the dataset before running the app.

## Model Details

- **Vectorization**: The text data is vectorized using `TfidfVectorizer` to convert text into numerical format.
- **Algorithm**: A logistic regression model is used for binary classification (real vs. fake news).
- **Training and Testing**: The dataset is split into training and testing sets to evaluate the model's accuracy.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

### How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
