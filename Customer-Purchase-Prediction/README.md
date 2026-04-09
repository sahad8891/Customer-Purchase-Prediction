# Customer Purchase Prediction System

A machine learning project built with Python to predict whether a customer will purchase a product based on their **Age** and **Estimated Salary**.

## Project Description
This system uses **Logistic Regression**, a popular classification algorithm, to analyze customer data and predict buying behavior. It includes data preprocessing (feature scaling), model training, evaluation, and interactive user input for real-time predictions. The project also generates visual reports of the model's performance.

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For generating insightful data visualizations.
- **Scikit-learn**: For implementing the Machine Learning model and evaluation metrics.

## File Structure
```
Customer-Purchase-Prediction/
│
├── main.py              # Main Python script
├── dataset.csv          # Customer dataset (Age, Salary, Purchased)
├── screenshots/         # Folder containing generated plots
│   ├── prediction_chart.png
│   ├── confusion_matrix.png
│   └── accuracy_graph.png
└── README.md            # Project documentation
```

## How to Run the Project
1. **Ensure Python is installed** on your system.
2. **Install dependencies**:
   Run the following command in your terminal:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. **Execute the script**:
   ```bash
   python main.py
   ```
4. **Follow the prompts**:
   - The script will train the model and save three graphs in the `screenshots/` folder.
   - It will then ask you to enter an **Age** and **Salary** to predict a purchase.

## Output Explanation
- **Accuracy Score**: Indicates how well the model predicts (e.g., 0.85 means 85% accuracy).
- **Confusion Matrix**: Shows the count of True Positives, True Negatives, False Positives, and False Negatives.
- **Classification Report**: Provides detailed metrics like Precision, Recall, and F1-score.
- **Visualizations**:
  - `prediction_chart.png`: Shows the distribution of predicted "Buy" vs "Not Buy" outcomes.
  - `confusion_matrix.png`: A heatmap representing the model's errors and successes.
  - `accuracy_graph.png`: A visual representation of the overall model accuracy.

## Sample Interactive Output
```
Enter Customer Age: 34
Enter Customer Estimated Salary: 80000

Prediction: The customer will likely BUY the product! (Yes)
```
