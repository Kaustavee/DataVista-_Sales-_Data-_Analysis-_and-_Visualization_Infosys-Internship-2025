# DataVista-_Sales-_Data-_Analysis-_and-_Visualization_Infosys-Internship-2025
# DataVista-Sales-Data-Analysis-and-Visualization---Infosys-Internship-2025



# Energy Consumption Web Application

This web application visualizes and predicts energy consumption data. It allows users to:

* **Visualize historical energy consumption:** View energy consumption by fuel type for different countries over the years.
* **Predict future energy consumption:** Use a machine learning model to predict energy consumption for a selected country in a future year.

##   Features

* **Interactive visualizations:** Displays energy consumption data using Matplotlib, including line charts and bar charts.
* **Future energy prediction:** Predicts energy consumption for a selected country and year using a Random Forest Regressor model.
* **Flask web framework:** Built using Flask to handle routing, form submissions, and rendering of HTML templates.
* **Data handling:** Reads energy consumption data from CSV files using Pandas.
* **Virtual environment:** Uses a virtual environment to manage project dependencies.

##   Dependencies

The project requires the following Python libraries, which are listed in `requirements.txt`:

* `numpy`
* `matplotlib`
* `pandas`
* `scikit-learn`
* `Flask`
* `Flask-Caching`

##   Setup Instructions

Follow these steps to set up the project:

###   1. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to isolate project dependencies.

* **Using `venv` (Python 3.3+):**

    \`\`\`bash
    python -m venv venv
    \`\`\`

* **Activate the virtual environment:**
    * **On Windows:**

        \`\`\`bash
        venv\\Scripts\\activate
        \`\`\`

    * **On macOS and Linux:**

        \`\`\`bash
        source venv/bin/activate
        \`\`\`

###   2. Install Dependencies

Install the required Python packages from the `requirements.txt` file:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Download the Dataset

The application requires the following CSV files:

* `Energy_consumption_by_fuel_EJ.csv`
* `Primary energy consumption - EJ.csv`

Make sure these files are located in the same directory as the main application script (`app.py`). If the application is run and the files are not in the correct location you will receive this error: "Error: Energy_consumption_by_fuel_EJ.csv not found."

###   4. Run the Application

Run the Flask application:

\`\`\`bash
python app.py
\`\`\`

The application will start, and you can access it in your web browser (usually at `http://127.0.0.1:5000/`).

##   Usage

The application has two main sections:

###   1. Visualizations

* **Home Page (`/`):**
    * Select a country to visualize its energy consumption by fuel type over the years.
    * Select a country and year to see a bar chart of energy consumption by fuel type for that specific year.

###   2. Predictions

* **Predictions Page (`/predict`):**
    * Select a country and a future year (after 2023) to predict its total energy consumption for that year.
    * The application will display the predicted energy consumption and show a plot.
    * The application also shows a comparison bar chart of predictions for all countries for the selected future year.
