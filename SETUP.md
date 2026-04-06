# 🚀 Setup & User Guide: House Price Prediction System

This guide will help you set up and run the House Price Prediction project on your computer, even if you have no technical background.

---

## 🌟 Section 1: Non-Technical User Guide (Easiest Method)

If you are unfamiliar with code or terminal commands, follow these 3 simple steps:

### **Step 1: Install Python**
- Go to [python.org/downloads](https://www.python.org/downloads/).
- Click the **"Download Python 3.xx.x"** button.
- **IMPORTANT**: During installation, make sure to check the box that says **"Add Python to PATH"**.

### **Step 2: Automated Setup**
- Locate the project folder on your computer.
- Double-click the file named **`setup.bat`**.
- A black window will open. Just wait for it to finish (it might take 2-5 minutes).
- Once it says "SETUP COMPLETE!", you are ready.

### **Step 3: Launch the Project**
- Double-click the file named **`RUN.bat`**.
- This will automatically start the background server and launch the dashboard in your web browser.
- **That's it!** You can now explore house prices and market trends.

---

## 💻 Section 2: Technical Explanation (How it Works)

For those who want to understand what happens under the hood, here is a detailed breakdown of the setup process.

### **1. Virtual Environments (`venv`)**
- **What it is**: A private "bubble" for this project.
- **Why we use it**: Python projects often need specific versions of libraries (like `pandas` or `xgboost`). A virtual environment ensures this project’s requirements don't conflict with other software on your computer.

### **2. Dependency Installation (`pip`)**
- The `requirements.txt` file is essentially a "shopping list" of every library the code needs to run. 
- The command `pip install -r requirements.txt` tells your computer to download and install all these tools exactly as the project expects.

### **3. The Two-Layer Runtime**
To run this project, two separate "engines" must be active at the same time:
1.  **The API (Backend)**: This is the brain of the project. It loads the AI models and handles the complex math and geocoding. It runs quietly in the background on port `8000`.
2.  **The Dashboard (Frontend)**: This is the visual part you see in your browser. It talks to the API to get predictions and displays the maps and charts. It runs on port `8501`.

---

## 🛠️ Step-by-Step Manual Setup (Advanced Users)

If you prefer to run commands yourself:

1.  **Open your terminal** (PowerShell or Command Prompt).
2.  **Activate the environment**: 
    ```bash
    .\venv\Scripts\activate
    ```
3.  **Start the Backend API**: 
    ```bash
    python src/api/main.py
    ```
4.  **Open a NEW terminal tab** and start the Dashboard:
    ```bash
    .\venv\Scripts\activate
    streamlit run src/dashboard/app.py
    ```

### **Troubleshooting**
- **"Python not found"**: Ensure Python is installed and Added to PATH!
- **"Port already in use"**: Close any previous windows where you were running the project.
- **"ModuleNotFoundError"**: Run `setup.bat` again to ensure all libraries were installed correctly.
