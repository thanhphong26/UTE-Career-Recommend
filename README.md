# UTE-Career-Recommend
## Steps to Run the Virtual Environment (venv)

1. **Create a Virtual Environment**  
    Run the following command to create a virtual environment:
    ```bash
    python -m venv venv
    ```

2. **Activate the Virtual Environment**  
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3. **Install Dependencies**  
    Install the required dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install Tesseract OCR and Poppler (Required for Image-based PDF Processing)**  
    To process PDF files that contain scanned images, install both Tesseract OCR and Poppler:
    - On Windows:
      ```bash
      # Tesseract OCR: Download from https://github.com/UB-Mannheim/tesseract/wiki
      # IMPORTANT: During installation, select "Additional language data (Vietnamese)" to support Vietnamese documents
      # Add Tesseract installation path to your PATH environment variable (e.g., C:\Program Files\Tesseract-OCR)
      
      # Poppler: Download from https://github.com/oschwartz10612/poppler-windows/releases/
      # Extract the zip file to a location like C:\Program Files\poppler
      # Add the bin directory to your PATH (e.g., C:\Program Files\poppler\Library\bin)
      ```
    - On macOS:
      ```bash
      brew install tesseract tesseract-lang poppler
      ```
    - On Linux:
      ```bash
      sudo apt-get install tesseract-ocr tesseract-ocr-vie poppler-utils
      ```
      
    After installation, verify your OCR setup by running:
      ```bash
      # Activate your virtual environment first
      python tools/test_ocr_config.py
      ```
      
    If you see "OCR system status: ✅ Ready", your setup is complete. If not, follow the instructions provided by the test script.

5. **Run the Application**
    ```bash
    uvicorn main:app --reload --port 8000
    ```

6. **Deactivate the Virtual Environment**  
    When done, deactivate the virtual environment with:
    ```bash
    deactivate
    ```