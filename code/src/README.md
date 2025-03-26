# GenAI Orchestrator for Email and Document Triage

## Overview
- Generative AI (GenAI)-powered financial Email & document processing system.
- Extracts key financial data, classifies requests, and detects duplicate content.
- Utilizes **Mistral-7B** for NLP-based text extraction and classification.
- Runs in a **Spyder IDE** environment.

## Features
- **Multi-Format Document Processing**:
  - Supports PDFs, DOCX, and EML files.
- **GenAI-Powered Data Extraction**:
  - Uses Mistral-7B with 4-bit quantization for optimized GPU performance.
- **Advanced Classification**:
  - Identifies request types and sub-types based on extracted data fields.
- **Duplicate Detection**:
  - Compares extracted field similarity to identify and flag duplicate documents.
- **Structured Output**:
  - Outputs data in structured JSON format for easy integration with databases.

## Requirements
- **Python 3.8+** installed.
- **Spyder IDE** (recommended) or any other Python environment.
- **Required Python Libraries** (install using `requirements.txt`).
- **CUDA-enabled GPU** for optimized model performance.

## Installation
1. Clone the repository or download the source code.
2. Navigate to the project directory:
   ```bash
   cd GenAI_FinanceData_Extractor
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use
1. Place your financial documents in the `Input/` folder.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Extracted data will be saved in the `Output/` folder in JSON format.

## Folder Structure
```
/GenAI_FinanceData_Extractor
│-- main.py  # Primary execution script
│-- requirements.txt  # List of required Python libraries
│-- Input/  # Folder containing input financial documents
│-- Output/  # Processed JSON results
│-- README.md  # Project documentation
```

## Customization
- **Modify classification rules**:
  - Adjust `EXPECTED_FIELDS_MAP` to change field mappings.
- **Tune duplicate detection thresholds**:
  - Modify `is_duplicate()` to refine duplicate detection sensitivity.
- **Adjust model parameters**:
  - Optimize Mistral-7B settings in `extract_financial_entities()`.

## Contact
- Reach out via **GitHub Issues** for bug reports or enhancements.
- Email the development team for further support.
