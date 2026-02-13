# 🏠 Roof Project Bid Analyzer

An intelligent document analysis tool for roof project bids that extracts key information from PDFs and images, powered by AI insights.

## Features

- 📄 **PDF Analysis**: Extract text from specification documents and drawings
- 🖼️ **Image OCR**: Process blueprints and photos using EasyOCR
- 📏 **Square Footage Detection**: Automatically identify roofing area measurements
- 🏭 **Manufacturer Identification**: Detect approved roofing manufacturers (20+ brands)
- 🔧 **Material Detection**: Identify roofing types (EPDM, TPO, PVC, shingles, etc.)
- 🤖 **AI-Powered Insights**: Get intelligent analysis and recommendations using OpenAI
- 📊 **Comparison**: Compare multiple documents side-by-side
- ☁️ **Word Cloud**: Visualize roof-related terms
- 🔍 **Search**: Find specific terms across all documents
- 💾 **Export**: Download summary as TXT, PDF, or CSV

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download the spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

4. Configure OpenAI (optional, for AI insights):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

Get your API key from: https://platform.openai.com/api-keys

## Usage

Run the application:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### How to Use

1. **Upload Documents**: Click "Browse files" to upload PDFs or images
2. **View Summary**: Check the aggregated metrics and individual file summaries
3. **Compare**: Review the comparison table across all documents
4. **Visualize**: Explore the word cloud of roof-related terms
5. **Search**: Find specific terms within the extracted text
6. **Export**: Download your analysis results

## Supported File Types

- **PDF**: Specification documents, drawings
- **Images**: JPG, PNG (blueprints, photos)

## What Gets Extracted

- Total square footage
- Approved manufacturers (GAF, CertainTeed, Owens Corning, etc.)
- Roofing materials (membrane, shingles, tiles, etc.)
- Components (insulation, flashing, drainage, etc.)
- Warranty information
- **AI-Powered Insights**: Intelligent analysis, cost considerations, and recommended questions (when OpenAI is configured)

## Technologies Used

- **Streamlit**: Web interface
- **pdfplumber**: PDF text extraction
- **EasyOCR**: Optical character recognition
- **spaCy**: Natural language processing
- **WordCloud**: Text visualization
- **Pandas**: Data handling
- **ReportLab**: PDF generation

## Configuration

Adjust settings in the sidebar:
- Maximum file size limit
- Full text display toggle

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Verify spaCy model is downloaded: `python -m spacy download en_core_web_sm`
3. Check file size limits (default: 10MB)
4. For OCR issues, ensure torch and easyocr are properly installed

## License

MIT License - Feel free to use and modify
