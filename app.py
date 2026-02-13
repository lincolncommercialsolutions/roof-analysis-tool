# Roof Project Bid Analyzer Streamlit App
# Requirements: Run `pip install streamlit pdfplumber easyocr spacy wordcloud matplotlib reportlab pandas pillow python-dotenv openai` before running the app.
# Also, download spacy model: `python -m spacy download en_core_web_sm`
# For OCR on images/drawings/blueprints, easyocr is used (supports multiple languages, but defaults to English).
# This app prioritizes large document analysis by processing PDFs in chunks/pages and handling multiple files concurrently.
# Enhancements:
# - Uses AI (OpenAI) to classify document type (e.g., bid, specification, blueprint).
# - Passes full extracted text to AI for comprehensive understanding and intelligent analysis.
# - Improved prompts for AI to identify document type, avoid mixing info, and provide deeper insights.
# - Ensures per-file isolation to prevent mixing up data across documents.
# - AI analysis now includes document type, full context (up to token limits), and more structured output.

import streamlit as st
import pdfplumber
import easyocr
import spacy
from spacy.matcher import PhraseMatcher
import re
from io import BytesIO
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import base64
import concurrent.futures
import traceback
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Roof Bid Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    # Try to get API key from Streamlit secrets first (for cloud deployment)
    # Then fall back to environment variables (for local deployment)
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    return None

def get_openai_model():
    try:
        return st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    except:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

openai_client = get_openai_client()
openai_model = get_openai_model()

# Load spaCy model with caching
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}\nPlease run: python -m spacy download en_core_web_sm")
        return None

nlp = load_nlp_model()

# Known roofing manufacturers (expandable list used for matching in documents)
roofing_manufacturers = [
    "GAF", "CertainTeed", "Owens Corning", "Firestone", "Carlisle", "Johns Manville",
    "TAMKO", "IKO", "Atlas", "Malarkey", "Polyglass", "Soprema", "Tremco", "Siplast",
    "Derbigum", "EPDM", "TPO", "PVC", "Versico", "GenFlex", "GAF Materials",
    "CentiMark", "Duro-Last", "FiberTite"
]

# Initialize PhraseMatcher for manufacturers
@st.cache_resource
def get_matcher():
    if nlp is None:
        return None
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(manu) for manu in roofing_manufacturers]
    matcher.add("ROOF_MANU", patterns)
    return matcher

matcher = get_matcher()

# Initialize OCR reader with caching
@st.cache_resource
def get_ocr_reader():
    try:
        return easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        st.warning(f"OCR initialization warning: {e}")
        return None

# Function to extract text from PDF (handles large PDFs by page)
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            total_pages = len(pdf.pages)
            progress_bar = st.progress(0)
            for idx, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                progress_bar.progress((idx + 1) / total_pages)
            progress_bar.empty()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF {file.name}: {e}")
        return ""

# Function to perform OCR on image (for blueprints/drawings)
def ocr_image(file):
    try:
        reader = get_ocr_reader()
        if reader is None:
            st.error("OCR reader not available")
            return ""
        img = Image.open(file)
        with st.spinner(f"Performing OCR on {file.name}..."):
            result = reader.readtext(img, detail=0)
        return " ".join(result)
    except Exception as e:
        st.error(f"Error performing OCR on {file.name}: {e}")
        return ""

# Function to filter roof-related content
def filter_roof_content(text):
    if not text or nlp is None:
        return text
    try:
        # Split into sentences/paragraphs
        doc = nlp(text)
        roof_sentences = []
        for sent in doc.sents:
            lower_sent = sent.text.lower()
            if "roof" in lower_sent or "roofing" in lower_sent:
                roof_sentences.append(sent.text)
        return "\n".join(roof_sentences) if roof_sentences else text
    except Exception as e:
        st.warning(f"Error filtering content: {e}. Returning full text.")
        return text

# Function to extract total square feet (looks for patterns like "X sq ft" near roof context)
def extract_sq_ft(roof_text):
    sq_ft_pattern = re.compile(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|square\s*feet|sf)', re.IGNORECASE)
    matches = sq_ft_pattern.findall(roof_text)
    if matches:
        # Clean and sum potential sq ft (assuming multiple areas might be listed)
        total = sum(float(m.replace(",", "")) for m in matches)
        return total, matches
    return None, []

# Function to identify accepted manufacturers
def identify_manufacturers(roof_text):
    if not roof_text or nlp is None or matcher is None:
        return []
    try:
        doc = nlp(roof_text)
        matches = matcher(doc)
        found_manus = set()
        for match_id, start, end in matches:
            found_manus.add(doc[start:end].text)
        # Look for context like "accepted", "approved", "specified"
        context_pattern = re.compile(r'(accepted|approved|specified|equivalent|or equal)\s*(manufacturers?|brands?):\s*(.+)', re.IGNORECASE)
        context_matches = context_pattern.findall(roof_text)
        if context_matches:
            for _, _, manus in context_matches:
                for manu in roofing_manufacturers:
                    if manu.lower() in manus.lower():
                        found_manus.add(manu)
        return sorted(list(found_manus))
    except Exception as e:
        st.warning(f"Error identifying manufacturers: {e}")
        return []

# Comprehensive roofing keywords for better extraction
MATERIALS_KEYWORDS = [
    "epdm", "tpo", "pvc", "shingle", "shingles", "tile", "tiles", "metal", "asphalt",
    "membrane", "modified bitumen", "sbs", "app", "built-up", "bur", "polyiso"
]
COMPONENTS_KEYWORDS = [
    "insulation", "deck", "flashing", "drainage", "underlayment", "penetration",
    "valley", "ridge", "gutter", "coverboard", "gypsum", "securock", "parapet",
    "drain", "cricket", "taper", "substrate", "eave", "fascia", "coping",
    "termination bar", "sealant", "adhesive", "base sheet", "cap sheet"
]
OTHER_KEYWORDS = [
    "slope", "warranty", "guarantee", "workmanship", "wind uplift", "hail resistance",
    "mechanically attached", "fully adhered", "ballasted", "torch applied", "self-adhered"
]

# Preferred words for word cloud visualization
WORDCLOUD_PREFERRED_WORDS = {
    "roof", "roofing", "membrane", "tpo", "epdm", "pvc", "shingle", "insulation",
    "polyiso", "flashing", "drain", "warranty", "gaf", "firestone", "carlisle",
    "soprema", "square feet", "sq ft", "underlayment", "penetration", "parapet",
    "adhered", "attached", "torch", "sbs", "modified bitumen", "drainage",
    "coverboard", "deck", "slope", "valley", "ridge", "gutter", "certainteed",
    "owens corning", "johns manville", "tremco", "iko", "atlas", "malarkey"
}

# Function to extract materials, components, and other roof-related info
def extract_roof_details(roof_text):
    if not roof_text or nlp is None:
        return {"materials": [], "components": [], "other": []}
    try:
        doc = nlp(roof_text)
        materials = set()
        components = set()
        other = set()
        # Check both individual tokens and phrases
        text_lower = roof_text.lower()
        for token in doc:
            lower_token = token.text.lower()
            if lower_token in MATERIALS_KEYWORDS:
                materials.add(lower_token)
            elif lower_token in COMPONENTS_KEYWORDS:
                components.add(lower_token)
            elif lower_token in OTHER_KEYWORDS:
                other.add(lower_token)
        # Check for multi-word keywords
        for keyword in MATERIALS_KEYWORDS + COMPONENTS_KEYWORDS + OTHER_KEYWORDS:
            if " " in keyword and keyword in text_lower:
                if keyword in MATERIALS_KEYWORDS:
                    materials.add(keyword)
                elif keyword in COMPONENTS_KEYWORDS:
                    components.add(keyword)
                elif keyword in OTHER_KEYWORDS:
                    other.add(keyword)
        return {
            "materials": sorted(list(materials)),
            "components": sorted(list(components)),
            "other": sorted(list(other))
        }
    except Exception as e:
        st.warning(f"Error extracting roof details: {e}")
        return {"materials": [], "components": [], "other": []}

# Function to extract warranty-related content
def extract_warranty_info(roof_text):
    if not roof_text or nlp is None:
        return ""
    try:
        doc = nlp(roof_text)
        warranty_sentences = []
        for sent in doc.sents:
            lower_sent = sent.text.lower()
            if "warranty" in lower_sent or "guarantee" in lower_sent:
                warranty_sentences.append(sent.text)
        return "\n".join(warranty_sentences) if warranty_sentences else "No warranty information found."
    except Exception as e:
        st.warning(f"Error extracting warranty info: {e}")
        return ""

# Function to get document type using OpenAI
def get_document_type(full_text):
    """Use OpenAI to classify the document type."""
    if not openai_client or not full_text:
        return "Unknown", "AI not available"
    try:
        context = f"""Classify this document based on its content:
Document excerpt: {full_text[:4000]}

Possible types: Bid Proposal, Specification Document, Blueprint/Drawing, Contract, Invoice, Report, or Other.
Provide the type and a brief reason."""
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a document classification expert."},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=100
        )
        analysis = response.choices[0].message.content.strip()
        doc_type = analysis.split("\n")[0].replace("Type: ", "").strip()
        reason = analysis.split("\n")[1] if "\n" in analysis else ""
        return doc_type, reason
    except Exception as e:
        st.warning(f"Document type classification unavailable: {e}")
        return "Unknown", str(e)

# Function to get AI-enhanced analysis using OpenAI (uses full text for better understanding)
def get_ai_analysis(full_text, roof_text, sq_ft, manufacturers, roof_details, doc_type):
    """Use OpenAI to provide intelligent insights on the entire document."""
    if not openai_client or not full_text:
        return None
    try:
        # Prepare context for AI with full text (truncated to avoid token limits)
        context = f"""You are analyzing a roof project document.
Document Type: {doc_type}
Full Document Text (excerpt): {full_text[:8000]}  # Increased limit for better context
Roof-Specific Excerpt: {roof_text[:2000]}

Extracted Info:
Square Footage: {sq_ft if sq_ft else 'Not specified'}
Manufacturers: {', '.join(manufacturers) if manufacturers else 'None'}
Materials: {', '.join(roof_details.get('materials', [])) if roof_details.get('materials') else 'None'}
Components: {', '.join(roof_details.get('components', [])) if roof_details.get('components') else 'None'}

Provide a structured analysis:
1. Document Type Confirmation and Summary: Confirm type and give a 2-3 sentence overview.
2. Key Highlights: Important details from the entire document.
3. Potential Concerns/Missing Info: Any red flags or gaps.
4. Cost Considerations: Estimates or factors based on specs (high-level, no real numbers unless in doc).
5. Recommended Questions: 3-5 questions to ask contractor/owner.
6. Overall Assessment: Quality rating (e.g., High/Medium/Low) with reason."""
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are an expert roof construction analyst specializing in commercial and residential roofing projects. Provide concise, actionable insights based on the full document context to avoid mixing information."},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"AI analysis unavailable: {e}")
        return None

# Function to generate word cloud
def generate_wordcloud(text):
    try:
        # Custom stopwords to exclude common non-roofing words
        from wordcloud import STOPWORDS
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(['will', 'shall', 'must', 'may', 'per', 'including', 'section'])
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=80,
            prefer_horizontal=0.9,
            collocations=True,
            stopwords=custom_stopwords,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)
        return fig
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        return None

# Function to process a single file (now includes full_text and document_type)
def process_file(file, file_type):
    try:
        if file_type == "pdf":
            full_text = extract_text_from_pdf(file)
        else:  # image
            full_text = ocr_image(file)
        if not full_text:
            return {
                "file_name": file.name,
                "error": "No text extracted",
                "full_text": "",
                "roof_text": "",
                "document_type": "Unknown",
                "document_type_reason": "No text",
                "sq_ft_total": None,
                "sq_ft_matches": [],
                "manufacturers": [],
                "details": {"materials": [], "components": [], "other": []},
                "warranty_info": ""
            }
        roof_text = filter_roof_content(full_text)
        document_type, type_reason = get_document_type(full_text)
        sq_ft_total, sq_ft_matches = extract_sq_ft(roof_text)
        manufacturers = identify_manufacturers(roof_text)
        details = extract_roof_details(roof_text)
        warranty_info = extract_warranty_info(roof_text)
        return {
            "file_name": file.name,
            "full_text": full_text,
            "roof_text": roof_text,
            "document_type": document_type,
            "document_type_reason": type_reason,
            "sq_ft_total": sq_ft_total,
            "sq_ft_matches": sq_ft_matches,
            "manufacturers": manufacturers,
            "details": details,
            "warranty_info": warranty_info
        }
    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")
        traceback.print_exc()
        return {
            "file_name": file.name,
            "error": str(e),
            "full_text": "",
            "roof_text": "",
            "document_type": "Unknown",
            "document_type_reason": str(e),
            "sq_ft_total": None,
            "sq_ft_matches": [],
            "manufacturers": [],
            "details": {"materials": [], "components": [], "other": []},
            "warranty_info": ""
        }

# Function to generate PDF export
def generate_pdf(summary_text):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        for line in summary_text.split("\n"):
            if line.strip():
                # Properly handle markdown bold syntax
                formatted_line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
                # Escape any remaining special characters
                formatted_line = formatted_line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Re-apply the bold tags after escaping
                formatted_line = formatted_line.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
                try:
                    para = Paragraph(formatted_line, styles['Normal'])
                    story.append(para)
                except:
                    # Fallback to plain text if formatting fails
                    plain_text = line.replace('**', '')
                    para = Paragraph(plain_text, styles['Normal'])
                    story.append(para)
            story.append(Spacer(1, 12))
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        # Fallback to simple canvas method
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        y = height - 50
        for line in summary_text.split("\n"):
            if y < 50:
                c.showPage()
                y = height - 50
            if line.strip():
                c.drawString(50, y, line[:100])  # Truncate long lines
            y -= 15
        c.save()
        buffer.seek(0)
        return buffer

# Main app
st.title("🏠 Roof Project Bid Analyzer")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    max_file_size = st.number_input("Max file size (MB)", min_value=1, max_value=100, value=10)
    show_full_text = st.checkbox("Show full extracted text", value=False)
    st.markdown("---")
    st.markdown("### 🤖 AI Configuration")
    if openai_client:
        st.success(f"✅ OpenAI Connected ({openai_model})")
    else:
        st.warning("⚠️ OpenAI Not Configured")
        with st.expander("Setup Instructions"):
            st.markdown("""
1. Create a `.env` file in the project root
2. Add your OpenAI API key: