# Roof Project Bid Analyzer Streamlit App
# Requirements: Run `pip install streamlit pdfplumber easyocr spacy wordcloud matplotlib reportlab pillow` before running the app.
# Also, download spacy model: `python -m spacy download en_core_web_sm`
# For OCR on images/drawings/blueprints, easyocr is used (supports multiple languages, but defaults to English).
# This app prioritizes large document analysis by processing PDFs in chunks/pages and handling multiple files concurrently.
# Additional features:
# - Handles both PDFs (specs/drawings) and images (blueprints/photos).
# - Keyword-based filtering for roof-related content only.
# - NLP (spaCy) for entity recognition to identify manufacturers, measurements, etc.
# - Comprehensive summary with extracted sq ft, manufacturers, and other roof info.
# - Visualization: Word cloud of roof-related terms.
# - Comparison across multiple documents.
# - Export summary to TXT or PDF.
# - Search function within extracted text.
# - Handles large docs by streaming page-by-page to avoid memory issues.

# Roof Project Bid Analyzer Streamlit App
# Requirements: Run `pip install streamlit pdfplumber easyocr spacy wordcloud matplotlib reportlab pandas` before running the app.
# Also, download spacy model: `python -m spacy download en_core_web_sm`
# For OCR on images/drawings/blueprints, easyocr is used (supports multiple languages, but defaults to English).
# This app prioritizes large document analysis by processing PDFs in chunks/pages and handling multiple files concurrently.
# Additional features:
# - Handles both PDFs (specs/drawings) and images (blueprints/photos).
# - Keyword-based filtering for roof-related content only.
# - NLP (spaCy) for entity recognition to identify manufacturers, measurements, etc.
# - Comprehensive summary with extracted sq ft, manufacturers, materials, components, warranty info, and other roof info.
# - Visualization: Word cloud of roof-related terms.
# - Comparison across multiple documents.
# - Export summary to TXT, PDF, or CSV.
# - Search function within extracted text.
# - Handles large docs by streaming page-by-page to avoid memory issues.
# - Extracts warranty-related content specifically.

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

# Known roofing manufacturers (expandable list - excludes material types like EPDM/TPO/PVC)
roofing_manufacturers = [
    "GAF", "GAF Materials", "EverGuard",
    "CertainTeed",
    "Owens Corning",
    "Firestone", "Firestone Building Products", "Red Shield",
    "Carlisle", "Carlisle SynTec", "Sure-Seal", "SecurShield",
    "Johns Manville", "JM",
    "TAMKO", "IKO", "Atlas", "Malarkey",
    "Polyglass",
    "Soprema", "Sopralene",
    "Tremco", "Siplast",
    "Derbigum",
    "Versico",
    "GenFlex",
    "CentiMark",
    "Duro-Last",
    "FiberTite",
    "Sarnafil",
    "Mule-Hide"
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
        found_manus = set()
        
        # Strategy 1: Look for "ACCEPTED MANUFACTURERS" section and extract all following lines
        # that match manufacturer patterns
        section_start = re.search(
            r'(?:accepted|approved|specified)\s+(?:roofing\s+)?manufacturers?',
            roof_text,
            re.IGNORECASE
        )
        
        if section_start:
            # Get text from section start onwards
            section_text = roof_text[section_start.end():]
            # Take up to 1500 characters or until we hit certain section markers
            section_end = re.search(r'\n\s*(?:PART|SECTION|ARTICLE|EXECUTION|SUBMITTALS|PRODUCTS|[A-Z\s]{20,})', section_text)
            if section_end:
                section_text = section_text[:section_end.start()]
            else:
                section_text = section_text[:1500]
            
            # Split by newlines to get individual manufacturer entries
            lines = section_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Check each known manufacturer
                for manu in roofing_manufacturers:
                    # Case-insensitive match
                    if re.search(r'\b' + re.escape(manu) + r'\b', line, re.IGNORECASE):
                        # Extract the full manufacturer entry (up to — or - for product details)
                        if '—' in line or ' - ' in line:
                            # Split by em dash or regular dash with spaces
                            parts = re.split(r'\s*[—]\s*|\s+-\s+', line, maxsplit=1)
                            if len(parts) > 1:
                                manu_name = parts[0].strip()
                                product_info = parts[1].strip()
                                # Clean up and limit length
                                product_info = re.sub(r'\s+', ' ', product_info)[:150]
                                found_manus.add(f"{manu_name} — {product_info}")
                            else:
                                found_manus.add(line.strip())
                        else:
                            found_manus.add(line.strip())
                        break
        
        # Strategy 2: Use PhraseMatcher for general detection (fallback)
        if not found_manus:
            doc = nlp(roof_text)
            matches = matcher(doc)
            for match_id, start, end in matches:
                manu_name = doc[start:end].text
                found_manus.add(manu_name)
        
        # Strategy 3: Context-based extraction for simple comma-separated lists (fallback)
        if not found_manus:
            context_pattern = re.compile(
                r'(?:accepted|approved|specified|equivalent|or equal)\s*(?:roofing\s+)?(?:manufacturers?|brands?)\s*[:\s]+(.+?)(?:\.|\n\n|$)',
                re.IGNORECASE | re.DOTALL
            )
            context_matches = context_pattern.findall(roof_text)
            for match_text in context_matches:
                for manu in roofing_manufacturers:
                    if re.search(r'\b' + re.escape(manu) + r'\b', match_text, re.IGNORECASE):
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

# Function to get AI-enhanced analysis using OpenAI
def get_ai_analysis(roof_text, sq_ft, manufacturers, roof_details):
    """Use OpenAI to provide intelligent insights on the roof project."""
    if not openai_client or not roof_text:
        return None
    
    try:
        # Prepare context for AI
        context = f"""Analyze this roof project bid document:

Square Footage: {sq_ft if sq_ft else 'Not specified'}
Manufacturers Found: {', '.join(manufacturers) if manufacturers else 'None'}
Materials: {', '.join(roof_details.get('materials', [])) if roof_details.get('materials') else 'None'}
Components: {', '.join(roof_details.get('components', [])) if roof_details.get('components') else 'None'}

Document Text (excerpt):
{roof_text[:3000]}

Please provide:
1. Key highlights and important details
2. Potential concerns or missing information
3. Cost considerations based on the specifications
4. Recommended questions to ask the contractor
5. Overall assessment of the bid quality"""

        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are an expert roof construction analyst specializing in commercial and residential roofing projects. Provide concise, actionable insights."},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
            max_tokens=1000
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

# Function to process a single file
def process_file(file, file_type):
    try:
        if file_type == "pdf":
            text = extract_text_from_pdf(file)
        else:  # image
            text = ocr_image(file)
        if not text:
            return {
                "file_name": file.name,
                "error": "No text extracted",
                "roof_text": "",
                "sq_ft_total": None,
                "sq_ft_matches": [],
                "manufacturers": [],
                "details": {"materials": [], "components": [], "other": []},
                "warranty_info": ""
            }
        roof_text = filter_roof_content(text)
        sq_ft_total, sq_ft_matches = extract_sq_ft(roof_text)
        manufacturers = identify_manufacturers(roof_text)
        details = extract_roof_details(roof_text)
        warranty_info = extract_warranty_info(roof_text)
        return {
            "file_name": file.name,
            "roof_text": roof_text,
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
            "roof_text": "",
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
            ```
            OPENAI_API_KEY=sk-your-key-here
            OPENAI_MODEL=gpt-4o-mini
            ```
            3. Restart the application
            """)
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This tool analyzes roof project bid documents to extract roof-related information from uploaded files. "
        "With AI integration, get intelligent insights and recommendations."
    )

st.markdown("""
Upload multiple bid documents (PDFs for specs/drawings, images for blueprints). 
The app will analyze and summarize only roof-related information.
""")

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents",
    type=['pdf', 'png', 'jpg', 'jpeg'],
    accept_multiple_files=True,
    help="Upload PDFs or images (blueprints, specifications, drawings)"
)

if uploaded_files:
    # Check file sizes
    max_size_bytes = max_file_size * 1024 * 1024
    valid_files = []
    for file in uploaded_files:
        file_size = file.size
        if file_size > max_size_bytes:
            st.warning(f"⚠️ {file.name} exceeds {max_file_size}MB limit (size: {file_size / 1024 / 1024:.2f}MB)")
        else:
            valid_files.append(file)
    if not valid_files:
        st.error("No valid files to process!")
        st.stop()
    st.success(f"✅ Processing {len(valid_files)} file(s)...")
    
    # Process files concurrently for efficiency (handles large/multiple docs)
    results = []
    with st.spinner("Analyzing documents..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for file in valid_files:
                file_type = "pdf" if file.type == "application/pdf" else "image"
                futures.append(executor.submit(process_file, file, file_type))
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    
    # Comprehensive summary
    all_sq_ft = 0
    all_manufacturers = set()
    all_materials = set()
    all_components = set()
    all_other = set()
    summaries = []
    for result in results:
        summary = f"**File: {result['file_name']}**\n"
        if result.get('error'):
            summary += f"❌ Error: {result['error']}\n"
        else:
            if result['sq_ft_total']:
                summary += f"📏 Total Roofing Square Feet: {result['sq_ft_total']:,.2f} (from matches: {', '.join(result['sq_ft_matches'])})\n"
                all_sq_ft += result['sq_ft_total']
            if result['manufacturers']:
                summary += f"🏭 Accepted Roofing Manufacturers: {', '.join(result['manufacturers'])}\n"
                all_manufacturers.update(result['manufacturers'])
            if result['details']['materials']:
                summary += f"🛠 Materials: {', '.join(result['details']['materials'])}\n"
                all_materials.update(result['details']['materials'])
            if result['details']['components']:
                summary += f"🔩 Components: {', '.join(result['details']['components'])}\n"
                all_components.update(result['details']['components'])
            if result['details']['other']:
                summary += f"ℹ️ Other Info: {', '.join(result['details']['other'])}\n"
                all_other.update(result['details']['other'])
            if result['warranty_info']:
                summary += f"📜 Warranty Information: {result['warranty_info'][:500]}{'...' if len(result['warranty_info']) > 500 else ''}\n"
            if show_full_text and result['roof_text']:
                summary += f"\n📄 Extracted Roof Text:\n{result['roof_text'][:2000]}{'...' if len(result['roof_text']) > 2000 else ''}\n"
        summaries.append(summary)
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Summary", "🤖 AI Insights", "💬 Chatbot", "📋 Comparison", "☁️ Word Cloud", "🔍 Search", "💾 Export"])
    
    with tab1:
        st.header("Comprehensive Roof Summary")
        # Aggregated across all files
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Square Feet", f"{all_sq_ft:,.2f}" if all_sq_ft > 0 else "N/A")
        with col2:
            st.metric("Manufacturers Found", len(all_manufacturers))
        with col3:
            st.metric("Files Processed", len(results))
        st.markdown("---")
        if all_manufacturers:
            st.subheader("🏭 Accepted Roofing Manufacturers (All Files)")
            st.write(", ".join(sorted(all_manufacturers)))
        if all_materials:
            st.subheader("🛠 Materials (All Files)")
            st.write(", ".join(sorted(all_materials)))
        if all_components:
            st.subheader("🔩 Components (All Files)")
            st.write(", ".join(sorted(all_components)))
        if all_other:
            st.subheader("ℹ️ Other Info (All Files)")
            st.write(", ".join(sorted(all_other)))
        st.markdown("---")
        # Display individual summaries
        st.subheader("Individual File Summaries")
        for idx, result in enumerate(results):
            with st.container():
                st.markdown(f"### 📄 {result['file_name']}")
                
                if result.get('error'):
                    st.error(f"❌ Error: {result['error']}")
                else:
                    # Create metrics row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if result['sq_ft_total']:
                            st.metric("Square Footage", f"{result['sq_ft_total']:,.2f}")
                    with col2:
                        st.metric("Manufacturers", len(result['manufacturers']))
                    with col3:
                        st.metric("Materials", len(result['details']['materials']))
                    
                    # Details in expandable sections
                    with st.expander("🔍 Detailed Information", expanded=False):
                        if result['sq_ft_total']:
                            st.write(f"**📏 Total Square Feet:** {result['sq_ft_total']:,.2f}")
                            if result['sq_ft_matches']:
                                st.write(f"*Found in: {', '.join(result['sq_ft_matches'])}*")
                        
                        if result['manufacturers']:
                            st.write(f"**🏭 Manufacturers:** {', '.join(result['manufacturers'])}")
                        
                        if result['details']['materials']:
                            st.write(f"**🛠 Materials:** {', '.join(result['details']['materials'])}")
                        
                        if result['details']['components']:
                            st.write(f"**🔩 Components:** {', '.join(result['details']['components'])}")
                        
                        if result['details']['other']:
                            st.write(f"**ℹ️ Other:** {', '.join(result['details']['other'])}")
                        
                        if result['warranty_info'] and result['warranty_info'] != "No warranty information found.":
                            st.write(f"**📜 Warranty:** {result['warranty_info'][:300]}{'...' if len(result['warranty_info']) > 300 else ''}")
                    
                    if show_full_text and result['roof_text']:
                        with st.expander("📄 Full Extracted Text"):
                            st.text_area("Roof-related content", result['roof_text'], height=200, key=f"text_{idx}")
                
                st.markdown("---")
    
    with tab2:
        st.header("🤖 AI-Powered Insights")
        
        if not openai_client:
            st.warning("⚠️ OpenAI API key not configured. Please add your API key to the .env file to use AI insights.")
            st.code("OPENAI_API_KEY=your-api-key-here", language="bash")
        else:
            st.info("💡 AI insights powered by " + openai_model)
            
            # Generate AI analysis for each document
            for idx, result in enumerate(results):
                if result.get('error'):
                    continue
                    
                with st.expander(f"🔍 AI Analysis: {result['file_name']}", expanded=(idx == 0)):
                    with st.spinner("Generating AI insights..."):
                        ai_insights = get_ai_analysis(
                            result['roof_text'],
                            result['sq_ft_total'],
                            result['manufacturers'],
                            result['details']
                        )
                        
                        if ai_insights:
                            st.markdown(ai_insights)
                        else:
                            st.warning("Unable to generate AI insights for this document.")
    
    with tab3:
        st.header("💬 Document Chatbot")
        
        if not openai_client:
            st.warning("⚠️ OpenAI API key not configured. Chatbot requires API access.")
            st.code("OPENAI_API_KEY=your-api-key-here", language="bash")
        else:
            st.info("Ask questions about your uploaded roof bid documents. Select a document and start chatting!")
            
            # Document selector
            if results:
                selected_doc = st.selectbox(
                    "Select a document to chat about:",
                    options=range(len(results)),
                    format_func=lambda x: results[x]['file_name']
                )
                
                doc_data = results[selected_doc]
                
                # Show quick summary of selected document
                with st.expander("📋 Quick Summary", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        if doc_data['sq_ft_total']:
                            st.write(f"**Square Feet:** {doc_data['sq_ft_total']:,.2f}")
                        if doc_data['manufacturers']:
                            st.write(f"**Manufacturers:** {', '.join(doc_data['manufacturers'])}")
                    with col2:
                        if doc_data['details']['materials']:
                            st.write(f"**Materials:** {', '.join(doc_data['details']['materials'])}")
                        if doc_data['details']['components']:
                            st.write(f"**Components:** {', '.join(doc_data['details']['components'])}")
                
                # Initialize chat history in session state
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = {}
                
                if selected_doc not in st.session_state.chat_history:
                    st.session_state.chat_history[selected_doc] = []
                
                # Display chat history
                for message in st.session_state.chat_history[selected_doc]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input
                if prompt := st.chat_input(f"Ask anything about {doc_data['file_name']}..."):
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Add to history
                    st.session_state.chat_history[selected_doc].append({"role": "user", "content": prompt})
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                # Prepare context for the chatbot
                                context = f"""Document: {doc_data['file_name']}

Square Footage: {doc_data['sq_ft_total'] if doc_data['sq_ft_total'] else 'Not specified'}
Manufacturers: {', '.join(doc_data['manufacturers']) if doc_data['manufacturers'] else 'None found'}
Materials: {', '.join(doc_data['details']['materials']) if doc_data['details']['materials'] else 'None found'}
Components: {', '.join(doc_data['details']['components']) if doc_data['details']['components'] else 'None found'}

Document Text (excerpt):
{doc_data['roof_text'][:4000]}"""
                                
                                # Build messages for API
                                messages = [
                                    {"role": "system", "content": f"You are an expert roof construction analyst. You have access to a roofing bid document and should answer questions about it accurately and helpfully. Base your answers on the document content provided.\n\n{context}"},
                                ]
                                
                                # Add chat history (last 5 exchanges to keep context manageable)
                                for msg in st.session_state.chat_history[selected_doc][-10:]:
                                    messages.append({"role": msg["role"], "content": msg["content"]})
                                
                                response = openai_client.chat.completions.create(
                                    model=openai_model,
                                    messages=messages,
                                    temperature=0.7,
                                    max_tokens=800,
                                    stream=True
                                )
                                
                                # Stream the response
                                response_text = st.write_stream(response)
                                
                                # Add to history
                                st.session_state.chat_history[selected_doc].append({"role": "assistant", "content": response_text})
                                
                            except Exception as e:
                                error_msg = f"Error generating response: {e}"
                                st.error(error_msg)
                                st.session_state.chat_history[selected_doc].append({"role": "assistant", "content": error_msg})
                
                # Clear chat button
                if st.session_state.chat_history[selected_doc]:
                    if st.button("🗑️ Clear Chat History"):
                        st.session_state.chat_history[selected_doc] = []
                        st.rerun()
    
    with tab4:
        st.header("📋 Comparison Across Documents")
        comparison_data = {
            "File": [r['file_name'] for r in results],
            "Sq Ft": [f"{r['sq_ft_total']:,.2f}" if r['sq_ft_total'] else "N/A" for r in results],
            "Manufacturers": [", ".join(r['manufacturers']) or "N/A" for r in results],
            "Materials": [", ".join(r['details']['materials']) or "N/A" for r in results],
            "Components": [", ".join(r['details']['components']) or "N/A" for r in results],
            "Warranty": ["Yes" if r['warranty_info'] else "No" for r in results]
        }
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        # Download comparison as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison (CSV)",
            data=csv,
            file_name="roof_comparison.csv",
            mime="text/csv"
        )
    
    with tab5:
        st.header("☁️ Roof-Related Word Cloud")
        agg_roof_text = " ".join([r['roof_text'] for r in results if r['roof_text']])
        if agg_roof_text and len(agg_roof_text.strip()) > 0:
            fig = generate_wordcloud(agg_roof_text)
            if fig:
                st.pyplot(fig)
        else:
            st.warning("No text available to generate word cloud")
    
    with tab6:
        st.header("🔍 Search Within Extracted Roof Text")
        search_query = st.text_input("Enter search term (e.g., 'warranty', 'GAF', 'TPO')")
        if search_query:
            found_results = False
            for result in results:
                if result['roof_text'] and search_query.lower() in result['roof_text'].lower():
                    found_results = True
                    st.subheader(f"📄 {result['file_name']}")
                    # Highlight search term
                    highlighted = re.sub(
                        f"({re.escape(search_query)})",
                        r"**\1**",
                        result['roof_text'][:2000],
                        flags=re.IGNORECASE
                    )
                    st.markdown(highlighted + ("..." if len(result['roof_text']) > 2000 else ""))
                    st.markdown("---")
            if not found_results:
                st.info(f"No results found for '{search_query}'")
    
    with tab7:
        st.header("💾 Export Summary")
        # Prepare full summary text
        agg_summary = "ROOF PROJECT BID ANALYSIS SUMMARY\n"
        agg_summary += "=" * 50 + "\n\n"
        agg_summary += f"Total Files Analyzed: {len(results)}\n"
        if all_sq_ft > 0:
            agg_summary += f"Total Roofing Square Feet: {all_sq_ft:,.2f}\n"
        if all_manufacturers:
            agg_summary += f"Accepted Roofing Manufacturers: {', '.join(sorted(all_manufacturers))}\n"
        if all_materials:
            agg_summary += f"Materials: {', '.join(sorted(all_materials))}\n"
        if all_components:
            agg_summary += f"Components: {', '.join(sorted(all_components))}\n"
        if all_other:
            agg_summary += f"Other Info: {', '.join(sorted(all_other))}\n"
        agg_summary += "\n" + "-" * 50 + "\n\n"
        full_summary_text = agg_summary + "\n".join(summaries)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download as TXT",
                data=full_summary_text,
                file_name="roof_summary.txt",
                mime="text/plain"
            )
        with col2:
            pdf_buffer = generate_pdf(full_summary_text)
            if pdf_buffer:
                st.download_button(
                    label="📑 Download as PDF",
                    data=pdf_buffer,
                    file_name="roof_summary.pdf",
                    mime="application/pdf"
                )
        st.markdown("---")
        st.subheader("Preview")
        st.text_area("Summary Preview", full_summary_text, height=300)
else:
    # Show instructions when no files uploaded
    st.info("👆 Upload PDF or image files to begin analysis")

# Footer
st.markdown("---")
st.markdown("*Developed for roof project bid analysis | Powered by Streamlit, spaCy & EasyOCR*")