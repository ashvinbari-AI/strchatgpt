import os
import json
import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path
from tqdm import tqdm
import pytesseract
import re
import unicodedata
from collections import defaultdict
import sys

# Set UTF-8 encoding for console output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =====================================================
# CONFIG
# =====================================================

PDF_FILE = "testLOKSATTA.pdf"

POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TXT_OUTPUT = os.path.join(OUTPUT_DIR, "loksatta_complete.txt")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "loksatta_complete.json")

# =====================================================
# FIXED: PROPER MARATHI OCR WITH PIL CONVERSION
# =====================================================

def pil_to_cv2(pil_image):
    """
    Convert PIL Image to OpenCV format
    """
    # Convert PIL to numpy array
    np_image = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV uses BGR)
    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    else:
        cv_image = np_image
    
    return cv_image

def preprocess_for_marathi(pil_image):
    """
    Specialized preprocessing for Marathi newspaper
    """
    # Convert PIL to OpenCV format
    img = pil_to_cv2(pil_image)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Threshold
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def ocr_marathi_page(pil_image):
    """
    OCR specifically for Marathi text
    """
    # Preprocess
    processed = preprocess_for_marathi(pil_image)
    
    # Save debug image for first page
    if not hasattr(ocr_marathi_page, '_debug_saved'):
        debug_path = os.path.join(OUTPUT_DIR, "debug_processed_page.png")
        cv2.imwrite(debug_path, processed)
        print(f"  Debug image saved to: {debug_path}")
        ocr_marathi_page._debug_saved = True
    
    # Use optimal settings for Marathi
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    
    try:
        # Get detailed data
        data = pytesseract.image_to_data(
            processed,
            lang='mar+eng',
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text preserving order
        text_blocks = []
        prev_line = -1
        current_line = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            
            # Handle confidence properly
            try:
                conf = float(data['conf'][i])
            except (ValueError, TypeError):
                conf = 0
            
            if text and conf > 30:
                line_num = data['line_num'][i]
                
                if line_num != prev_line and prev_line != -1:
                    # Save previous line
                    if current_line:
                        text_blocks.append(' '.join(current_line))
                        current_line = []
                
                current_line.append(text)
                prev_line = line_num
        
        # Add last line
        if current_line:
            text_blocks.append(' '.join(current_line))
        
        # Join with newlines
        extracted_text = '\n'.join(text_blocks)
        
        # Clean up the text
        extracted_text = clean_marathi_text(extracted_text)
        
        return extracted_text, 100 if extracted_text else 0
        
    except Exception as e:
        print(f"  OCR error: {e}")
        return "", 0

def clean_marathi_text(text):
    """
    Clean and fix Marathi text
    """
    if not text:
        return text
    
    # Fix common OCR errors for Marathi
    replacements = {
        # Numbers
        '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
        '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
        
        # Remove garbage characters that appear
        '§': '', '€': '', '¢': '', '™': '', '®': '',
        'â': '', '€¢': '', 'Â': '', 'Ã': '', 'Å': '',
        '|': '', '•': '', '●': '', '○': '', '▪': '',
        '♦': '', '♥': '', '♠': '', '†': '', '‡': '',
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove lines that are mostly garbage (less than 10% Devanagari characters)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if not line.strip():
            continue
        
        # Count Devanagari characters (Marathi script range)
        devanagari_count = sum(1 for char in line if '\u0900' <= char <= '\u097F')
        
        # Also count spaces and basic punctuation as valid
        valid_count = devanagari_count + sum(1 for char in line if char in ' .,!?;:"\'')
        
        if len(line) > 0 and (devanagari_count / len(line) > 0.1 or len(line) < 100):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_pdf_fixed(pdf_path, dpi=300):
    """
    Extract text from PDF with proper Marathi support
    """
    # Get total pages
    info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
    total_pages = info["Pages"]
    print(f"Total pages to process: {total_pages}")
    
    all_pages_data = []
    
    # Process each page
    for page_no in tqdm(range(1, total_pages + 1), desc="Processing pages"):
        try:
            # Convert page to image
            page_images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_no,
                last_page=page_no,
                poppler_path=POPPLER_PATH
            )
            
            if not page_images:
                print(f"  Page {page_no}: Failed to convert")
                all_pages_data.append({
                    "page_number": page_no,
                    "confidence": 0,
                    "text": "",
                    "length": 0,
                    "error": "Failed to convert"
                })
                continue
            
            page_image = page_images[0]
            
            # Perform OCR
            text, confidence = ocr_marathi_page(page_image)
            
            # Save page data
            page_data = {
                "page_number": page_no,
                "confidence": confidence,
                "text": text,
                "length": len(text)
            }
            all_pages_data.append(page_data)
            
            # Print progress with sample
            if text and len(text) > 50:
                preview = text[:50].replace('\n', ' ')
                print(f"\n  Page {page_no}: Extracted {len(text)} chars - Preview: {preview}...")
            else:
                print(f"\n  Page {page_no}: No text extracted")
            
            # Save intermediate results every 3 pages
            if page_no % 3 == 0:
                save_intermediate_results(all_pages_data, page_no)
                
        except Exception as e:
            print(f"  Error on page {page_no}: {e}")
            import traceback
            traceback.print_exc()
            all_pages_data.append({
                "page_number": page_no,
                "confidence": 0,
                "text": "",
                "length": 0,
                "error": str(e)
            })
    
    return all_pages_data

def save_intermediate_results(pages_data, up_to_page):
    """
    Save results periodically
    """
    temp_file = os.path.join(OUTPUT_DIR, f"temp_up_to_page_{up_to_page}.json")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 Saved intermediate results up to page {up_to_page}")

def save_final_results(pages_data):
    """
    Save final results
    """
    # Save as text file
    with open(TXT_OUTPUT, "w", encoding="utf-8") as txt_file:
        for page_data in pages_data:
            txt_file.write(f"\n{'='*100}\n")
            txt_file.write(f"PAGE {page_data['page_number']}\n")
            txt_file.write(f"{'='*100}\n\n")
            
            # Write text if exists
            if page_data['text']:
                txt_file.write(page_data['text'])
            else:
                txt_file.write("[No text extracted]")
            
            txt_file.write("\n\n")
    
    # Save as JSON
    final_output = {
        "source_pdf": PDF_FILE,
        "total_pages": len(pages_data),
        "extraction_date": str(np.datetime64('today')),
        "pages": pages_data
    }
    
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Text saved to: {TXT_OUTPUT}")
    print(f"✅ JSON saved to: {JSON_OUTPUT}")

def verify_results(pages_data):
    """
    Verify extraction quality
    """
    print("\n" + "="*60)
    print("EXTRACTION QUALITY REPORT")
    print("="*60)
    
    successful = [p for p in pages_data if p['length'] > 100]
    
    if not successful:
        print("❌ No pages successfully extracted!")
        print("\nDebug info:")
        for p in pages_data[:3]:  # Show first 3 pages
            print(f"  Page {p['page_number']}: length={p['length']}, error={p.get('error', 'None')}")
        return
    
    total_chars = sum(p['length'] for p in successful)
    
    print(f"✅ Successfully extracted: {len(successful)}/{len(pages_data)} pages")
    print(f"📝 Total characters extracted: {total_chars:,}")
    
    # Show samples
    print("\n📄 Sample from first successful page:")
    print("-" * 60)
    first_success = successful[0]
    sample = first_success['text'][:500]
    print(sample)
    print("-" * 60)
    
    # Pages with issues
    empty = [p for p in pages_data if p['length'] == 0]
    if empty:
        print(f"\n⚠️  {len(empty)} pages with no text")

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    
    print("="*60)
    print("MARATHI NEWSPAPER EXTRACTOR (FIXED)")
    print("="*60)
    print(f"PDF: {PDF_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()
    
    # Check if PDF exists
    if not os.path.exists(PDF_FILE):
        print(f"❌ PDF file not found: {PDF_FILE}")
        exit(1)
    
    # Extract text
    print("Starting extraction...")
    pages_data = extract_pdf_fixed(PDF_FILE, dpi=300)
    
    # Save results
    save_final_results(pages_data)
    
    # Verify quality
    verify_results(pages_data)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    
    # Show output file location
    print(f"\n📄 Check the output file: {TXT_OUTPUT}")
    print("💡 Open it with a text editor that supports UTF-8 (Notepad, VS Code, etc.)")