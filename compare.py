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
from datetime import datetime

# Set UTF-8 encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =====================================================
# CONFIG
# =====================================================

PDF_FILE = "Loksatta_Nagpur_20260608.pdf"

POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TXT_OUTPUT = os.path.join(OUTPUT_DIR, "loksatta_complete.txt")
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "loksatta_complete.json")
CONFIDENCE_REPORT = os.path.join(OUTPUT_DIR, "confidence_report.json")
VALIDATION_REPORT = os.path.join(OUTPUT_DIR, "validation_report.txt")

# =====================================================
# CONFIDENCE SCORING SYSTEM
# =====================================================

class ConfidenceScorer:
    """Calculate confidence scores for extracted text"""
    
    @staticmethod
    def calculate_word_confidence(ocr_data):
        """Calculate confidence from OCR data"""
        if not ocr_data or 'conf' not in ocr_data:
            return 0, 0, []
        
        confidences = []
        valid_words = []
        
        for i in range(len(ocr_data['conf'])):
            try:
                conf = float(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if text and conf > 0:
                    confidences.append(conf)
                    valid_words.append({
                        'text': text,
                        'confidence': conf,
                        'block': ocr_data['block_num'][i],
                        'line': ocr_data['line_num'][i]
                    })
            except (ValueError, TypeError):
                continue
        
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            return avg_conf, min_conf, max_conf, valid_words
        return 0, 0, 0, []
    
    @staticmethod
    def calculate_marathi_density(text):
        """Calculate percentage of Marathi (Devanagari) characters"""
        if not text:
            return 0
        
        marathi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return 0
        
        return (marathi_chars / total_chars) * 100
    
    @staticmethod
    def calculate_text_quality_metrics(text):
        """Calculate various text quality metrics"""
        if not text:
            return {
                'length': 0,
                'word_count': 0,
                'line_count': 0,
                'avg_word_length': 0,
                'unique_words': 0,
                'marathi_density': 0
            }
        
        words = text.split()
        lines = text.split('\n')
        
        return {
            'length': len(text),
            'word_count': len(words),
            'line_count': len(lines),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'unique_words': len(set(words)),
            'marathi_density': ConfidenceScorer.calculate_marathi_density(text)
        }
    
    @staticmethod
    def calculate_layout_confidence(image, text_blocks):
        """Estimate layout preservation confidence"""
        if not text_blocks:
            return 0
        
        # Heuristic: more blocks = better layout preservation
        num_blocks = len(set(block['block'] for block in text_blocks))
        
        # Normalize to 0-100 scale (assuming 20+ blocks is excellent)
        layout_score = min(100, (num_blocks / 20) * 100)
        
        return layout_score

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    np_image = np.array(pil_image)
    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
        cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    else:
        cv_image = np_image
    return cv_image

def preprocess_for_marathi(pil_image):
    """Specialized preprocessing for Marathi newspaper"""
    img = pil_to_cv2(pil_image)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Threshold
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def ocr_with_confidence(pil_image):
    """OCR with detailed confidence scoring"""
    # Preprocess
    processed = preprocess_for_marathi(pil_image)
    
    # OCR with detailed output
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    
    try:
        # Get detailed data
        data = pytesseract.image_to_data(
            processed,
            lang='mar+eng',
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Calculate confidence scores
        scorer = ConfidenceScorer()
        avg_conf, min_conf, max_conf, valid_words = scorer.calculate_word_confidence(data)
        
        # Extract text
        text_blocks = []
        prev_line = -1
        current_line = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            try:
                conf = float(data['conf'][i])
            except (ValueError, TypeError):
                conf = 0
            
            if text and conf > 30:
                line_num = data['line_num'][i]
                
                if line_num != prev_line and prev_line != -1:
                    if current_line:
                        text_blocks.append(' '.join(current_line))
                        current_line = []
                
                current_line.append(text)
                prev_line = line_num
        
        if current_line:
            text_blocks.append(' '.join(current_line))
        
        extracted_text = '\n'.join(text_blocks)
        
        # Calculate quality metrics
        quality_metrics = scorer.calculate_text_quality_metrics(extracted_text)
        layout_score = scorer.calculate_layout_confidence(processed, valid_words)
        
        # Overall confidence score (weighted average)
        overall_confidence = (
            avg_conf * 0.4 +           # OCR confidence
            quality_metrics['marathi_density'] * 0.3 +  # Marathi density
            layout_score * 0.2 +       # Layout preservation
            min(100, quality_metrics['word_count'] / 10) * 0.1  # Content volume
        )
        
        return {
            'text': extracted_text,
            'ocr_confidence': avg_conf,
            'min_confidence': min_conf,
            'max_confidence': max_conf,
            'overall_confidence': overall_confidence,
            'quality_metrics': quality_metrics,
            'layout_score': layout_score,
            'word_details': valid_words
        }
        
    except Exception as e:
        print(f"  OCR error: {e}")
        return {
            'text': '',
            'ocr_confidence': 0,
            'min_confidence': 0,
            'max_confidence': 0,
            'overall_confidence': 0,
            'quality_metrics': {},
            'layout_score': 0,
            'word_details': []
        }

# =====================================================
# EXTRACTION WITH CONFIDENCE
# =====================================================

def extract_with_confidence(pdf_path, dpi=300):
    """Extract text with detailed confidence scoring"""
    info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
    total_pages = info["Pages"]
    print(f"Total pages to process: {total_pages}")
    
    all_pages_data = []
    
    for page_no in tqdm(range(1, total_pages + 1), desc="Processing pages"):
        try:
            # Convert page
            page_images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_no,
                last_page=page_no,
                poppler_path=POPPLER_PATH
            )
            
            if not page_images:
                all_pages_data.append({
                    "page_number": page_no,
                    "error": "Failed to convert",
                    "success": False
                })
                continue
            
            # OCR with confidence
            result = ocr_with_confidence(page_images[0])
            
            # Store results
            page_data = {
                "page_number": page_no,
                "success": True,
                "text": result['text'],
                "confidence_scores": {
                    "overall": result['overall_confidence'],
                    "ocr": result['ocr_confidence'],
                    "min_word": result['min_confidence'],
                    "max_word": result['max_confidence'],
                    "layout": result['layout_score']
                },
                "quality_metrics": result['quality_metrics']
            }
            
            all_pages_data.append(page_data)
            
            # Print progress
            if result['text']:
                preview = result['text'][:80].replace('\n', ' ')
                print(f"\n  Page {page_no}: Conf={result['overall_confidence']:.1f}% | {len(result['text'])} chars | {preview}...")
            else:
                print(f"\n  Page {page_no}: ❌ No text extracted")
            
        except Exception as e:
            print(f"\n  Page {page_no}: Error - {e}")
            all_pages_data.append({
                "page_number": page_no,
                "success": False,
                "error": str(e)
            })
    
    return all_pages_data

# =====================================================
# VALIDATION AND COMPARISON
# =====================================================

def validate_extraction(pages_data):
    """Validate extraction quality and identify issues"""
    
    validation_results = {
        "summary": {},
        "page_details": [],
        "recommendations": [],
        "comparison_stats": {}
    }
    
    # Separate successful pages
    successful = [p for p in pages_data if p.get('success', False) and p.get('text', '')]
    
    if not successful:
        validation_results["summary"]["error"] = "No pages successfully extracted"
        return validation_results
    
    # Calculate overall statistics
    overall_conf = np.mean([p['confidence_scores']['overall'] for p in successful])
    avg_ocr_conf = np.mean([p['confidence_scores']['ocr'] for p in successful])
    avg_marathi_density = np.mean([p['quality_metrics'].get('marathi_density', 0) for p in successful])
    total_chars = sum(p['quality_metrics'].get('length', 0) for p in successful)
    total_words = sum(p['quality_metrics'].get('word_count', 0) for p in successful)
    
    validation_results["summary"] = {
        "total_pages": len(pages_data),
        "successful_pages": len(successful),
        "failed_pages": len(pages_data) - len(successful),
        "success_rate": (len(successful) / len(pages_data)) * 100,
        "average_overall_confidence": overall_conf,
        "average_ocr_confidence": avg_ocr_conf,
        "average_marathi_density": avg_marathi_density,
        "total_characters_extracted": total_chars,
        "total_words_extracted": total_words,
        "quality_rating": get_quality_rating(overall_conf, avg_marathi_density)
    }
    
    # Analyze each page
    for page in pages_data:
        if page.get('success', False):
            page_validation = {
                "page_number": page['page_number'],
                "status": "SUCCESS",
                "confidence": page['confidence_scores']['overall'],
                "ocr_confidence": page['confidence_scores']['ocr'],
                "marathi_density": page['quality_metrics'].get('marathi_density', 0),
                "char_count": page['quality_metrics'].get('length', 0),
                "word_count": page['quality_metrics'].get('word_count', 0),
                "quality_issues": identify_quality_issues(page)
            }
        else:
            page_validation = {
                "page_number": page['page_number'],
                "status": "FAILED",
                "error": page.get('error', 'Unknown error'),
                "quality_issues": ["Extraction failed"]
            }
        
        validation_results["page_details"].append(page_validation)
    
    # Generate recommendations
    validation_results["recommendations"] = generate_recommendations(validation_results["summary"], pages_data)
    
    # Create comparison stats (if we have multiple extraction runs)
    validation_results["comparison_stats"] = generate_comparison_stats(pages_data)
    
    return validation_results

def get_quality_rating(overall_conf, marathi_density):
    """Get quality rating based on metrics"""
    if overall_conf >= 80 and marathi_density >= 70:
        return "EXCELLENT"
    elif overall_conf >= 60 and marathi_density >= 50:
        return "GOOD"
    elif overall_conf >= 40 and marathi_density >= 30:
        return "FAIR"
    elif overall_conf > 0:
        return "POOR"
    else:
        return "FAILED"

def identify_quality_issues(page):
    """Identify specific quality issues for a page"""
    issues = []
    
    conf = page['confidence_scores']['overall']
    marathi_density = page['quality_metrics'].get('marathi_density', 0)
    word_count = page['quality_metrics'].get('word_count', 0)
    
    if conf < 50:
        issues.append("Low confidence score")
    if marathi_density < 30:
        issues.append("Low Marathi character density (possible encoding issues)")
    if word_count < 50:
        issues.append("Very few words extracted")
    if page['confidence_scores']['min_word'] < 30:
        issues.append("Some words have very low confidence")
    
    # Check for potential garbage text
    text = page.get('text', '')
    if text and len(text) > 100:
        garbage_ratio = sum(1 for c in text if ord(c) > 0x097F and not c.isspace()) / len(text)
        if garbage_ratio > 0.3:
            issues.append("High ratio of non-Marathi characters")
    
    return issues

def generate_recommendations(summary, pages_data):
    """Generate actionable recommendations"""
    recommendations = []
    
    if summary['success_rate'] < 80:
        recommendations.append("⬆️ Consider increasing DPI to 400 for better accuracy")
    
    if summary['average_marathi_density'] < 50:
        recommendations.append("🔤 Install/update Marathi language pack for Tesseract")
        recommendations.append("📝 Apply additional post-processing for Marathi script")
    
    if summary['average_overall_confidence'] < 60:
        recommendations.append("🎨 Improve preprocessing: try different thresholding methods")
        recommendations.append("🖼️ Consider deskewing pages before OCR")
    
    # Check for problematic pages
    low_quality_pages = [p for p in pages_data if p.get('success') and p['confidence_scores']['overall'] < 40]
    if low_quality_pages:
        pages_list = [p['page_number'] for p in low_quality_pages]
        recommendations.append(f"⚠️ Pages {pages_list} need manual review or reprocessing")
    
    if not recommendations:
        recommendations.append("✅ Extraction quality is good! No immediate improvements needed")
    
    return recommendations

def generate_comparison_stats(pages_data):
    """Generate statistics comparing confidence vs extracted data"""
    
    successful = [p for p in pages_data if p.get('success', False)]
    
    if not successful:
        return {}
    
    # Group by confidence levels
    high_conf = [p for p in successful if p['confidence_scores']['overall'] >= 70]
    med_conf = [p for p in successful if 40 <= p['confidence_scores']['overall'] < 70]
    low_conf = [p for p in successful if p['confidence_scores']['overall'] < 40]
    
    return {
        "high_confidence_pages": {
            "count": len(high_conf),
            "avg_chars": np.mean([p['quality_metrics'].get('length', 0) for p in high_conf]) if high_conf else 0,
            "avg_words": np.mean([p['quality_metrics'].get('word_count', 0) for p in high_conf]) if high_conf else 0
        },
        "medium_confidence_pages": {
            "count": len(med_conf),
            "avg_chars": np.mean([p['quality_metrics'].get('length', 0) for p in med_conf]) if med_conf else 0,
            "avg_words": np.mean([p['quality_metrics'].get('word_count', 0) for p in med_conf]) if med_conf else 0
        },
        "low_confidence_pages": {
            "count": len(low_conf),
            "avg_chars": np.mean([p['quality_metrics'].get('length', 0) for p in low_conf]) if low_conf else 0,
            "avg_words": np.mean([p['quality_metrics'].get('word_count', 0) for p in low_conf]) if low_conf else 0
        },
        "confidence_vs_extraction": {
            "correlation": "Positive" if len(high_conf) > len(low_conf) else "Negative",
            "high_conf_avg_chars": np.mean([p['quality_metrics'].get('length', 0) for p in high_conf]) if high_conf else 0,
            "low_conf_avg_chars": np.mean([p['quality_metrics'].get('length', 0) for p in low_conf]) if low_conf else 0
        }
    }

def save_validation_report(validation_results, output_path):
    """Save comprehensive validation report"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PDF EXTRACTION VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary
        f.write("📊 SUMMARY STATISTICS\n")
        f.write("-"*40 + "\n")
        for key, value in validation_results['summary'].items():
            if isinstance(value, float):
                f.write(f"{key:30}: {value:.2f}\n")
            else:
                f.write(f"{key:30}: {value}\n")
        
        f.write("\n\n📈 PAGE-BY-PAGE ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        for page in validation_results['page_details']:
            f.write(f"\nPage {page['page_number']}: [{page['status']}]\n")
            if page['status'] == 'SUCCESS':
                f.write(f"  Overall Confidence: {page['confidence']:.1f}%\n")
                f.write(f"  OCR Confidence: {page['ocr_confidence']:.1f}%\n")
                f.write(f"  Marathi Density: {page['marathi_density']:.1f}%\n")
                f.write(f"  Characters: {page['char_count']:,}\n")
                f.write(f"  Words: {page['word_count']:,}\n")
                if page['quality_issues']:
                    f.write(f"  Issues: {', '.join(page['quality_issues'])}\n")
            else:
                f.write(f"  Error: {page.get('error', 'Unknown')}\n")
        
        f.write("\n\n💡 RECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        for rec in validation_results['recommendations']:
            f.write(f"  • {rec}\n")
        
        f.write("\n\n📊 CONFIDENCE VS EXTRACTION COMPARISON\n")
        f.write("-"*40 + "\n")
        stats = validation_results.get('comparison_stats', {})
        if stats:
            f.write(f"High Confidence Pages (>70%): {stats['high_confidence_pages']['count']} pages\n")
            f.write(f"  Average extracted chars: {stats['high_confidence_pages']['avg_chars']:.0f}\n")
            f.write(f"Medium Confidence Pages (40-70%): {stats['medium_confidence_pages']['count']} pages\n")
            f.write(f"  Average extracted chars: {stats['medium_confidence_pages']['avg_chars']:.0f}\n")
            f.write(f"Low Confidence Pages (<40%): {stats['low_confidence_pages']['count']} pages\n")
            f.write(f"  Average extracted chars: {stats['low_confidence_pages']['avg_chars']:.0f}\n")
            
            f.write(f"\nCorrelation: {stats['confidence_vs_extraction']['correlation']}\n")
            if stats['confidence_vs_extraction']['high_conf_avg_chars'] > 0:
                ratio = stats['confidence_vs_extraction']['low_conf_avg_chars'] / stats['confidence_vs_extraction']['high_conf_avg_chars']
                f.write(f"Low confidence pages extract {ratio:.1%} of the text compared to high confidence pages\n")
    
    print(f"\n✅ Validation report saved to: {output_path}")

def create_comparison_chart(pages_data, output_dir):
    """Create a simple text-based comparison chart"""
    chart_path = os.path.join(output_dir, "confidence_chart.txt")
    
    with open(chart_path, 'w', encoding='utf-8') as f:
        f.write("Confidence vs Extraction Quality Chart\n")
        f.write("="*60 + "\n\n")
        
        successful = [p for p in pages_data if p.get('success', False)]
        
        f.write("Page | Confidence | Marathi% | Words | Rating\n")
        f.write("-"*60 + "\n")
        
        for page in successful:
            conf = page['confidence_scores']['overall']
            marathi = page['quality_metrics'].get('marathi_density', 0)
            words = page['quality_metrics'].get('word_count', 0)
            
            # Create bar for confidence
            bar_length = int(conf / 10)
            conf_bar = "█" * bar_length + "░" * (10 - bar_length)
            
            rating = get_quality_rating(conf, marathi)
            
            f.write(f"{page['page_number']:3}  | {conf_bar} {conf:5.1f}% | {marathi:5.1f}% | {words:5} | {rating}\n")
    
    print(f"✅ Confidence chart saved to: {chart_path}")

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    
    print("="*60)
    print("CONFIDENCE-BASED EXTRACTION SYSTEM")
    print("="*60)
    print(f"PDF: {PDF_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()
    
    # Check if PDF exists
    if not os.path.exists(PDF_FILE):
        print(f"❌ PDF file not found: {PDF_FILE}")
        exit(1)
    
    # Extract with confidence scoring
    print("Starting extraction with confidence scoring...")
    pages_data = extract_with_confidence(PDF_FILE, dpi=300)
    
    # Validate extraction
    print("\nValidating extraction quality...")
    validation_results = validate_extraction(pages_data)
    
    # Save validation report
    save_validation_report(validation_results, VALIDATION_REPORT)
    
    # Create comparison chart
    create_comparison_chart(pages_data, OUTPUT_DIR)
    
    # Save detailed JSON with confidence data
    with open(CONFIDENCE_REPORT, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)
    
    # Save extracted text
    with open(TXT_OUTPUT, 'w', encoding='utf-8') as f:
        for page in pages_data:
            if page.get('success', False):
                f.write(f"\n{'='*100}\n")
                f.write(f"PAGE {page['page_number']}\n")
                f.write(f"Confidence: {page['confidence_scores']['overall']:.1f}%\n")
                f.write(f"Marathi Density: {page['quality_metrics'].get('marathi_density', 0):.1f}%\n")
                f.write(f"{'='*100}\n\n")
                f.write(page['text'])
                f.write("\n\n")
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\n📊 Quality Rating: {validation_results['summary']['quality_rating']}")
    print(f"📈 Success Rate: {validation_results['summary']['success_rate']:.1f}%")
    print(f"🎯 Average Confidence: {validation_results['summary']['average_overall_confidence']:.1f}%")
    print(f"📝 Total Characters: {validation_results['summary']['total_characters_extracted']:,}")
    
    print(f"\n📄 Output files:")
    print(f"  • Text: {TXT_OUTPUT}")
    print(f"  • Validation Report: {VALIDATION_REPORT}")
    print(f"  • Confidence Report: {CONFIDENCE_REPORT}")
    print(f"  • Comparison Chart: {OUTPUT_DIR}/confidence_chart.txt")
    
    # Show top recommendations
    if validation_results['recommendations']:
        print(f"\n💡 Top recommendations:")
        for rec in validation_results['recommendations'][:3]:
            print(f"  • {rec}")