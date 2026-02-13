"""
HTML to text conversion functions.
"""
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
import re


def html_to_text(html_file_path: Path) -> str:
    """
    Extract clean text content from an HTML file.
    
    Args:
        html_file_path: Path to HTML file
    
    Returns:
        Clean text content
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "meta", "link", "noscript"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text


def convert_html_files_in_directory(input_dir: Path) -> List[Path]:
    """
    Convert all HTML files in a directory to text files.
    
    Args:
        input_dir: Directory containing HTML files
    
    Returns:
        List of paths to created .txt files
    """
    html_files = list(input_dir.glob("*.html"))
    created_files = []
    
    for html_file in html_files:
        text_content = html_to_text(html_file)
        
        # Save as .txt file
        txt_file = input_dir / f"{html_file.stem}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        created_files.append(txt_file)
    
    return created_files
