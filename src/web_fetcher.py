"""
Web fetching functions for SearXNG search and HTML page downloading.
"""
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests


def search_searxng(
    query: str,
    base_url: str = "http://localhost:8080",
    format: str = "json",
    engines: Optional[str] = None,
    language: str = "en-US",
    pageno: int = 1,
    time_range: Optional[str] = None,
    safesearch: int = 0
) -> Dict:
    """
    Perform a search using SearXNG API.
    
    Based on SearXNG Search API documentation: https://docs.searxng.org/dev/search_api.html
    
    Args:
        query: Search query string
        base_url: SearXNG instance URL (default: http://localhost:8080)
        format: Response format ('json', 'csv', or 'rss') - default: 'json'
        engines: Comma-separated list of engines (optional)
        language: Language code (default: 'en-US')
        pageno: Page number (default: 1)
        time_range: Time range filter ('day', 'week', 'month', 'year')
        safesearch: SafeSearch level (0=off, 1=moderate, 2=strict)
    
    Returns:
        Dictionary containing search results:
        - 'results': List of result dictionaries
        - 'query': Original query
        - 'number_of_results': Estimated total
        - 'error': Error message if request failed
    """
    search_url = f"{base_url.rstrip('/')}/search"
    
    params = {
        'q': query,
        'format': format,
        'language': language,
        'pageno': pageno,
        'safesearch': safesearch
    }
    
    if engines:
        params['engines'] = engines
    if time_range:
        params['time_range'] = time_range
    
    try:
        response = requests.get(search_url, params=params, timeout=30)
        
        if response.status_code == 200:
            if format == 'json':
                return response.json()
            else:
                return {'content': response.text, 'status_code': 200}
        else:
            return {
                'error': f"HTTP {response.status_code}",
                'message': response.text[:500],
                'status_code': response.status_code
            }
    except requests.exceptions.RequestException as e:
        return {
            'error': 'Request failed',
            'message': str(e)
        }


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Sanitize text to create a valid filename.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length of filename
    
    Returns:
        Sanitized filename-safe string
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Replace multiple underscores with single
    sanitized = re.sub(r'_+', '_', sanitized)
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def download_html_page(url: str, timeout: int = 30) -> Optional[str]:
    """
    Download HTML content from a URL.
    
    Args:
        url: URL to download
        timeout: Request timeout in seconds
    
    Returns:
        HTML content as string, or None if download failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        # Check if content is HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type and 'text' not in content_type:
            return None
        
        return response.text
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None


def fetch_keyword_pages(
    keyword: str,
    output_dir: Path,
    searxng_url: str = "http://localhost:8080",
    max_pages: int = 10
) -> List[Path]:
    """
    Search for keyword using SearXNG, download top HTML pages, and save them.
    
    Args:
        keyword: Search keyword/query
        output_dir: Directory to save HTML files
        searxng_url: SearXNG instance URL
        max_pages: Maximum number of pages to download
    
    Returns:
        List of paths to downloaded HTML files
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Search using SearXNG
    print(f"   🔍 Searching SearXNG for: {keyword}")
    search_results = search_searxng(keyword, base_url=searxng_url)
    
    if 'error' in search_results:
        raise Exception(f"SearXNG search failed: {search_results.get('error', 'Unknown error')} - {search_results.get('message', '')}")
    
    results = search_results.get('results', [])
    if not results:
        raise Exception(f"No search results found for keyword: {keyword}")
    
    print(f"   ✓ Found {len(results)} search results")
    
    # Extract URLs (limit to max_pages)
    urls = []
    for result in results[:max_pages]:
        url = result.get('url')
        if url:
            urls.append(url)
    
    if not urls:
        raise Exception("No valid URLs found in search results")
    
    print(f"   📥 Downloading {len(urls)} HTML pages...")
    
    # Download and save HTML pages; collect (filename, url) for url_manifest.json
    downloaded_files = []
    url_manifest_entries = []
    failed_count = 0
    
    for i, url in enumerate(urls, 1):
        try:
            # Download HTML
            html_content = download_html_page(url)
            
            if html_content is None:
                print(f"      ⚠️  [{i}/{len(urls)}] Failed to download: {url}")
                failed_count += 1
                continue
            
            # Generate filename from URL or title
            title = results[i-1].get('title', '') if i-1 < len(results) else ''
            if title:
                filename = sanitize_filename(title) + '.html'
            else:
                # Use URL domain and path
                parsed = urlparse(url)
                path_part = parsed.path.strip('/').replace('/', '_')
                domain = parsed.netloc.replace('.', '_')
                if path_part:
                    filename = f"{domain}_{path_part}.html"
                else:
                    filename = f"{domain}.html"
                filename = sanitize_filename(filename)
            
            # Handle duplicates
            filepath = output_dir / filename
            counter = 1
            while filepath.exists():
                base_name = filename.rsplit('.', 1)[0]
                filepath = output_dir / f"{base_name}_{counter}.html"
                counter += 1
            filename = filepath.name
            
            # Save HTML file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            downloaded_files.append(filepath)
            url_manifest_entries.append({"filename": filename, "url": url})
            print(f"      ✓ [{i}/{len(urls)}] Saved: {filepath.name}")
            
            # Small delay to be respectful
            time.sleep(0.5)
            
        except Exception as e:
            print(f"      ⚠️  [{i}/{len(urls)}] Error downloading {url}: {str(e)}")
            failed_count += 1
            continue
    
    if not downloaded_files:
        raise Exception(f"Failed to download any pages. {failed_count} failed downloads.")
    
    # Write url_manifest.json so curation can resolve document titles to website URLs
    if url_manifest_entries:
        manifest_path = output_dir / "url_manifest.json"
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({"entries": url_manifest_entries}, f, indent=2)
        except Exception:
            pass  # non-fatal
    
    if failed_count > 0:
        print(f"   ⚠️  {failed_count} download(s) failed, but {len(downloaded_files)} succeeded")
    
    print(f"   ✓ Successfully downloaded {len(downloaded_files)} HTML page(s)")
    return downloaded_files
