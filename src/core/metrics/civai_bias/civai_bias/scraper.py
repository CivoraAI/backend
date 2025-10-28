#!/usr/bin/env python3
"""
US Politics News Scraper
Scrapes US politics news from specified sites with bias scores.
Runs every hour and stores all articles in a single JSON file.
"""

import requests
import json
import time
import os
import re
import schedule
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, parse_qs, urlencode

# Import advanced cleaner
try:
    from advanced_cleaner import clean_article_text_advanced
    ADVANCED_CLEANER_AVAILABLE = True
except ImportError:
    ADVANCED_CLEANER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('us_politics_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_unicode_text(text: str) -> str:
    """Convert Unicode characters to ASCII equivalents"""
    if not isinstance(text, str):
        return text
    
    # Common Unicode replacements
    replacements = {
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark (apostrophe)
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Horizontal ellipsis
        '\u00a0': ' ',  # Non-breaking space
        '\u00e9': 'e',  # é
        '\u00e1': 'a',  # á
        '\u00ed': 'i',  # í
        '\u00f3': 'o',  # ó
        '\u00fa': 'u',  # ú
        '\u00f1': 'n',  # ñ
        '\u00c9': 'E',  # É
        '\u2022': '*',  # Bullet point
        '\u00b0': ' degrees',  # Degree symbol
        '\u00ae': '(R)',  # Registered trademark
        '\u2122': '(TM)',  # Trademark
        '\u00a9': '(C)',  # Copyright
    }
    
    # Apply replacements
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def remove_boilerplate_text(text: str) -> str:
    """Remove common boilerplate and navigation text from scraped articles"""
    if not isinstance(text, str):
        return text
    
    # Common boilerplate patterns to remove (case-insensitive)
    boilerplate_patterns = [
        # Browser warnings
        r'IE \d+ is not supported\.?\s*For an optimal experience visit our site on another browser\.?',
        r'This browser is no longer supported\.?',
        r'Please upgrade (your|to a) (browser|modern browser)\.?',
        
        # Navigation elements
        r'Skip to (Content|Main Content|Navigation|Footer)',
        r'Jump to (Content|Navigation)',
        r'Skip to main content',
        r'ReadMore',
        
        # Cookie/Privacy notices
        r'(This|Our) (site|website) uses cookies\..*?(\.|Learn more)',
        r'By continuing to use this site.*?(\.|privacy policy)',
        r'We use cookies.*?(\.|Accept)',
        
        # Social media prompts
        r'Share this.*?-\s*Copied',
        r'Share this\s*-\s*Copied',
        r'Follow us on (Twitter|Facebook|Instagram|LinkedIn)',
        r'Subscribe to our (newsletter|channel)',
        
        # Video/Media controls
        r'Now Playing.*?UP NEXT',
        r'UP NEXT\s*',
        r'Play All',
        r'\d+:\d+\s*(AM|PM|Now Playing)',
        
        # Copyright/Legal
        r'\(C\) \d{4}.*?(LLC|Inc|Corp|L\.L\.C\.)\.?',
        r'Copyright \d{4}.*?(\.|All rights reserved)',
        r'All rights reserved\.?',
        
        # Common navigation text
        r'Read More\s*$',
        r'Click here for more.*?$',
        r'Sign up.*?newsletter',
        
        # MSNBC/News specific
        r'MSNBC (HIGHLIGHTS|Cable).*?(BEST OF MSNBC)?',
        r'\(BEST OF MSNBC\)',
        r'(NBC|CBS|ABC|CNN|Fox|MSNBC|Today) (News )?Logo',
        r'The (Latest|Weekend|Morning|Evening)(\s+Show)?',
        
        # Show titles and sections that appear as navigation
        r'Ali Velshi\s*',
        r'The Weekend:?\s*(Primetime)?',
        r': Primetime\s*',
        
        # Repeated video timestamps and dates
        r'\d{1,2}:\d{2}(?:\s*\d{1,2}:\d{2})*\s*',
        r'\w+\.?\s+\d{1,2},?\s+\d{4}(?=\s*[A-Z])',  # Dates followed by caps
        
        # Author/publication metadata at start
        r'^(By|Author):\s*\w+.*?\d{4}\s*',
        r'^Published:?\s*\w+\s+\d+,?\s+\d{4}\s*',
        
        # Video page artifacts
        r'EXCLUSIVE:\s*',
        r"'Complete detachment from reality':\s*",
    ]
    
    # Apply all boilerplate removal patterns
    cleaned_text = text
    for pattern in boilerplate_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove standalone timestamps (e.g., "12:31")
    cleaned_text = re.sub(r'\b\d{1,2}:\d{2}\b', '', cleaned_text)
    
    # Remove month/day/year patterns when followed by capital letter (likely start of new section)
    cleaned_text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}(?=\s*[A-Z])', '', cleaned_text)
    
    # Remove multiple consecutive spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Remove multiple consecutive newlines
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    
    # Trim whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

@dataclass
class PoliticsArticle:
    """Data class for politics article"""
    url: str
    title: str
    author: Optional[str] = None
    publish_date: Optional[str] = None
    scraped_at: str = None
    full_text: str = ""
    description: Optional[str] = None
    source_domain: str = ""
    word_count: int = 0
    
    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.now(timezone.utc).isoformat()
        if self.source_domain == "":
            self.source_domain = urlparse(self.url).netloc

class USPoliticsScraper:
    """Scraper specifically for US Politics news from specified sources"""
    
    def __init__(self, output_file: str = "/Users/arav/Documents/GitHub/backend/src/core/metrics/civai_bias/data/news_data.json"):
        self.output_file = Path(output_file)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Cache of existing URLs for faster duplicate checking
        self._existing_urls_cache = None
        self._cache_loaded = False
        
                # News sites with politics URLs
        self.news_sites = {
            "apnews.com": {
                "politics_urls": [
                    "https://apnews.com/hub/politics",
                    "https://apnews.com/hub/election-2024",
                    "https://apnews.com/hub/congress"
                ]
            },
            "theguardian.com/us": {
                "politics_urls": [
                    "https://www.theguardian.com/us-news/us-politics",
                    "https://www.theguardian.com/us-news/us-congress",
                    "https://www.theguardian.com/us-news/biden-administration"
                ]
            },
            "msnbc.com": {
                "politics_urls": [
                    "https://www.msnbc.com/politics"
                ]
            },
            "theatlantic.com": {
                "politics_urls": [
                    "https://www.theatlantic.com/politics/",
                    "https://www.theatlantic.com/ideas/"
                ]
            },
            "vox.com": {
                "politics_urls": [
                    "https://www.vox.com/policy-and-politics",
                    "https://www.vox.com/politics"
                ]
            },
            "abcnews.go.com": {
                "politics_urls": [
                    "https://abcnews.go.com/Politics"
                ]
            },
            "cnn.com": {
                "politics_urls": [
                    "https://www.cnn.com/politics",
                    "https://www.cnn.com/election/2024"
                ]
            },
            "nytimes.com": {
                "politics_urls": [
                    "https://www.nytimes.com/section/politics",
                    "https://www.nytimes.com/section/us/politics"
                ]
            },
            "nbcnews.com": {
                "politics_urls": [
                    "https://www.nbcnews.com/politics",
                    "https://www.nbcnews.com/politics/congress"
                ]
            },
            "cbsnews.com": {
                "politics_urls": [
                    "https://www.cbsnews.com/politics/",
                    "https://www.cbsnews.com/election-2024/"
                ]
            },
            "cnbc.com": {
                "politics_urls": [
                    "https://www.cnbc.com/politics/"
                ]
            },
            "npr.org": {
                "politics_urls": [
                    "https://www.npr.org/sections/politics/",
                    "https://www.npr.org/sections/congress/"
                ]
            },
            "time.com": {
                "politics_urls": [
                    "https://time.com/section/politics/"
                ]
            },
            "washingtonpost.com": {
                "politics_urls": [
                    "https://feeds.washingtonpost.com/rss/politics"
                ]
            },
            "bbc.com": {
                "politics_urls": [
                    "https://www.bbc.com/news/world/us_and_canada"
                ]
            },
            "reuters.com": {
                "politics_urls": [
                    "https://www.reuters.com/world/us/"
                ]
            },
            "newsweek.com": {
                "politics_urls": [
                    "https://www.newsweek.com/politics"
                ]
            },
            "san.com": {
                "politics_urls": [
                    "https://www.san.com/politics/"
                ]
            },
            "thehill.com": {
                "politics_urls": [
                    "https://thehill.com/policy/"
                ]
            },
            "newsnationnow.com": {
                "politics_urls": [
                    "https://www.newsnationnow.com/politics/"
                ]
            },
            "reason.com": {
                "politics_urls": [
                    "https://reason.com/politics/"
                ]
            },
            "theepochtimes.com": {
                "politics_urls": [
                    "https://www.theepochtimes.com/c-us-politics"
                ]
            },
            "justthenews.com": {
                "politics_urls": [
                    "https://justthenews.com/politics-policy"
                ]
            },
            "nypost.com": {
                "politics_urls": [
                    "https://nypost.com/news/politics/"
                ]
            },
            "washingtontimes.com": {
                "politics_urls": [
                    "https://www.washingtontimes.com/news/politics/"
                ]
            },
            "thedispatch.com": {
                "politics_urls": [
                    "https://thedispatch.com/politics/"
                ]
            },
            "foxnews.com": {
                "politics_urls": [
                    "https://www.foxnews.com/politics"
                ]
            },
            "ijr.com": {
                "politics_urls": [
                    "https://ijr.com/politics/"
                ]
            },
            "theblaze.com": {
                "politics_urls": [
                    "https://www.theblaze.com/news/politics"
                ]
            },
            "freebeacon.com": {
                "politics_urls": [
                    "https://freebeacon.com/politics/"
                ]
            }
        }
        
        # Site-specific selectors for better extraction
        self.site_selectors = {
            'abcnews.go.com': {
                'title': ['h1', '.Article__Headline', '[data-testid="headline"]', '.ContentRoll__Headline', '.Link__text'],
                'author': ['.Byline__Author', '.author', '[data-testid="byline"]', '.ContentRoll__Byline'],
                'date': ['.Timestamp', 'time', '[data-testid="timestamp"]', '.ContentRoll__Timestamp'],
                'content': ['.Article__Content', '.RichTextArticleBody', '.article-content p', '.ContentRoll__Summary'],
                'links': ['.ContentRoll__Headline a', '.Link', 'h1 a', 'h2 a', 'h3 a', 'a[href*="/Politics/"]', 'a[href*="/story"]']
            },
            'cnbc.com': {
                'title': ['h1', '.ArticleHeader-headline', '[data-testid="headline"]'],
                'author': ['.Author-authorName', '.byline', '[data-testid="author"]'],
                'date': ['.ArticleHeader-time', 'time', '[data-testid="published-timestamp"]'],
                'content': ['.ArticleBody-articleBody', '.InlineVideo-container ~ p', '.article-content p'],
                'links': ['.Card-titleLink', '.RelatedContent a', 'h1 a', 'h2 a', 'h3 a']
            },
            'apnews.com': {
                'title': ['h1', '[data-key="card-headline"]', '.CardHeadline'],
                'author': ['.Component-bylines-0-2-114', '.byline'],
                'date': ['[data-key="timestamp"]', '.Timestamp'],
                'content': ['.Article', '.RichTextStoryBody', 'div[data-key="article"] p'],
                'links': ['h3 a', '.PageList-items a', '.CardHeadline a']
            },
            'theguardian.com': {
                'title': ['h1', '[data-gu-name="headline"]'],
                'author': ['[rel="author"]', '.dcr-u0h1qy'],
                'date': ['[data-testid="timestamp"]', 'time'],
                'content': ['[data-gu-name="body"]', '.content__article-body'],
                'links': ['[data-link-name="article"]', '.fc-item__link']
            },
            'msnbc.com': {
                'title': ['h1', '.articleTitle', '[data-testid="headline"]', '.tease-card__headline', '.tease__headline'],
                'author': ['.byline-author', '.author', '[data-testid="byline"]', '.tease__meta'],
                'date': ['.timestamp', '.datePublished', 'time', '.tease__meta'],
                'content': ['.articleBody', '.InlineVideo-container ~ p', '[data-testid="article-content"]', '.article-content p', '.tease__content'],
                'links': ['.tease-card__headline a', '.tease__headline a', '.related-content a', 'h1 a', 'h2 a', 'h3 a', 'a[href*="/news/"]', 'a[href*="/politics/"]']
            },
            'cnn.com': {
                'title': ['h1', '.headline__text'],
                'author': ['.byline__name', '.metadata__byline__author'],
                'date': ['.timestamp', '.metadata__timestamp'],
                'content': ['.article__content', '.zn-body__paragraph'],
                'links': ['.cd__headline-text a', '.container__link']
            },
            'foxnews.com': {
                'title': ['h1', '.headline'],
                'author': ['.author-byline', '.byline'],
                'date': ['.article-date', 'time'],
                'content': ['.article-body', '.content'],
                'links': ['.title a', '.content a[href*="/politics"]']
            },
            'nytimes.com': {
                'title': ['h1', '[data-testid="headline"]'],
                'author': ['[data-testid="byline"]', '.byline'],
                'date': ['[data-testid="timestamp"]', 'time'],
                'content': ['[data-testid="article-content"]', '.StoryBodyCompanionColumn p'],
                'links': ['[data-testid="headline-link"]', '.story-link']
            },
            'washingtonpost.com': {
                'title': ['h1', '[data-testid="headline"]'],
                'author': ['.author', '.wpds-c-byline'],
                'date': ['[data-testid="timestamp"]', 'time'],
                'content': ['.article-body', '.wpds-c-article-body'],
                'links': ['.headline a', '.story-headline a']
            },
            'thedispatch.com': {
                'title': ['h1', '.entry-title', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['article p', '.entry-content', '.article-content', '.post-content'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.entry-title a', '.article-link a']
            },
            'newsweek.com': {
                'title': ['h1', '.headline', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['.article-content', '.entry-content', '.post-content'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.headline a', '.article-link a']
            },
            'san.com': {
                'title': ['h1', '.headline', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['.article-content', '.entry-content', '.post-content'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.headline a', '.article-link a']
            },
            'theamericanconservative.com': {
                'title': ['h1', '.entry-title', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['main p', '.entry-content', '.article-content', '.post-content', 'article p'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.entry-title a', '.article-link a']
            },
            'newsweek.com': {
                'title': ['h1', '.headline', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['.article-content', '.entry-content', '.post-content', 'article p'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.headline a', '.article-link a']
            },
            'newsnationnow.com': {
                'title': ['h1', '.headline', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['.article-content', '.entry-content', '.post-content'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.headline a', '.article-link a']
            },
            'reason.com': {
                'title': ['h1', '.entry-title', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['.entry-content', '.article-content', '.post-content'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.entry-title a', '.article-link a']
            },
            'ijr.com': {
                'title': ['h1', '.headline', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['.article-content', '.entry-content', '.post-content'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.headline a', '.article-link a']
            },
            'newsmax.com': {
                'title': ['h1', '.headline', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['.article-content', '.entry-content', '.post-content'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.headline a', '.article-link a']
            },
            'theblaze.com': {
                'title': ['h1', '.headline', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['.article-content', '.entry-content', '.post-content'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.headline a', '.article-link a']
            },
            'freebeacon.com': {
                'title': ['h1', '.entry-title', '.article-title'],
                'author': ['.author', '.byline', '.post-author'],
                'date': ['.date', '.published', 'time'],
                'content': ['.entry-content', '.article-content', '.post-content'],
                'links': ['h1 a', 'h2 a', 'h3 a', '.entry-title a', '.article-link a']
            }
        }
        
        # Common selectors as fallback
        self.common_selectors = {
            'title': ['h1', 'h2.title', '.headline', '.article-title', '.entry-title'],
            'author': ['.author', '.byline', '[rel="author"]', '.article-author', '.writer'],
            'date': ['.date', '.publish-date', '.article-date', 'time', '[datetime]', '.timestamp'],
            'content': ['.article-content', '.entry-content', '.post-content', '.article-body', '.story-body', 'article p'],
            'links': ['h1 a', 'h2 a', 'h3 a', '.headline a', '.title a', '.article-title a']
        }
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL to handle variations of the same article.
        Removes tracking parameters, ensures consistent scheme, etc.
        """
        try:
            parsed = urlparse(url)
            
            # Convert to lowercase
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()
            
            # Remove www. prefix for consistency
            if netloc.startswith('www.'):
                netloc = netloc[4:]
            
            # Remove common tracking parameters
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'fbclid', 'gclid', 'msclkid', 'mc_cid', 'mc_eid',
                'ref', 'source', '_ga', '_gl', 'click_id'
            }
            
            # Parse and filter query parameters
            query_params = parse_qs(parsed.query, keep_blank_values=False)
            filtered_params = {
                k: v for k, v in query_params.items() 
                if k not in tracking_params
            }
            
            # Rebuild query string
            query = urlencode(sorted(filtered_params.items()), doseq=True) if filtered_params else ''
            
            # Remove trailing slashes from path
            path = parsed.path.rstrip('/')
            
            # Reconstruct normalized URL
            normalized = f"{scheme}://{netloc}{path}"
            if query:
                normalized += f"?{query}"
            
            return normalized
            
        except Exception as e:
            logger.debug(f"Error normalizing URL {url}: {e}")
            return url
    
    def load_existing_urls(self) -> set:
        """
        Load and cache existing article URLs for fast duplicate checking.
        Returns a set of normalized URLs.
        """
        if not self._cache_loaded:
            existing_articles = self.load_existing_articles()
            self._existing_urls_cache = {
                self.normalize_url(article.get('url', '')) 
                for article in existing_articles
            }
            self._cache_loaded = True
            logger.info(f"Loaded {len(self._existing_urls_cache)} existing URLs into cache")
        
        return self._existing_urls_cache
    
    def is_duplicate(self, url: str) -> bool:
        """
        Check if an article URL already exists in the database.
        Uses normalized URLs for comparison.
        """
        normalized_url = self.normalize_url(url)
        existing_urls = self.load_existing_urls()
        return normalized_url in existing_urls
    
    def get_page(self, url: str, timeout: int = 30) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page with improved error handling"""
        try:
            logger.info(f"Fetching: {url}")
            
            # Add some headers to look more like a real browser
            headers = {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = self.session.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            
            # Use XML parser for RSS feeds
            if url.endswith('/feed/') or 'feed' in url:
                soup = BeautifulSoup(response.content, 'xml')
            else:
                soup = BeautifulSoup(response.content, 'html.parser')
            return soup
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching {url}")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.error(f"Access forbidden for {url} (403)")
            elif e.response.status_code == 404:
                logger.error(f"Page not found for {url} (404)")
            else:
                logger.error(f"HTTP error fetching {url}: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def get_selectors_for_site(self, url: str) -> Dict[str, List[str]]:
        """Get selectors for a specific site"""
        domain = urlparse(url).netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check for exact match or partial match
        for site_domain, selectors in self.site_selectors.items():
            if site_domain in domain or domain in site_domain:
                return selectors
        
        return self.common_selectors
    
    def extract_text_by_selectors(self, soup: BeautifulSoup, selectors: List[str]) -> Optional[str]:
        """Extract text using multiple selectors"""
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    if text:
                        return text
            except Exception as e:
                logger.debug(f"Selector '{selector}' failed: {e}")
                continue
        return None
    
    def extract_full_text(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Extract full article text"""
        full_text = ""
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    paragraphs = []
                    for element in elements:
                        text = element.get_text(strip=True)
                        if text and len(text) > 20:  # Filter out very short text
                            paragraphs.append(text)
                    
                    if paragraphs:
                        full_text = "\n\n".join(paragraphs)
                        break
            except Exception as e:
                logger.debug(f"Content selector '{selector}' failed: {e}")
                continue
        
        # Fallback: try to extract any substantial text from the page
        if not full_text or len(full_text) < 100:
            fallback_selectors = ['main', 'article', '.content', '#content', '.post', '.entry']
            for selector in fallback_selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        # Remove navigation, ads, etc.
                        for unwanted in element.select('nav, .nav, .menu, .sidebar, .ad, .advertisement, .social, .share'):
                            unwanted.decompose()
                        
                        text = element.get_text(strip=True)
                        if len(text) > 200:
                            full_text = text
                            break
                except Exception:
                    continue
        
        # Clean up the text
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = re.sub(r'\n\s*\n', '\n\n', full_text)
        return full_text.strip()
    
    def find_article_links(self, soup: BeautifulSoup, base_url: str, site_domain: str) -> List[str]:
        """Find article links on a politics page or RSS feed"""
        links = []
        
        # Check if this is an RSS feed
        if base_url.endswith('/feed/') or 'feed' in base_url:
            return self.parse_rss_feed(soup, base_url, site_domain)
        
        selectors = self.get_selectors_for_site(base_url)
        link_selectors = selectors.get('links', self.common_selectors['links'])
        
        # Add common politics link patterns
        politics_patterns = [
            'a[href*="/politics"]',
            'a[href*="/election"]',
            'a[href*="/congress"]',
            'a[href*="/white-house"]',
            'a[href*="/government"]',
            'a[href*="/biden"]',
            'a[href*="/trump"]',
            'a[href*="/campaign"]'
        ]
        
        all_selectors = link_selectors + politics_patterns
        
        for selector in all_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if self.is_politics_article(full_url, site_domain) and full_url not in links:
                            links.append(full_url)
            except Exception as e:
                logger.debug(f"Link selector '{selector}' failed: {e}")
                continue
        
        return links[:20]  # Limit to 20 articles per page
    
    def parse_rss_feed(self, soup: BeautifulSoup, feed_url: str, site_domain: str) -> List[str]:
        """Parse RSS feed to extract article links"""
        links = []
        
        try:
            # Look for RSS items
            items = soup.find_all('item')
            if not items:
                # Try alternative RSS formats (Atom feeds)
                items = soup.find_all('entry')
            
            logger.info(f"Found {len(items)} RSS items in feed")
            
            for item in items[:30]:  # Increase limit for RSS feeds
                link_elem = item.find('link')
                title_elem = item.find('title')
                
                if link_elem:
                    link = link_elem.get_text(strip=True) if link_elem.string else link_elem.get('href')
                    title = title_elem.get_text(strip=True) if title_elem else ""
                    
                    # For RSS feeds, be more lenient with politics filtering
                    # since the feed itself is often politics-focused
                    if link and (self.is_politics_article(link, site_domain) or 
                                self.is_politics_title(title)):
                        links.append(link)
                        
        except Exception as e:
            logger.error(f"RSS parsing failed for {feed_url}: {e}")
        
        return links
    
    def is_politics_title(self, title: str) -> bool:
        """Check if article title contains politics-related keywords"""
        if not title:
            return False
            
        title_lower = title.lower()
        politics_keywords = [
            'trump', 'biden', 'congress', 'senate', 'house', 'election',
            'democrat', 'republican', 'gop', 'president', 'white house',
            'campaign', 'vote', 'politic', 'government', 'federal',
            'administration', 'capitol', 'washington'
        ]
        
        return any(keyword in title_lower for keyword in politics_keywords)
    
    def is_politics_article(self, url: str, site_domain: str) -> bool:
        """Check if URL is likely a politics article"""
        url_lower = url.lower()
        
        # Skip non-article URLs
        skip_patterns = [
            '/category/', '/tag/', '/author/', '/search/', '/about/',
            '/contact/', '/privacy/', '/terms/', '/subscribe/',
            '/newsletter/', '/rss/', '/feed/', '/video/', '/photos/',
            '/live/', '/interactive/', '/opinion/', '/editorial/',
            '/podcast/', '/radio/', '/tv/', '/weather/', '/sports/',
            '/entertainment/', '/lifestyle/', '/health/', '/tech/',
            '/business/', '/markets/', '/world/', '/international/'
        ]
        
        for pattern in skip_patterns:
            if pattern in url_lower:
                return False
        
        # Must contain politics-related keywords
        politics_keywords = [
            'politic', 'election', 'congress', 'senate', 'house',
            'white-house', 'president', 'biden', 'trump', 'campaign',
            'government', 'federal', 'washington', 'capitol', 'vote',
            'democrat', 'republican', 'gop', 'administration'
        ]
        
        for keyword in politics_keywords:
            if keyword in url_lower:
                return True
        
        return False
    
    def scrape_article(self, url: str, source_domain: str) -> Optional[PoliticsArticle]:
        """Scrape a single politics article"""
        # Early duplicate check - skip scraping if article already exists
        if self.is_duplicate(url):
            logger.debug(f"Skipping duplicate article: {url}")
            return None
        
        soup = self.get_page(url)
        if not soup:
            return None
        
        selectors = self.get_selectors_for_site(url)
        
        # Extract metadata
        title = self.extract_text_by_selectors(soup, selectors['title'])
        author = self.extract_text_by_selectors(soup, selectors['author'])
        publish_date = self.extract_text_by_selectors(soup, selectors['date'])
        
        # Extract full article text
        full_text = self.extract_full_text(soup, selectors['content'])
        
        # Extract description from meta tags if available
        description = None
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            description = meta_desc.get('content', '').strip()
        
        # Be more lenient for RSS feed articles
        min_content_length = 50 if any('feed' in u or 'rss' in u for u in [url]) else 100
        
        if not title or not full_text or len(full_text) < min_content_length:
            logger.warning(f"Insufficient content for article: {url} (title: {bool(title)}, text_len: {len(full_text) if full_text else 0})")
            return None
        
        # Clean the full text - first unicode, then boilerplate
        if full_text:
            full_text = clean_unicode_text(full_text)
            # Use advanced cleaner if available, otherwise fall back to basic
            if ADVANCED_CLEANER_AVAILABLE:
                full_text = clean_article_text_advanced(full_text)
            else:
                full_text = remove_boilerplate_text(full_text)
        
        word_count = len(full_text.split()) if full_text else 0
        
        return PoliticsArticle(
            url=url,
            title=clean_unicode_text(title) if title else title,
            author=clean_unicode_text(author) if author else author,
            publish_date=publish_date,
            full_text=full_text,
            description=clean_unicode_text(description) if description else description,
            source_domain=source_domain,
            word_count=word_count
        )
    
    def scrape_site_politics(self, site_domain: str, site_config: Dict) -> List[PoliticsArticle]:
        """Scrape politics articles from a single news site"""
        articles = []
        politics_urls = site_config['politics_urls']
        
        logger.info(f"Scraping {site_domain}")
        
        for politics_url in politics_urls:
            try:
                soup = self.get_page(politics_url)
                if not soup:
                    continue
                
                # Find article links
                article_links = self.find_article_links(soup, politics_url, site_domain)
                logger.info(f"Found {len(article_links)} articles on {politics_url}")
                
                # Scrape each article (limit to avoid overwhelming)
                for article_url in article_links[:10]:  # Max 10 articles per politics page
                    article = self.scrape_article(article_url, site_domain)
                    if article:
                        articles.append(article)
                        logger.info(f"Scraped: {article.title[:60]}...")
                    
                    time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error scraping {politics_url}: {e}")
                continue
        
        logger.info(f"Scraped {len(articles)} articles from {site_domain}")
        return articles
    
    def scrape_all_sites(self) -> List[PoliticsArticle]:
        """Scrape all configured news sites"""
        all_articles = []
        
        # Use ThreadPoolExecutor for concurrent scraping
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_site = {
                executor.submit(self.scrape_site_politics, site_domain, site_config): site_domain
                for site_domain, site_config in self.news_sites.items()
            }
            
            for future in as_completed(future_to_site):
                site_domain = future_to_site[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    logger.error(f"Error scraping {site_domain}: {e}")
        
        return all_articles
    
    def load_existing_articles(self) -> List[Dict]:
        """Load existing articles from JSON file"""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both old format (with 'articles' key) and new format (direct list)
                    if isinstance(data, list):
                        return data
                    else:
                        return data.get('articles', [])
            except Exception as e:
                logger.error(f"Error loading existing articles: {e}")
        return []
    
    def save_articles(self, new_articles: List[PoliticsArticle]):
        """Save articles to JSON file, merging with existing ones"""
        # Load existing data structure (including groups if they exist)
        existing_groups = []
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        existing_groups = data.get('groups', [])
            except Exception as e:
                logger.error(f"Error loading existing groups: {e}")
        
        existing_articles = self.load_existing_articles()
        # Use normalized URLs for duplicate checking
        existing_urls = {self.normalize_url(article.get('url', '')) for article in existing_articles}
        
        # Find the highest existing article_id
        max_id = 0
        for article in existing_articles:
            if 'article_id' in article and article['article_id'] is not None:
                max_id = max(max_id, article['article_id'])
        
        # Start assigning IDs from max_id + 1
        next_id = max_id + 1
        
        # Convert new articles to dict and filter duplicates
        new_article_dicts = []
        for article in new_articles:
            normalized_url = self.normalize_url(article.url)
            if normalized_url not in existing_urls:
                article_dict = asdict(article)
                # Add scrape field
                article_dict['scrape'] = 1
                # Add article_id (sequential)
                article_dict['article_id'] = next_id
                next_id += 1
                # Remove 'group' field if it exists (will be handled separately in groups dict)
                if 'group' in article_dict:
                    del article_dict['group']
                new_article_dicts.append(article_dict)
                # Add to existing URLs to prevent duplicates within the same batch
                existing_urls.add(normalized_url)
        
        # Remove 'group' field from existing articles if it exists
        for article in existing_articles:
            if 'group' in article:
                del article['group']
        
        # Combine all articles
        all_articles = existing_articles + new_article_dicts
        
        # Save to file with articles wrapper, groups, and ASCII encoding
        # Note: groups will be refreshed by the grouper, so we preserve existing groups here
        # Groups is an array where index = group_id
        output_data = {
            "articles": all_articles,
            "groups": existing_groups  # Preserve existing groups array (will be refreshed by grouper)
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=True)
        
        logger.info(f"Saved {len(new_article_dicts)} new articles. Total: {len(all_articles)}")
        if new_article_dicts:
            logger.info(f"Assigned article IDs: {new_article_dicts[0]['article_id']} to {new_article_dicts[-1]['article_id']}")
        
        # Update cache with new URLs
        if self._cache_loaded and new_article_dicts:
            for article_dict in new_article_dicts:
                self._existing_urls_cache.add(self.normalize_url(article_dict['url']))
    
    def run_scraping_cycle(self):
        """Run a complete scraping cycle"""
        logger.info("Starting scraping cycle...")
        start_time = time.time()
        
        try:
            # Refresh the URL cache at the start of each cycle
            self._cache_loaded = False
            
            # Track how many articles before saving
            existing_count = len(self.load_existing_articles())
            articles = self.scrape_all_sites()
            self.save_articles(articles)
            new_count = len(self.load_existing_articles())
            new_articles_added = new_count - existing_count
            
            # Only run full pipeline if new articles were added
            if new_articles_added > 0:
                logger.info(f"Running full pipeline for {new_articles_added} new articles...")
                
                # Step 1: Group articles
                try:
                    from grouper import add_groups_to_articles
                    add_groups_to_articles(
                        str(self.output_file), 
                        threshold=0.80,  # Tightened for more focused topic groups
                        allow_multi_group=True
                    )
                    logger.info("✓ Article grouping completed")
                except Exception as e:
                    logger.error(f"Error running grouper: {e}")
                    return
                
                # Step 2: Extract quotes/sentences from new articles
                try:
                    from create_article_object import sentences_quotes
                    sentences_quotes(str(self.output_file), process_all=False)
                    logger.info("✓ Quote/sentence extraction completed")
                except Exception as e:
                    logger.error(f"Error extracting quotes/sentences: {e}")
                
                # Step 3: Extract factbanks
                try:
                    from extraction_llm import extract_all_factbanks
                    extract_all_factbanks(str(self.output_file))
                    logger.info("✓ Factbank extraction completed")
                except Exception as e:
                    logger.error(f"Error extracting factbanks: {e}")
                    return
                
                # Step 4: Generate briefs (only if factbanks changed)
                try:
                    from brief import generate_all_briefs
                    generate_all_briefs(str(self.output_file))
                    logger.info("✓ Brief generation completed")
                except Exception as e:
                    logger.error(f"Error generating briefs: {e}")
                
                logger.info("✅ Full pipeline completed successfully")
            else:
                logger.info("No new articles added, skipping pipeline")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Scraping cycle completed in {elapsed_time:.2f} seconds")
            logger.info(f"New articles: {new_articles_added}, Total articles: {new_count}")
            
        except Exception as e:
            logger.error(f"Error in scraping cycle: {e}")
    
    def start_scheduled_scraping(self):
        """Start the scheduled scraping (every hour)"""
        logger.info("Starting scheduled scraping (every hour)...")
        
        # Run initial scraping
        self.run_scraping_cycle()
        
        # Schedule hourly runs
        schedule.every().hour.do(self.run_scraping_cycle)
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def main():
    """Main function to run the scraper"""
    scraper = USPoliticsScraper("/Users/arav/Documents/GitHub/backend/src/core/metrics/civai_bias/data/news_data.json")
    
    print("US Politics News Scraper")
    print("=" * 50)
    print(f"Configured {len(scraper.news_sites)} news sources")
    print(f"Text Cleaner: {'Advanced NLP Cleaner ✓' if ADVANCED_CLEANER_AVAILABLE else 'Basic Regex Cleaner'}")
    print("Sources and bias scores:")
    for domain, config in scraper.news_sites.items():
        print(f"  {domain}: {len(config['politics_urls'])} URLs")
    print()
    
    choice = input("Choose mode:\n1. Run once\n2. Run every hour (scheduled)\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        scraper.run_scraping_cycle()
        print(f"\nScraping complete! Check '{scraper.output_file}' for results.")
    elif choice == "2":
        print("Starting scheduled scraping (every hour)...")
        print("Press Ctrl+C to stop")
        try:
            scraper.start_scheduled_scraping()
        except KeyboardInterrupt:
            print("\nScraping stopped by user")
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
