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
    bias_score: float = 0.0
    category: str = "politics"
    word_count: int = 0
    
    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.now(timezone.utc).isoformat()
        if self.source_domain == "":
            self.source_domain = urlparse(self.url).netloc

class USPoliticsScraper:
    """Scraper specifically for US Politics news from specified sources"""
    
    def __init__(self, output_file: str = "us_politics_news.json"):
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
        
        # News sites with bias scores and politics URLs
        self.news_sites = {
            # Left-leaning sites
            "apnews.com": {
                "bias_score": -2.47,
                "politics_urls": [
                    "https://apnews.com/hub/politics",
                    "https://apnews.com/hub/election-2024",
                    "https://apnews.com/hub/congress"
                ]
            },
            "theguardian.com/us": {
                "bias_score": -8.12,
                "politics_urls": [
                    "https://www.theguardian.com/us-news/us-politics",
                    "https://www.theguardian.com/us-news/us-congress",
                    "https://www.theguardian.com/us-news/trump-administration"
                ]
            },
            "msnbc.com": {
                "bias_score": -13.87,
                "politics_urls": [
                    "https://www.msnbc.com/politics",
                ]
            },
            "theatlantic.com": {
                "bias_score": -9.33,
                "politics_urls": [
                    "https://www.theatlantic.com/politics/",
                    "https://www.theatlantic.com/ideas/",
                ]
            },
            "vox.com": {
                "bias_score": -9.99,
                "politics_urls": [
                    "https://www.vox.com/policy-and-politics",
                    "https://www.vox.com/politics",
                ]
            },
            
            # Center/Mainstream sites
            "abcnews.go.com": {
                "bias_score": -3.10,
                "politics_urls": [
                    "https://abcnews.go.com/Politics",
                ]
            },
            "cnn.com": {
                "bias_score": -6.24,
                "politics_urls": [
                    "https://www.cnn.com/politics",
                    "https://www.cnn.com/election/2024",
                ]
            },
            "nytimes.com": {
                "bias_score": -8.04,
                "politics_urls": [
                    "https://www.nytimes.com/section/politics",
                    "https://www.nytimes.com/section/us/politics",
                ]
            },
            "nbcnews.com": {
                "bias_score": -5.61,
                "politics_urls": [
                    "https://www.nbcnews.com/politics",
                    "https://www.nbcnews.com/politics/congress",
                ]
            },
            "cbsnews.com": {
                "bias_score": -3.03,
                "politics_urls": [
                    "https://www.cbsnews.com/politics/",
                    "https://www.cbsnews.com/election-2024/",
                ]
            },
            "cnbc.com": {
                "bias_score": -1.89,
                "politics_urls": [
                    "https://www.cnbc.com/politics/",
                ]
            },
            "npr.org": {
                "bias_score": -4.32,
                "politics_urls": [
                    "https://www.npr.org/sections/politics/",
                    "https://www.npr.org/sections/congress/",
                ]
            },
            "time.com": {
                "bias_score": -7.43,
                "politics_urls": [
                    "https://time.com/section/politics/",
                ]
            },
            "washingtonpost.com": {  # FIXED: Using RSS feed to bypass anti-scraping
                "bias_score": -7.16,
                "politics_urls": [
                    "https://feeds.washingtonpost.com/rss/politics",
                ]
            },
            
            # Center-right sites
            "bbc.com": {
                "bias_score": -1.34,
                "politics_urls": [
                    "https://www.bbc.com/news/world/us_and_canada",
                    "https://www.bbc.com/news/election/us2024",
                ]
            },
            "reuters.com": {
                "bias_score": -1.27,
                "politics_urls": [
                    "https://www.reuters.com/world/us/",
                    "https://www.reuters.com/world/us/politics/",
                ]
            },
            "newsweek.com": {  # FIXED: Using direct politics page and RSS
                "bias_score": -1.77,
                "politics_urls": [
                    "https://www.newsweek.com/politics",
                    "https://www.newsweek.com/rss",
                ]
            },
            "san.com": {
                "bias_score": 0.92,
                "politics_urls": [
                    "https://www.san.com/politics/",
                    "https://www.san.com/news/politics/",
                ]
            },
            "thehill.com": {  # FIXED: Using RSS feeds to bypass 403 errors
                "bias_score": -1.46,
                "politics_urls": [
                    "https://thehill.com/feed/",
                    "https://thehill.com/policy/feed/",
                ]
            },
            "newsnationnow.com": {
                "bias_score": 0.54,
                "politics_urls": [
                    "https://www.newsnationnow.com/politics/",
                    "https://www.newsnationnow.com/us-news/",
                ]
            },
            "reason.com": {  # FIXED: Using working URLs
                "bias_score": 4.66,
                "politics_urls": [
                    "https://reason.com/tag/politics/",
                    "https://reason.com/",
                ]
            },
            
            # Right-leaning sites
            "theepochtimes.com": {
                "bias_score": 8.43,
                "politics_urls": [
                    "https://www.theepochtimes.com/us/politics",
                ]
            },
            "justthenews.com": {
                "bias_score": 10.90,
                "politics_urls": [
                    "https://justthenews.com/politics-policy",
                    "https://justthenews.com/government",
                ]
            },
            "nypost.com": {
                "bias_score": 9.37,
                "politics_urls": [
                    "https://nypost.com/news/politics/",
                ]
            },
            "washingtontimes.com": {
                "bias_score": 11.61,
                "politics_urls": [
                    "https://www.washingtontimes.com/news/politics/",
                ]
            },
            "thedispatch.com": {  # FIXED: Using main page with politics filtering
                "bias_score": 5.50,
                "politics_urls": [
                    "https://thedispatch.com/",
                ]
            },
            "foxnews.com": {
                "bias_score": 11.05,
                "politics_urls": [
                    "https://www.foxnews.com/politics",
                    "https://www.foxnews.com/category/politics/elections",
                ]
            },
            "ijr.com": {
                "bias_score": 12.68,
                "politics_urls": [
                    "https://ijr.com/politics/",
                    "https://ijr.com/",
                ]
            },
            # "newsmax.com": {  # DISABLED: Connection issues
            #     "bias_score": 13.49,
            #     "politics_urls": [
            #         "https://www.newsmax.com/politics/",
            #     ]
            # },
            "theblaze.com": {
                "bias_score": 13.52,
                "politics_urls": [
                    "https://www.theblaze.com/politics",
                    "https://www.theblaze.com/news/politics",
                ]
            },
            "freebeacon.com": {
                "bias_score": 14.17,
                "politics_urls": [
                    "https://freebeacon.com/politics/",
                    "https://freebeacon.com/",
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
            'a[href*="/trump"]',
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
    
    def scrape_article(self, url: str, bias_score: float, source_domain: str) -> Optional[PoliticsArticle]:
        """Scrape a single politics article"""
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
        
        word_count = len(full_text.split())
        
        return PoliticsArticle(
            url=url,
            title=title,
            author=author,
            publish_date=publish_date,
            full_text=full_text,
            description=description,
            source_domain=source_domain,
            bias_score=bias_score,
            word_count=word_count
        )
    
    def scrape_site_politics(self, site_domain: str, site_config: Dict) -> List[PoliticsArticle]:
        """Scrape politics articles from a single news site"""
        articles = []
        bias_score = site_config['bias_score']
        politics_urls = site_config['politics_urls']
        
        logger.info(f"Scraping {site_domain} (bias: {bias_score})")
        
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
                    article = self.scrape_article(article_url, bias_score, site_domain)
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
                    return data.get('articles', [])
            except Exception as e:
                logger.error(f"Error loading existing articles: {e}")
        return []
    
    def save_articles(self, new_articles: List[PoliticsArticle]):
        """Save articles to JSON file, merging with existing ones"""
        existing_articles = self.load_existing_articles()
        existing_urls = {article.get('url') for article in existing_articles}
        
        # Convert new articles to dict and filter duplicates
        new_article_dicts = []
        for article in new_articles:
            if article.url not in existing_urls:
                new_article_dicts.append(asdict(article))
        
        # Combine all articles
        all_articles = existing_articles + new_article_dicts
        
        # Create output data
        output_data = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'total_articles': len(all_articles),
            'new_articles_this_run': len(new_article_dicts),
            'sources': list(self.news_sites.keys()),
            'articles': all_articles
        }
        
        # Save to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(new_article_dicts)} new articles. Total: {len(all_articles)}")
    
    def run_scraping_cycle(self):
        """Run a complete scraping cycle"""
        logger.info("Starting scraping cycle...")
        start_time = time.time()
        
        try:
            articles = self.scrape_all_sites()
            self.save_articles(articles)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Scraping cycle completed in {elapsed_time:.2f} seconds")
            logger.info(f"Scraped {len(articles)} new articles")
            
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
    scraper = USPoliticsScraper("us_politics_news.json")
    
    print("US Politics News Scraper")
    print("=" * 50)
    print(f"Configured {len(scraper.news_sites)} news sources")
    print("Sources and bias scores:")
    for domain, config in scraper.news_sites.items():
        print(f"  {domain}: {config['bias_score']}")
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
