import feedparser
import logging
import sys
import time
import re
import json
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------------
# RSS FEEDS (EXTENSIVE)
# ---------------------------
RSS_FEEDS = {
    "US Politics": [
        "https://rss.cnn.com/rss/cnn_allpolitics.rss",
        "https://www.foxnews.com/politics/rss",
        "https://feeds.npr.org/1004/rss.xml",
        "https://www.politico.com/rss/politics08.xml",
        "https://thehill.com/feed/",
        "https://www.nbcnews.com/politics/rss.xml",
        "https://www.cbsnews.com/latest/rss/politics",
        "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
        "https://feeds.washingtonpost.com/rss/politics",
        "https://www.huffpost.com/section/politics/feed",
        "https://www.usatoday.com/rss/news/politics.xml",
        "https://abcnews.go.com/abcnews/politicsheadlines",
        "https://www.reuters.com/rssFeed/politicsNews",
        "https://www.breitbart.com/politics/feed/",
        "https://www.nationalreview.com/feed/",
        "https://www.cnn.com/specials/politics/rss"
    ],
    "World Politics": [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.reuters.com/rssFeed/worldNews",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "https://www.theguardian.com/world/rss",
        "https://rss.dw.com/xml/rss-en-all",
        "https://www.france24.com/en/rss",
        "https://apnews.com/hub/apf-topnews?format=rss",
        "https://www.voanews.com/rss",
        "https://www.npr.org/rss/rss.php?id=1001",
        "https://www.cnn.com/specials/world/rss"
    ],
    "US Economy & Business": [
        "https://www.wsj.com/xml/rss/3_7014.xml",
        "https://www.bloomberg.com/feed/podcast/etf-report.xml",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://www.forbes.com/business/feed2/",
        "https://www.reuters.com/rssFeed/businessNews"
    ],
    "Tech & Science": [
        "https://www.wired.com/feed/rss",
        "https://www.theverge.com/rss/index.xml",
        "https://www.cnet.com/rss/news/",
        "https://www.scientificamerican.com/feed/rss/",
        "https://www.nature.com/subjects/news/rss"
    ]
}

SOURCE_BIAS = {
    "CNN": "Left",
    "Fox News": "Right",
    "NYT": "Left",
    "Washington Post": "Left",
    "Politico": "Center",
    "The Hill": "Center",
    "NBC News": "Center",
    "CBS News": "Center",
    "Reuters": "Center",
    "HuffPost": "Left",
    "USA Today": "Center",
    "ABC News": "Center",
    "BBC": "Center",
    "Al Jazeera": "Center",
    "The Guardian": "Left",
    "DW": "Center",
    "France24": "Center",
    "AP": "Center",
    "NPR": "Center",
    "VOA": "Center",
    "Breitbart": "Right",
    "National Review": "Right",
    "WSJ": "Right",
    "Bloomberg": "Center",
    "CNBC": "Center",
    "Forbes": "Right",
    "Wired": "Center",
    "The Verge": "Center",
    "CNET": "Center",
    "Scientific American": "Center",
    "Nature": "Center"
}

# ---------------------------
# LOGGING CONFIG
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------------------
# CONTENT FILTER (copied & slightly adapted)
# ---------------------------

def filter_content(text):
    """Filter out unwanted content while keeping the main article text"""
    if not text:
        return ""
    
    lines = text.split('\n')
    filtered_lines = []
    
    unwanted_keywords = [
        'subscribe', 'newsletter', 'sign up', 'log in', 'register',
        'advertisement', 'advertise', 'sponsored', 'promoted',
        'cookie', 'privacy', 'terms', 'conditions',
        'follow us', 'share', 'comment', 'comments',
        'related', 'recommended', 'popular', 'trending',
        'most read', 'also on', 'more from', 'you might also like',
        'breaking news', 'live updates', 'updated', 'published',
        'menu', 'navigation', 'search', 'breadcrumb',
        'footer', 'header', 'sidebar', 'aside',
        'loading', 'please wait', 'click here', 'read more',
        'continue reading', 'full story', 'full article',
        '©', 'all rights reserved', 'contact us', 'about us',
        'home', 'back to top', 'top stories', 'latest news'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 20:
            continue
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in unwanted_keywords):
            continue
        if re.match(r'^(by|written by|author:|reporter:|journalist:)\s*[A-Za-z\s]+$', line, re.IGNORECASE):
            continue
        if re.match(r'^\d{1,2}:\d{2}\s*(AM|PM|am|pm)?$', line):
            continue
        if re.match(r'^[A-Z][A-Za-z\s\.\-]{1,60}\s*—?\s*$', line) and len(line) < 60:
            # likely publication line like "Reuters —"
            continue
        filtered_lines.append(line)
    
    filtered_text = '\n'.join(filtered_lines)
    filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text)
    return filtered_text.strip()

# ---------------------------
# SELENIUM DRIVER SETUP & EXTRACTOR (reuses a single driver)
# ---------------------------

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # works with newer Chrome; change to "--headless" if older
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    # hide webdriver property
    try:
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    except:
        pass
    return driver

def extract_article_text(driver, url, wait_time=3):
    """Given an existing Selenium driver, return (title, author, text)."""
    title = ""
    author = ""
    text = ""
    try:
        driver.get(url)
        time.sleep(wait_time)

        # --- TITLE ---
        title_selectors = [
            "h1",
            "[class*='title']",
            "[class*='headline']",
            "title",
            "h1[class*='title']",
            "h1[class*='headline']"
        ]
        for selector in title_selectors:
            try:
                title_element = driver.find_element(By.CSS_SELECTOR, selector)
                title = title_element.text.strip()
                if title:
                    break
            except:
                continue
        
        # If no title found, try to get it from page title
        if not title:
            title = driver.title

        # --- AUTHOR ---
        author_selectors = [
            "[class*='author']",
            "[class*='byline']",
            "[class*='writer']",
            "[class*='reporter']",
            "[class*='journalist']",
            ".author",
            ".byline",
            ".writer",
            ".reporter",
            "span[class*='author']",
            "div[class*='author']",
            "p[class*='author']",
            "[rel='author']",
            "a[rel='author']",
            "[class*='by']",
            "span[class*='by']",
            "div[class*='by']"
        ]
        
        for selector in author_selectors:
            try:
                author_element = driver.find_element(By.CSS_SELECTOR, selector)
                author = author_element.text.strip()
                if author and len(author) > 2:
                    # Clean up common author prefixes
                    author = author.replace('By ', '').replace('by ', '').replace('BY ', '')
                    author = author.replace('Written by ', '').replace('written by ', '')
                    author = author.replace('Author: ', '').replace('author: ', '')
                    author = author.replace('Reporter: ', '').replace('reporter: ', '')
                    author = author.replace('Journalist: ', '').replace('journalist: ', '')
                    break
            except:
                continue

        # --- MAIN CONTENT ---
        content_selectors = [
            "[class*='article']",
            "[class*='content']",
            "[class*='story']",
            "article",
            "main",
            "[role='main']",
            ".article-body",
            ".story-body",
            ".content-body"
        ]
        
        for selector in content_selectors:
            try:
                content_element = driver.find_element(By.CSS_SELECTOR, selector)
                text = content_element.text.strip()
                if len(text) > 100:  # Make sure we got substantial content
                    break
            except:
                continue
        
        # If no specific content found, try to get all text from body
        if not text or len(text) < 100:
            try:
                body = driver.find_element(By.TAG_NAME, "body")
                text = body.text.strip()
            except:
                text = "Could not extract content"
        
        # Filter the content to remove unwanted elements
        text = filter_content(text)
        
        return title.strip(), author.strip(), text.strip()

    except Exception as e:
        logger.exception("Error extracting article at %s: %s", url, e)
        return (title or "Error occurred"), "", f"Failed to extract content: {str(e)}"

# ---------------------------
# HELPER: SAVE LINKS TO JSON
# ---------------------------

def save_links_to_json(links, filename="scrapedLinks.json"):
    """Save scraped links to a JSON file"""
    try:
        links_data = {
            "timestamp": datetime.now().isoformat(),
            "total_links": len(links),
            "links": links
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(links_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(links)} links to {filename}")
    except Exception as e:
        logger.error(f"Failed to save links to {filename}: {e}")

# ---------------------------
# HELPER: SOURCE NAME EXTRACTION
# ---------------------------

def extract_source_name(feed_url):
    feed_url = feed_url.lower()
    for key in SOURCE_BIAS:
        if key.lower() in feed_url:
            return key
    # fallback heuristics
    if "cnn" in feed_url:
        return "CNN"
    if "reuters" in feed_url:
        return "Reuters"
    if "bbc" in feed_url:
        return "BBC"
    return "Unknown"

# ---------------------------
# MAIN FETCHING LOOP
# ---------------------------

def fetch_rss_articles(max_articles=200, per_feed_wait=0.5):
    all_articles = []
    scraped_links = []
    driver = None
    try:
        logger.info("Starting driver...")
        driver = setup_driver()
        count = 0
        for category, feeds in RSS_FEEDS.items():
            logger.info(f"Checking category: {category}")
            for feed_url in feeds:
                feed = feedparser.parse(feed_url)
                if not feed.entries:
                    logger.warning(f"No entries found in feed: {feed_url}")
                for entry in feed.entries:
                    if count >= max_articles:
                        logger.info(f"Reached maximum of {max_articles} articles. Stopping.")
                        # Save links before returning
                        save_links_to_json(scraped_links)
                        return all_articles

                    source = extract_source_name(feed_url)
                    bias = SOURCE_BIAS.get(source, "Unknown")
                    link = getattr(entry, 'link', getattr(entry, 'id', 'No link'))
                    title_from_feed = getattr(entry, 'title', None)

                    # Add link to scraped_links list
                    link_info = {
                        "url": link,
                        "source": source,
                        "bias": bias,
                        "title_from_feed": title_from_feed,
                        "category": category,
                        "scraped_at": datetime.now().isoformat()
                    }
                    scraped_links.append(link_info)

                    # extract text, author & more robust title using Selenium
                    try:
                        title_extracted, author, text = extract_article_text(driver, link)
                        # prefer cleaned/extracted title if it's sensible, else fallback to feed title
                        final_title = title_extracted if title_extracted and len(title_extracted) > 5 else (title_from_feed or "No title")
                    except Exception as e:
                        logger.exception("Failed to extract link %s : %s", link, e)
                        final_title = title_from_feed or "No title"
                        author = ""
                        text = f"Extraction failed: {str(e)}"

                    article_data = {
                        "source": source,
                        "bias": bias,
                        "articleLink": link,
                        "articleTitle": final_title,
                        "author": author,
                        "text": text
                    }
                    all_articles.append(article_data)
                    count += 1

                    # Log and print in requested format as fetched
                    logger.info(f"Fetched article ({count}): {final_title[:80]} | Source: {source} | Bias: {bias}")
                    print("\n" + "-"*80)
                    print(f"source: {source}")
                    print(f"bias: {bias}")
                    print(f"title: {final_title}")
                    if author:
                        print(f"author: {author}")
                    print(f"link: {link}")
                    print("text:")
                    print(text)
                    print("-"*80 + "\n")

                    time.sleep(per_feed_wait)
        
        # Save links after processing all articles
        save_links_to_json(scraped_links)
        return all_articles
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# ---------------------------
# RUN
# ---------------------------

if __name__ == "__main__":
    articles = fetch_rss_articles(max_articles=200)
    logger.info(f"Total articles fetched: {len(articles)}")