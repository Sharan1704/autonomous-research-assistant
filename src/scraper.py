import logging
import time

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Tags that reliably carry article / document content.
CONTENT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote"]

# Tags whose entire subtree should be discarded before extraction.
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]

# Minimum delay between successive requests to the same host (seconds).
_REQUEST_DELAY_S = 1.0


class WebScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AutonomousResearchAgent/1.0; "
                "+https://github.com/your-repo)"
            )
        }
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce a minimum delay between requests to avoid hammering servers."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _REQUEST_DELAY_S:
            sleep_for = _REQUEST_DELAY_S - elapsed
            logger.debug("Rate-limiting: sleeping %.2fs before next request.", sleep_for)
            time.sleep(sleep_for)

    def scrape_url(self, url: str) -> Document | None:
        """Scrape a URL and return a clean Document object.

        Note on robots.txt:
            This scraper does not currently fetch or respect robots.txt.
            Always verify that scraping a site is permitted by its terms of
            service before use.  A _REQUEST_DELAY_S delay is applied between
            successive requests as a basic courtesy measure.
        """
        try:
            self._rate_limit()
            logger.info("Scraping URL: %s", url)
            response = requests.get(url, headers=self.headers, timeout=10)
            self._last_request_time = time.time()
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # 1. Remove noisy structural elements entirely.
            for tag in soup(NOISE_TAGS):
                tag.decompose()

            # 2. Extract text only from meaningful content tags.
            content_elements = soup.find_all(CONTENT_TAGS)
            lines = [el.get_text(separator=" ", strip=True) for el in content_elements]
            lines = [line for line in lines if len(line) > 20]
            text = "\n".join(lines)

            if not text.strip():
                # Fallback for JS-heavy pages where content is injected at runtime.
                text = soup.get_text(separator=" ", strip=True)

            title = (
                soup.title.string.strip()
                if soup.title and soup.title.string
                else "Unknown Title"
            )

            logger.info("Scraped '%s' — %d chars extracted.", title, len(text))
            return Document(
                page_content=text,
                metadata={"source": url, "title": title, "type": "web_scrape"},
            )

        except Exception as e:
            logger.error("Error scraping '%s': %s", url, e)
            return None

    def scrape_multiple_urls(self, urls: list[str]) -> dict:
        """Scrape multiple URLs and return a results summary.

        Args:
            urls: List of URLs to scrape.

        Returns:
            A dict with:
            - ``documents``: list of successfully scraped Document objects.
            - ``succeeded``: list of URLs that were scraped successfully.
            - ``failed``: list of URLs that failed (network error, empty content, etc.).
        """
        documents = []
        succeeded = []
        failed = []

        for url in urls:
            url = url.strip()
            if not url:
                continue
            doc = self.scrape_url(url)
            if doc:
                documents.append(doc)
                succeeded.append(url)
            else:
                failed.append(url)

        logger.info(
            "Batch scrape complete — %d succeeded, %d failed.",
            len(succeeded),
            len(failed),
        )
        return {"documents": documents, "succeeded": succeeded, "failed": failed}