import numpy as np
import re
import math
from urllib.parse import urlparse
from collections import Counter
import tldextract
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class URLFeatureExtractor:
    """
    Extracts 60 features from URLs for phishing detection.
    Returns feature array in exact order as specified.
    """
    
    def __init__(self, timeout=5):
        """
        Initialize the feature extractor.
        
        Args:
            timeout (int): Timeout for web requests in seconds
        """
        self.timeout = timeout
        self.banking_keywords = ['bank', 'account', 'banking', 'secure', 'verify', 'login']
        self.payment_keywords = ['pay', 'payment', 'paypal', 'transaction', 'card', 'credit']
        self.crypto_keywords = ['crypto', 'bitcoin', 'blockchain', 'wallet', 'coin']
        self.copyright_keywords = ['copyright', '©', 'all rights reserved']
    
    def extract_features(self, url):
        """
        Extract all 60 features from a URL.
        
        Args:
            url (str): The URL to extract features from
            
        Returns:
            list: Array of 60 feature values in exact order
        """
        # Parse URL
        parsed = urlparse(url)
        extracted = tldextract.extract(url)
        
        # Fetch web content
        html_content = self._fetch_content(url)
        soup = BeautifulSoup(html_content, 'html.parser') if html_content else None
        
        # Extract features in exact order
        features = [
            self._length_of_url(url),                                    # 1. lengthofurl
            self._url_complexity(url),                                   # 2. urlcomplexity
            self._character_complexity(url),                             # 3. charactercomplexity
            self._domain_length(extracted),                              # 4. domainlengthofurl
            self._is_domain_ip(parsed.netloc),                          # 5. isdomainip
            self._tld_length(extracted),                                # 6. tldlength
            self._letter_count_in_url(url),                             # 7. lettercntinurl
            self._url_letter_ratio(url),                                # 8. urlletterratio
            self._digit_count_in_url(url),                              # 9. digitcntinurl
            self._url_digit_ratio(url),                                 # 10. urldigitratio
            self._equal_char_count(url),                                # 11. equalcharcntinurl
            self._question_mark_count(url),                             # 12. quesmarkcntinurl
            self._ampersand_count(url),                                 # 13. ampcharcntinurl
            self._other_special_char_count(url),                        # 14. otherspclcharcntinurl
            self._url_other_special_char_ratio(url),                    # 15. urlotherspclcharratio
            self._hashtag_count(url),                                   # 16. numberofhashtags
            self._number_of_subdomains(extracted),                      # 17. numberofsubdomains
            self._having_path(parsed),                                  # 18. havingpath
            self._path_length(parsed),                                  # 19. pathlength
            self._having_query(parsed),                                 # 20. havingquery
            self._having_fragment(parsed),                              # 21. havingfragment
            self._having_anchor(parsed),                                # 22. havinganchor
            self._has_ssl(parsed),                                      # 23. hasssl
            self._is_unreachable(url),                                  # 24. isunreachable
            self._line_of_code(html_content),                           # 25. lineofcode
            self._longest_line_length(html_content),                    # 26. longestlinelength
            self._has_title(soup),                                      # 27. hastitle
            self._has_favicon(soup),                                    # 28. hasfavicon
            self._has_robots_blocked(url),                              # 29. hasrobotsblocked
            self._is_responsive(soup),                                  # 30. isresponsive
            self._is_url_redirects(url),                                # 31. isurlredirects
            self._is_self_redirects(url, parsed.netloc),                # 32. isselfredirects
            self._has_description(soup),                                # 33. hasdescription
            self._has_popup(html_content),                              # 34. haspopup
            self._has_iframe(soup),                                     # 35. hasiframe
            self._is_form_submit_external(soup, parsed.netloc),         # 36. isformsubmitexternal
            self._has_social_media(soup),                               # 37. hassocialmediapage
            self._has_submit_button(soup),                              # 38. hassubmitbutton
            self._has_hidden_fields(soup),                              # 39. hashiddenfields
            self._has_password_fields(soup),                            # 40. haspasswordfields
            self._has_banking_keywords(url, html_content),              # 41. hasbankingkey
            self._has_payment_keywords(url, html_content),              # 42. haspaymentkey
            self._has_crypto_keywords(url, html_content),               # 43. hascryptokey
            self._has_copyright_keywords(html_content),                 # 44. hascopyrightinfokey
            self._count_images(soup),                                   # 45. cntimages
            self._count_css_files(soup),                                # 46. cntfilescss
            self._count_js_files(soup),                                 # 47. cntfilesjs
            self._count_self_href(soup, parsed.netloc),                 # 48. cntselfhref
            self._count_empty_ref(soup),                                # 49. cntemptyref
            self._count_external_ref(soup, parsed.netloc),              # 50. cntexternalref
            self._count_popup(html_content),                            # 51. cntpopup
            self._count_iframe(soup),                                   # 52. cntiframe
            self._unique_feature_count(url),                            # 53. uniquefeaturecnt
            self._wap_legitimate_prob(url),                             # 54. waplegitimate
            self._wap_phishing_prob(url),                               # 55. wapphishing
            self._shannon_entropy(url),                                 # 56. shannonentropy
            self._fractal_dimension(url),                               # 57. fractaldimension
            self._kolmogorov_complexity(url),                           # 58. kolmogorovcomplexity
            self._hex_pattern_count(url),                               # 59. hexpatterncnt
            self._base64_pattern_count(url),                            # 60. base64patterncnt
            self._likeliness_index(url)                                 # 61. likelinessindex
        ]
        
        return features
    
    # ==================== Helper Methods ====================
    
    def _fetch_content(self, url):
        """Fetch HTML content from URL"""
        try:
            response = requests.get(url, timeout=self.timeout, verify=False, 
                                    headers={'User-Agent': 'Mozilla/5.0'})
            return response.text if response.status_code == 200 else None
        except:
            return None
    
    # ==================== Feature Extraction Methods ====================
    
    def _length_of_url(self, url):
        """Length of the URL"""
        return len(url)
    
    def _url_complexity(self, url):
        """URL complexity based on unique characters / total length"""
        if len(url) == 0:
            return 0.0
        unique_chars = len(set(url))
        return round(unique_chars / len(url), 6)
    
    def _character_complexity(self, url):
        """Character complexity: entropy of character distribution"""
        if not url:
            return 0.0
        char_freq = Counter(url)
        total = len(url)
        entropy = -sum((count/total) * math.log2(count/total) for count in char_freq.values())
        return round(entropy, 6)
    
    def _domain_length(self, extracted):
        """Length of domain"""
        domain = extracted.domain + '.' + extracted.suffix
        return len(domain)
    
    def _is_domain_ip(self, netloc):
        """Check if domain is an IP address"""
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return 1 if re.match(ip_pattern, netloc.split(':')[0]) else 0
    
    def _tld_length(self, extracted):
        """Length of TLD"""
        return len(extracted.suffix)
    
    def _letter_count_in_url(self, url):
        """Count of letters in URL"""
        return sum(c.isalpha() for c in url)
    
    def _url_letter_ratio(self, url):
        """Ratio of letters to total characters"""
        return round(self._letter_count_in_url(url) / len(url), 6) if len(url) > 0 else 0.0
    
    def _digit_count_in_url(self, url):
        """Count of digits in URL"""
        return sum(c.isdigit() for c in url)
    
    def _url_digit_ratio(self, url):
        """Ratio of digits to total characters"""
        return round(self._digit_count_in_url(url) / len(url), 6) if len(url) > 0 else 0.0
    
    def _equal_char_count(self, url):
        """Count of equal signs in URL"""
        return url.count('=')
    
    def _question_mark_count(self, url):
        """Count of question marks in URL"""
        return url.count('?')
    
    def _ampersand_count(self, url):
        """Count of ampersands in URL"""
        return url.count('&')
    
    def _other_special_char_count(self, url):
        """Count of other special characters"""
        special_chars = set('!@#$%^*()_+-[]{}|;:,.<>/')
        return sum(c in special_chars for c in url)
    
    def _url_other_special_char_ratio(self, url):
        """Ratio of other special characters"""
        return round(self._other_special_char_count(url) / len(url), 6) if len(url) > 0 else 0.0
    
    def _hashtag_count(self, url):
        """Count of hashtags in URL"""
        return url.count('#')
    
    def _number_of_subdomains(self, extracted):
        """Number of subdomains"""
        subdomain = extracted.subdomain
        return len(subdomain.split('.')) if subdomain else 0
    
    def _having_path(self, parsed):
        """Check if URL has a path"""
        return 1 if parsed.path and parsed.path != '/' else 0
    
    def _path_length(self, parsed):
        """Length of URL path"""
        return len(parsed.path) if parsed.path else 0
    
    def _having_query(self, parsed):
        """Check if URL has query parameters"""
        return 1 if parsed.query else 0
    
    def _having_fragment(self, parsed):
        """Check if URL has fragment"""
        return 1 if parsed.fragment else 0
    
    def _having_anchor(self, parsed):
        """Check if URL has anchor (same as fragment)"""
        return 1 if parsed.fragment else 0
    
    def _has_ssl(self, parsed):
        """Check if URL uses HTTPS"""
        return 1 if parsed.scheme == 'https' else 0
    
    def _is_unreachable(self, url):
        """Check if URL is unreachable"""
        try:
            response = requests.head(url, timeout=self.timeout, verify=False,
                                    headers={'User-Agent': 'Mozilla/5.0'})
            return 0 if response.status_code < 400 else 1
        except:
            return 1
    
    def _line_of_code(self, html_content):
        """Count lines of code in HTML"""
        return len(html_content.split('\n')) if html_content else 0
    
    def _longest_line_length(self, html_content):
        """Length of longest line in HTML"""
        if not html_content:
            return 0
        lines = html_content.split('\n')
        return max(len(line) for line in lines) if lines else 0
    
    def _has_title(self, soup):
        """Check if page has title tag"""
        if not soup:
            return 0
        return 1 if soup.find('title') else 0
    
    def _has_favicon(self, soup):
        """Check if page has favicon"""
        if not soup:
            return 0
        favicon = soup.find('link', rel=lambda x: x and 'icon' in str(x).lower())
        return 1 if favicon else 0
    
    def _has_robots_blocked(self, url):
        """Check if robots.txt blocks crawling"""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            response = requests.get(robots_url, timeout=self.timeout, verify=False)
            return 1 if 'Disallow' in response.text else 0
        except:
            return 0
    
    def _is_responsive(self, soup):
        """Check if page is responsive (has viewport meta tag)"""
        if not soup:
            return 0
        viewport = soup.find('meta', attrs={'name': 'viewport'})
        return 1 if viewport else 0
    
    def _is_url_redirects(self, url):
        """Check if URL redirects"""
        try:
            response = requests.get(url, timeout=self.timeout, verify=False, 
                                   allow_redirects=False,
                                   headers={'User-Agent': 'Mozilla/5.0'})
            return 1 if response.status_code in [301, 302, 303, 307, 308] else 0
        except:
            return 0
    
    def _is_self_redirects(self, url, netloc):
        """Check if URL redirects to same domain"""
        try:
            response = requests.get(url, timeout=self.timeout, verify=False,
                                   headers={'User-Agent': 'Mozilla/5.0'})
            final_netloc = urlparse(response.url).netloc
            return 1 if final_netloc == netloc else 0
        except:
            return 0
    
    def _has_description(self, soup):
        """Check if page has meta description"""
        if not soup:
            return 0
        description = soup.find('meta', attrs={'name': 'description'})
        return 1 if description else 0
    
    def _has_popup(self, html_content):
        """Check for popup indicators in HTML"""
        if not html_content:
            return 0
        popup_keywords = ['window.open', 'popup', 'alert(', 'prompt(']
        return 1 if any(keyword in html_content.lower() for keyword in popup_keywords) else 0
    
    def _has_iframe(self, soup):
        """Check if page has iframe"""
        if not soup:
            return 0
        return 1 if soup.find('iframe') else 0
    
    def _is_form_submit_external(self, soup, netloc):
        """Check if form submits to external domain"""
        if not soup:
            return 0
        forms = soup.find_all('form')
        for form in forms:
            action = form.get('action', '')
            if action and action.startswith('http'):
                action_netloc = urlparse(action).netloc
                if action_netloc and action_netloc != netloc:
                    return 1
        return 0
    
    def _has_social_media(self, soup):
        """Check for social media links"""
        if not soup:
            return 0
        social_domains = ['facebook.com', 'twitter.com', 'instagram.com', 
                         'linkedin.com', 'youtube.com']
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href'].lower()
            if any(domain in href for domain in social_domains):
                return 1
        return 0
    
    def _has_submit_button(self, soup):
        """Check if page has submit button"""
        if not soup:
            return 0
        submit = soup.find(['input', 'button'], attrs={'type': 'submit'})
        return 1 if submit else 0
    
    def _has_hidden_fields(self, soup):
        """Check if page has hidden input fields"""
        if not soup:
            return 0
        hidden = soup.find('input', attrs={'type': 'hidden'})
        return 1 if hidden else 0
    
    def _has_password_fields(self, soup):
        """Check if page has password fields"""
        if not soup:
            return 0
        password = soup.find('input', attrs={'type': 'password'})
        return 1 if password else 0
    
    def _has_banking_keywords(self, url, html_content):
        """Check for banking keywords"""
        text = (url + ' ' + (html_content or '')).lower()
        return 1 if any(keyword in text for keyword in self.banking_keywords) else 0
    
    def _has_payment_keywords(self, url, html_content):
        """Check for payment keywords"""
        text = (url + ' ' + (html_content or '')).lower()
        return 1 if any(keyword in text for keyword in self.payment_keywords) else 0
    
    def _has_crypto_keywords(self, url, html_content):
        """Check for crypto keywords"""
        text = (url + ' ' + (html_content or '')).lower()
        return 1 if any(keyword in text for keyword in self.crypto_keywords) else 0
    
    def _has_copyright_keywords(self, html_content):
        """Check for copyright information"""
        if not html_content:
            return 0
        text = html_content.lower()
        return 1 if any(keyword in text for keyword in self.copyright_keywords) else 0
    
    def _count_images(self, soup):
        """Count number of images"""
        if not soup:
            return 0
        return len(soup.find_all('img'))
    
    def _count_css_files(self, soup):
        """Count number of CSS files"""
        if not soup:
            return 0
        return len(soup.find_all('link', rel='stylesheet'))
    
    def _count_js_files(self, soup):
        """Count number of JavaScript files"""
        if not soup:
            return 0
        return len(soup.find_all('script', src=True))
    
    def _count_self_href(self, soup, netloc):
        """Count links to same domain"""
        if not soup:
            return 0
        links = soup.find_all('a', href=True)
        count = 0
        for link in links:
            href = link['href']
            if href.startswith('/') or netloc in href:
                count += 1
        return count
    
    def _count_empty_ref(self, soup):
        """Count empty href attributes"""
        if not soup:
            return 0
        links = soup.find_all('a', href=True)
        return sum(1 for link in links if not link['href'] or link['href'] in ['#', ''])
    
    def _count_external_ref(self, soup, netloc):
        """Count external links"""
        if not soup:
            return 0
        links = soup.find_all('a', href=True)
        count = 0
        for link in links:
            href = link['href']
            if href.startswith('http') and netloc not in href:
                count += 1
        return count
    
    def _count_popup(self, html_content):
        """Count popup occurrences"""
        if not html_content:
            return 0
        return html_content.lower().count('window.open')
    
    def _count_iframe(self, soup):
        """Count number of iframes"""
        if not soup:
            return 0
        return len(soup.find_all('iframe'))
    
    def _unique_feature_count(self, url):
        """Count unique character types"""
        has_upper = any(c.isupper() for c in url)
        has_lower = any(c.islower() for c in url)
        has_digit = any(c.isdigit() for c in url)
        has_special = any(not c.isalnum() for c in url)
        return sum([has_upper, has_lower, has_digit, has_special])
    
    def _wap_legitimate_prob(self, url):
        """Weighted average probability of being legitimate"""
        score = 0.0
        if 'https' in url:
            score += 0.3
        if len(url) < 75:
            score += 0.2
        if url.count('.') <= 3:
            score += 0.2
        if not any(c.isdigit() for c in urlparse(url).netloc):
            score += 0.3
        return round(score, 6)
    
    def _wap_phishing_prob(self, url):
        """Weighted average probability of being phishing"""
        return round(1.0 - self._wap_legitimate_prob(url), 6)
    
    def _shannon_entropy(self, url):
        """Calculate Shannon entropy of URL"""
        if not url:
            return 0.0
        char_freq = Counter(url)
        total = len(url)
        entropy = -sum((count/total) * math.log2(count/total) for count in char_freq.values())
        return round(entropy, 6)
    
    def _fractal_dimension(self, url):
        """Estimate fractal dimension (box-counting)"""
        if len(url) < 2:
            return 0.0
        bigrams = [url[i:i+2] for i in range(len(url)-1)]
        unique_bigrams = len(set(bigrams))
        return round(unique_bigrams / len(bigrams), 6) if bigrams else 0.0
    
    def _kolmogorov_complexity(self, url):
        """Approximate Kolmogorov complexity using compression"""
        if not url:
            return 0.0
        import zlib
        compressed = zlib.compress(url.encode())
        return round(len(compressed) / len(url), 6) if len(url) > 0 else 0.0
    
    def _hex_pattern_count(self, url):
        """Count hexadecimal patterns"""
        hex_pattern = r'[0-9a-fA-F]{6,}'
        matches = re.findall(hex_pattern, url)
        return len(matches)
    
    def _base64_pattern_count(self, url):
        """Count Base64-like patterns"""
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        matches = re.findall(base64_pattern, url)
        return len(matches)
    
    def _likeliness_index(self, url):
        """Overall likeliness index combining multiple factors"""
        score = 0.0
        score += self._wap_phishing_prob(url) * 0.3
        score += (1 - self._shannon_entropy(url) / 5) * 0.2
        score += self._is_domain_ip(urlparse(url).netloc) * 0.2
        score += (len(url) > 100) * 0.15
        score += (url.count('-') > 3) * 0.15
        return round(score, 6)


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    # Create extractor instance
    extractor = URLFeatureExtractor(timeout=5)
    
    # Example 1: Extract features from a single URL
    url = "	https://jinsanafof10.top:443/"
    features = extractor.extract_features(url)
    
    print("="*80)
    print(f"URL: {url}")
    print("="*80)
    print(f"\nFeature Array (Total: {len(features)} features):")
    print(features)
    print("\n" + "="*80)
    
    # Example 2: Extract from multiple URLs
    print("\nExample 2: Multiple URLs\n")
    
    test_urls = [
        "https://www.amazon.com",
        "http://192.168.1.1/admin",
        "https://phishing-site.com/login?user=test"
    ]
    
    for test_url in test_urls:
        features = extractor.extract_features(test_url)
        print(f"URL: {test_url}")
        print(f"Features (first 10): {features[:10]}")
        print(f"Total features: {len(features)}")
        print("-"*80)