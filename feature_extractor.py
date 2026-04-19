

import re
import math
import urllib.parse
from collections import Counter



# TRUSTED DOMAIN WHITELIST
# Agar domain yahan hai → phishing score automatically kam hoga

TRUSTED_DOMAINS = {
    # Search / Tech
    "google.com", "google.co.in", "google.co.uk",
    "youtube.com", "youtube.in",
    "microsoft.com", "microsoft.in",
    "apple.com", "icloud.com",
    "amazon.com", "amazon.in", "amazon.co.uk",
    "github.com", "gitlab.com",
    "stackoverflow.com",
    "wikipedia.org",
    "linkedin.com",
    "twitter.com", "x.com",
    "facebook.com", "fb.com",
    "instagram.com",
    "whatsapp.com",
    "reddit.com",
    "netflix.com",
    "spotify.com",
    "dropbox.com",
    "paypal.com"
    # Indian sites
    "flipkart.com",
    "naukri.com",
    "irctc.co.in",
    "sbi.co.in",
    "hdfcbank.com",
    "icicibank.com",
    "axisbank.com",
    "paytm.com",
    "phonepe.com",
    "incometax.gov.in",
    "uidai.gov.in",
    # Education
    "coursera.org",
    "udemy.com",
    "edx.org",
    "khanacademy.org",
    # Cloud / Dev
    "aws.amazon.com",
    "cloudflare.com",
    "vercel.app",
    "netlify.app",
    # News
    "bbc.com", "bbc.co.uk",
    "cnn.com",
    "ndtv.com",
    "timesofindia.indiatimes.com",
}

# Multi-word phishing patterns — very specific, low false positive rate
PHISHING_KEYWORDS = [
    "verify-account", "confirm-identity", "suspended-account",
    "update-billing", "unlock-account", "login-required",
    "secure-login", "account-verify", "bank-verify",
    "paypal-security", "ebay-security", "amazon-security",
    "password-reset-required", "credential", "webscr",
]

# Single weak keywords — only meaningful when combined with other signals
WEAK_KEYWORDS = [
    "login", "secure", "verify", "update", "bank",
    "account", "confirm", "password", "billing", "alert",
    "signin", "support", "paypal", "ebay",
]

RISKY_TLDS = {
    ".xyz", ".top", ".click", ".gq", ".ml", ".tk",
    ".cf", ".ga", ".pw", ".work", ".loan", ".online",
    ".site", ".website",
}

LEGIT_TLDS = {
    ".com", ".org", ".net", ".edu", ".gov",
    ".co.in", ".co.uk", ".ac.in", ".ac.uk",
    ".in", ".uk", ".us", ".io", ".dev",
}



# HELPERS


def _parse(url: str) -> urllib.parse.ParseResult:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return urllib.parse.urlparse(url)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _get_domain_parts(url: str):
    """Returns (full_hostname, registered_domain, tld)"""
    p        = _parse(url)
    hostname = (p.hostname or "").lower()
    parts    = hostname.split(".")
    if len(parts) >= 2:
        reg_domain = ".".join(parts[-2:])
        tld        = "." + parts[-1]
    else:
        reg_domain = hostname
        tld        = ""
    return hostname, reg_domain, tld


def _is_trusted(url: str) -> bool:
    """True if the registered domain is in the trusted whitelist."""
    hostname, reg_domain, _ = _get_domain_parts(url)
    if reg_domain in TRUSTED_DOMAINS:
        return True
    if hostname in TRUSTED_DOMAINS:
        return True
    for td in TRUSTED_DOMAINS:
        if hostname == td or hostname.endswith("." + td):
            return True
    return False



# FEATURES


def is_trusted_domain(url: str) -> int:
    return int(_is_trusted(url))


def has_ip_address(url: str) -> int:
    p = _parse(url)
    h = p.hostname or ""
    return int(bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", h)))


def uses_https(url: str) -> int:
    return int(url.strip().lower().startswith("https://"))


def has_at_symbol(url: str) -> int:
    return int("@" in url)


def has_double_slash_redirect(url: str) -> int:
    return int(url.find("//", 8) != -1)


def domain_has_hyphen(url: str) -> int:
    _, reg_domain, _ = _get_domain_parts(url)
    return int("-" in reg_domain)


def domain_has_digits(url: str) -> int:
    _, reg_domain, _ = _get_domain_parts(url)
    domain_only = reg_domain.split(".")[0]
    return int(bool(re.search(r"\d", domain_only)))


def subdomain_count(url: str) -> int:
    hostname, _, _ = _get_domain_parts(url)
    parts = hostname.split(".")
    return max(0, len(parts) - 2)


def excessive_subdomains(url: str) -> int:
    return int(subdomain_count(url) >= 3)


def domain_length(url: str) -> int:
    _, reg_domain, _ = _get_domain_parts(url)
    return len(reg_domain)


def domain_is_long(url: str) -> int:
    _, reg_domain, _ = _get_domain_parts(url)
    return int(len(reg_domain) > 30)


def tld_is_risky(url: str) -> int:
    _, _, tld = _get_domain_parts(url)
    return int(tld in RISKY_TLDS)


def tld_is_legit(url: str) -> int:
    hostname, _, _ = _get_domain_parts(url)
    for t in LEGIT_TLDS:
        if hostname.endswith(t):
            return 1
    return 0


def domain_entropy(url: str) -> float:
    _, reg_domain, _ = _get_domain_parts(url)
    return round(_entropy(reg_domain), 4)


def domain_is_random(url: str) -> int:
    return int(domain_entropy(url) > 3.5)


def path_entropy(url: str) -> float:
    p = _parse(url)
    return round(_entropy(p.path or ""), 4)


def brand_in_subdomain(url: str) -> int:
    """Known brand in subdomain but NOT in registered domain — classic trick."""
    p              = _parse(url)
    hostname       = (p.hostname or "").lower()
    _, reg_domain, _ = _get_domain_parts(url)
    brands = [
        "paypal", "amazon", "google", "facebook", "apple", "microsoft",
        "netflix", "instagram", "twitter", "ebay", "bank", "sbi",
        "hdfc", "icici", "paytm", "flipkart",
    ]
    subdomain_part = hostname.replace(reg_domain, "").strip(".")
    return int(any(b in subdomain_part for b in brands))


def keyword_in_domain(url: str) -> int:
    """Suspicious keyword in DOMAIN name (not path)."""
    hostname, _, _ = _get_domain_parts(url)
    return int(any(kw in hostname for kw in WEAK_KEYWORDS))


#  Ratio-based (fixes false positives) 

def url_length(url: str) -> int:
    return len(url.strip())


def url_is_very_long(url: str) -> int:
    return int(len(url.strip()) > 100)


def slash_ratio(url: str) -> float:
    """Slashes per 10 chars — ratio not raw count."""
    if not url:
        return 0.0
    return round((url.count("/") / len(url)) * 10, 4)


def path_slash_count(url: str) -> int:
    """Slashes in PATH only — excludes scheme and domain."""
    p = _parse(url)
    return p.path.count("/")


def digit_ratio(url: str) -> float:
    if not url:
        return 0.0
    return round(sum(c.isdigit() for c in url) / len(url), 4)


def special_char_ratio(url: str) -> float:
    """Ratio of truly special chars — excludes = and & (normal in queries)."""
    if not url:
        return 0.0
    specials = set("@%#~+$!")
    return round(sum(c in specials for c in url) / len(url), 4)


def num_query_params(url: str) -> int:
    p = _parse(url)
    return len(urllib.parse.parse_qs(p.query)) if p.query else 0


def excessive_query_params(url: str) -> int:
    return int(num_query_params(url) > 5)


def has_port(url: str) -> int:
    p = _parse(url)
    if p.port is None:
        return 0
    return int(p.port not in (80, 443))


def has_phishing_pattern(url: str) -> int:
    url_lower = url.lower()
    return int(any(kw in url_lower for kw in PHISHING_KEYWORDS))


def weak_keyword_count(url: str) -> int:
    url_lower = url.lower()
    return sum(kw in url_lower for kw in WEAK_KEYWORDS)


def phishing_risk_score(url: str) -> int:
    """
    Composite risk score — THE most important feature.
    Needs MULTIPLE bad signals. One slash or one keyword = low score.
    """
    score = 0

    # Strong signals (2 pts each)
    if has_ip_address(url):            score += 2
    if not uses_https(url):            score += 2
    if has_at_symbol(url):             score += 2
    if tld_is_risky(url):              score += 2
    if has_phishing_pattern(url):      score += 2
    if brand_in_subdomain(url):        score += 2
    if domain_has_digits(url):         score += 2
    if domain_is_random(url):          score += 2

    # Medium signals (1 pt each)
    if domain_has_hyphen(url):         score += 1
    if keyword_in_domain(url):         score += 1
    if excessive_subdomains(url):      score += 1
    if domain_is_long(url):            score += 1
    if has_double_slash_redirect(url): score += 1
    if has_port(url):                  score += 1
    if excessive_query_params(url):    score += 1

    # Trusted domain = discount risk
    if _is_trusted(url):
        score = max(0, score - 5)

    return score



# FEATURE LIST


FEATURE_NAMES = [
    "is_trusted_domain",
    "has_ip_address",
    "uses_https",
    "domain_has_hyphen",
    "domain_has_digits",
    "domain_length",
    "domain_is_long",
    "domain_entropy",
    "domain_is_random",
    "subdomain_count",
    "excessive_subdomains",
    "tld_is_risky",
    "tld_is_legit",
    "brand_in_subdomain",
    "keyword_in_domain",
    "url_length",
    "url_is_very_long",
    "slash_ratio",
    "path_slash_count",
    "digit_ratio",
    "special_char_ratio",
    "num_query_params",
    "excessive_query_params",
    "has_at_symbol",
    "has_double_slash_redirect",
    "has_port",
    "has_phishing_pattern",
    "weak_keyword_count",
    "path_entropy",
    "phishing_risk_score",
]


def extract_features(url: str) -> dict:
    url = str(url).strip()
    return {
        "is_trusted_domain":         is_trusted_domain(url),
        "has_ip_address":            has_ip_address(url),
        "uses_https":                uses_https(url),
        "domain_has_hyphen":         domain_has_hyphen(url),
        "domain_has_digits":         domain_has_digits(url),
        "domain_length":             domain_length(url),
        "domain_is_long":            domain_is_long(url),
        "domain_entropy":            domain_entropy(url),
        "domain_is_random":          domain_is_random(url),
        "subdomain_count":           subdomain_count(url),
        "excessive_subdomains":      excessive_subdomains(url),
        "tld_is_risky":              tld_is_risky(url),
        "tld_is_legit":              tld_is_legit(url),
        "brand_in_subdomain":        brand_in_subdomain(url),
        "keyword_in_domain":         keyword_in_domain(url),
        "url_length":                url_length(url),
        "url_is_very_long":          url_is_very_long(url),
        "slash_ratio":               slash_ratio(url),
        "path_slash_count":          path_slash_count(url),
        "digit_ratio":               digit_ratio(url),
        "special_char_ratio":        special_char_ratio(url),
        "num_query_params":          num_query_params(url),
        "excessive_query_params":    excessive_query_params(url),
        "has_at_symbol":             has_at_symbol(url),
        "has_double_slash_redirect": has_double_slash_redirect(url),
        "has_port":                  has_port(url),
        "has_phishing_pattern":      has_phishing_pattern(url),
        "weak_keyword_count":        weak_keyword_count(url),
        "path_entropy":              path_entropy(url),
        "phishing_risk_score":       phishing_risk_score(url),
    }


def extract_features_batch(urls) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([extract_features(u) for u in urls])



# SELF TEST

if __name__ == "__main__":
    import pandas as pd

    test_cases = [
        ("https://www.google.com/search?q=python+tutorial",              "LEGIT",  "Google search — was false positive before"),
        ("https://amazon.in/product/phone/dp/B08N5L5Z7K/ref=sr_1_1",    "LEGIT",  "Amazon deep URL with many slashes"),
        ("https://www.flipkart.com/mobiles/pr?sid=tyy,4io&otracker=hp", "LEGIT",  "Flipkart with query params"),
        ("https://github.com/user/repo/blob/main/src/utils/helper.py",  "LEGIT",  "GitHub deep path"),
        ("https://mail.google.com/mail/u/0/#inbox",                      "LEGIT",  "Gmail"),
        ("http://paypal-secure-login.xyz/verify?account=123",            "PHISH",  "Classic phishing"),
        ("http://192.168.1.1/login/bank/verify",                         "PHISH",  "IP address phishing"),
        ("http://google.com.phishing-site.tk/secure/login",              "PHISH",  "Brand in subdomain + risky TLD"),
        ("http://paypal.attacker.com/webscr?cmd=login",                  "PHISH",  "Paypal brand in subdomain"),
        ("https://secure-update-account-verify.ml/login",                "PHISH",  "Risky TLD + suspicious domain"),
    ]

    print("\n" + "="*75)
    print("FEATURE EXTRACTOR SELF TEST")
    print("="*75)

    correct = 0
    for url, expected, desc in test_cases:
        f         = extract_features(url)
        predicted = "PHISH" if f["phishing_risk_score"] >= 2 else "LEGIT"
        match     = "OK  " if predicted == expected else "FAIL"
        if predicted == expected:
            correct += 1
        print(f"\n[{match}] {desc}")
        print(f"  URL        : {url[:65]}")
        print(f"  Expected   : {expected}  |  Got: {predicted}  "
              f"|  Risk Score: {f['phishing_risk_score']}")
        print(f"  trusted={f['is_trusted_domain']}  https={f['uses_https']}  "
              f"tld_risky={f['tld_is_risky']}  brand_subdomain={f['brand_in_subdomain']}  "
              f"random_domain={f['domain_is_random']}")

    print(f"\n{'='*75}")
    print(f"Score: {correct}/{len(test_cases)} correct")
    print(f"Total features: {len(FEATURE_NAMES)}")
    print(f"{'='*75}")