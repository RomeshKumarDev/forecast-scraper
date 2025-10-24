# -----------------------
#  Helpers: scraping
# -----------------------

# app.py
import streamlit as st
import time, random
from typing import List


import requests

# Selenium imports (used only if user toggles it on or scraping fails)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
try:
    from selenium_stealth import stealth
    SELENIUM_STEALTH_AVAILABLE = True
except Exception:
    SELENIUM_STEALTH_AVAILABLE = False


def scrape_with_requests(url: str, timeout=15):
    
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text

def scrape_with_selenium(url: str, headless=True, wait_range=(2,5)):
    # NOTE: Requires chromedriver available in PATH or Service path set properly
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(service=Service(), options=chrome_options)
    if SELENIUM_STEALTH_AVAILABLE:
        stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True)
    try:
        driver.get(url)
        time.sleep(random.uniform(*wait_range))
        html = driver.page_source
        return html
    finally:
        try:
            driver.quit()
        except Exception:
            pass