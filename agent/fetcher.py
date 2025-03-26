import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os

def save_to_csv(data, filename):
    """
    Appends the fetched data to the CSV file. If the file doesn't exist, it creates it with the header.
    """
    df = pd.DataFrame(data)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)
    print(f"Appended {len(data)} entries to {filename}")

def scrape_listings():
    """
    Continuously fetches data from listings on the Unstoppable Domains marketplace.
    It navigates pages using an offset parameter, parses each listing’s domain name, URL,
    status, category, watchlists, and price, and immediately appends the page's data to a CSV file.
    """
    options = Options()
    options.headless = True  # Run headless
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    base_url = "https://unstoppabledomains.com/marketplace"
    offset = 0
    all_data = []  # Optional: to keep everything in memory if needed

    while True:
        url = f"{base_url}?offset={offset}"
        print(f"Scraping page with offset {offset}: {url}")
        driver.get(url)
        
        # Wait for listings to load (using a container class found in your HTML structure)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "css-1r0wvn4-root"))
            )
        except Exception as e:
            print(f"Listings did not load: {e}")
            break

        # Get all listing rows (each row has a unique container)
        listings = driver.find_elements(By.CSS_SELECTOR, "div.css-1r0wvn4-root")
        if not listings:
            print("No more listings found. Ending pagination.")
            break

        page_data = []  # Data for the current page

        for listing in listings:
            try:
                # DOMAIN: find the <a> tag inside the column that holds the domain name.
                domain_anchor = listing.find_element(By.CSS_SELECTOR, "div.css-n9c2ys-column-columnName a")
                domain_name = domain_anchor.text.strip()
                domain_url = domain_anchor.get_attribute("href")
                
                # STATUS: the first column with class css-iu6rg9-column; remove mobile label text if present.
                status_container = listing.find_elements(By.CSS_SELECTOR, "div.css-iu6rg9-column")
                if status_container:
                    status_text = status_container[0].text.strip().split('\n')[-1]
                else:
                    status_text = ""
                
                # CATEGORY: the second css-iu6rg9-column (if it exists)
                if len(status_container) > 1:
                    category_text = status_container[1].text.strip().split('\n')[-1]
                else:
                    category_text = ""
                
                # WATCHLISTS and PRICE: these are in number columns.
                number_columns = listing.find_elements(By.CSS_SELECTOR, "div.css-1skkg1t-column-columnNumber")
                if len(number_columns) >= 2:
                    watchlists = number_columns[0].text.strip().split('\n')[-1]
                    price = number_columns[1].text.strip().split('\n')[-1]
                else:
                    watchlists = ""
                    price = ""
                
                entry = {
                    "domain_name": domain_name,
                    "domain_url": domain_url,
                    "status": status_text,
                    "category": category_text,
                    "watchlists": watchlists,
                    "price": price
                }
                page_data.append(entry)
                all_data.append(entry)  # Optionally keep in memory as well
                print(f"Found listing: {entry}")
            except Exception as e:
                print(f"Error parsing listing: {e}")
                continue

        # Immediately store the current page's data to CSV
        if page_data:
            save_to_csv(page_data, "listings_data.csv")
        else:
            print("No data extracted from this page.")

        offset += 20  # Update the offset; adjust if your page shows a different number of listings.
        time.sleep(2)  # Polite delay

    driver.quit()
    return all_data

if __name__ == "__main__":
    scrape_listings()
    # Data from each page is appended to 'listings_data.csv'.
    # To start fresh, delete the existing CSV file before running.
