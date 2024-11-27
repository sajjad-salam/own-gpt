import requests
from bs4 import BeautifulSoup
import re
from googlesearch import search 

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text) 
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  
    return cleaned_text.strip()

def fetch_and_save_text(url, filename="output.txt"):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator="\n")
        
        cleaned_content = clean_text(text_content)
        
        with open(filename, "a", encoding="utf-8") as file:
            file.write(cleaned_content + "\n\n") 

        print(f"\nContent from {url} has been saved to {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")

def search_and_fetch(query):
    try:
        results = []
        for url in search(query, lang="ar"):
            results.append(url)
            if len(results) >= 5: 
                break
        
        for i, result_url in enumerate(results):
            print(f"\nFetching data from URL {i + 1}: {result_url}")
            fetch_and_save_text(result_url, filename="data.txt")

        print("\nAll content has been saved to data.txt")

    except Exception as e:
        print(f"Error during search or fetch: {e}")

def main():
    print("Choose an option:")
    print("1 - Fetch data via search")
    print("2 - Fetch data via URL input")

    choice = input("Enter your choice (1 or 2): ")
    if choice == "1":
        query = input("Enter your search query (e.g., health articles): ")
        search_and_fetch(query)
    elif choice == "2":
        url = input("Enter the URL: ")
        filename = url.split("//")[-1].split("/")[0] + ".txt" 
        fetch_and_save_text(url, filename)
    else:
        print("Invalid choice, please enter 1 or 2.")

if __name__ == "__main__":
    main()
