import os
import json
import urllib.request
import urllib.parse
from datetime import datetime

# Configure the API request to GDELT Doc 2.0 API
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Using GDELT themes to find major economic, political, and armed conflict events
QUERY = "(theme:ECON_ECONOMY OR theme:ARMEDCONFLICT OR theme:POLITICAL_TURMOIL) sourcelang:eng"

def fetch_gdelt_data():
    params = {
        "query": QUERY,
        "mode": "artlist",
        "maxrecords": "50",
        "format": "json"
    }
    
    # URL encode parameters but avoid encoding the parentheses/colons too strictly if it breaks the API
    query_string = urllib.parse.urlencode(params)
    url = f"{GDELT_API_URL}?{query_string}"
    
    print(f"Fetching data from GDELT...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (AGEMIS ETL Pipeline)'})
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = response.read().decode('utf-8')
                return json.loads(data)
            else:
                print(f"Failed to fetch data: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error fetching from GDELT: {e}")
        return None

def format_timestamp(seendate):
    try:
        # GDELT format: 20260420T141500Z -> Target format: 2026-04-20T14:15:00Z
        dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return seendate

def transform_to_signals(gdelt_data):
    signals = []
    if not gdelt_data or "articles" not in gdelt_data:
        return signals
        
    for article in gdelt_data["articles"]:
        seendate = article.get("seendate", "")
        timestamp = format_timestamp(seendate) if seendate else datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        title = article.get("title", "No Title")
        
        # Determine a basic domain tag based on text
        lower_title = title.lower()
        if "bank" in lower_title or "economy" in lower_title or "market" in lower_title:
            domain = "Econ"
        elif "war" in lower_title or "conflict" in lower_title or "military" in lower_title:
            domain = "Security"
        elif "tech" in lower_title or "cyber" in lower_title:
            domain = "Tech"
        else:
            domain = "Pol"

        signal = {
            "domain": domain,
            "source_entity": article.get("domain", "Unknown Source"),
            "source_trust_level": "medium",
            "timestamp": timestamp,
            "title": title,
            # Using the title and url as the content snippet
            "content": f"{title}. Source: {article.get('url', '')}"
        }
        signals.append(signal)
        
    return signals

def save_signals(signals, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(signals, f, indent=2)
    print(f"Saved {len(signals)} signals to {output_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    output_path = os.path.join(workspace_dir, "signals.json")
    
    data = fetch_gdelt_data()
    if data:
        signals = transform_to_signals(data)
        if signals:
            save_signals(signals, output_path)
        else:
            print("No valid articles found in GDELT data.")
    else:
        print("ETL failed. Note: GDELT API has strict rate limits. Please wait 5+ seconds before retrying.")
