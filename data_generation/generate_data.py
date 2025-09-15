import google.generativeai as genai
import os
import pandas as pd
import random
import json
import time

# --- IMPORTANT: Configure your API Key ---
# Option 1: Directly in script (less secure, but fast for quick demo)
# Replace 'YOUR_GEMINI_API_KEY' with the key you copied from Google Cloud Console
# os.environ['GOOGLE_API_KEY'] = 'YOUR_GEMINI_API_KEY'
# genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Option 2: Set as an environment variable (recommended for security)
# Before running the script, in your terminal:
# export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
# Then just run:
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBCxKmj162AV2uF1KUXiW9TTITR_vkhSg4' # <--- REPLACE THIS LINE with your actual key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the Gemini model (1.5 Flash is good for speed and free tier)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Your Data Parameters ---
industries = [
    "Gym", "Hair Salon", "Roofing Company", "Online Clothing Boutique",
    "Italian Restaurant", "Digital Marketing Agency", "Local Coffee Shop",
    "Plumbing Services", "Yoga Studio", "Pet Grooming"
]

regions = [
    "Munich, Germany", "Berlin, Germany", "Paris, France", "London, UK",
    "New York, USA", "Los Angeles, USA", "Toronto, Canada", "Sydney, Australia",
    "Tokyo, Japan", "Rome, Italy"
]

target_audiences = [
    "Fitness Enthusiasts", "Fashion-conscious Young Adults", "Homeowners needing repairs",
    "Foodies seeking authentic cuisine", "Small Business Owners", "Coffee Lovers",
    "Pet Owners"
]

# --- Generation Settings ---
CAMPAIGNS_PER_INDUSTRY = 100
NUM_CAMPAIGNS_TO_GENERATE = len(industries) * CAMPAIGNS_PER_INDUSTRY
OUTPUT_FILE_NAME = "../data/raw/synthetic_ad_data_tiered_performance.csv"
RATE_LIMIT_DELAY = 3 # seconds to wait between API calls to avoid hitting limits

# --- Performance Tier Definitions ---
# These ranges will guide the LLM to generate specific performance levels.
# The LLM will be asked to generate numbers *within these ranges*.
# We define ranges for Impressions, Clicks, Conversions, and Cost
# and ensure they roughly align with high/mid/low performance
performance_tiers = {
    "high": {
        "impressions": (20000, 50000), "clicks": (1000, 3000), "conversions": (30, 80), "cost": (200, 800) # High CTR, High Conv Rate, Low-Mid CPA/CPC
    },
    "mid": {
        "impressions": (5000, 15000), "clicks": (200, 800), "conversions": (5, 25), "cost": (50, 300) # Mid CTR, Mid Conv Rate, Mid CPA/CPC
    },
    "low": {
        "impressions": (1000, 5000), "clicks": (20, 150), "conversions": (0, 4), "cost": (20, 150) # Low CTR, Low Conv Rate, High CPA/CPC (or very few conversions)
    }
}

def generate_data():
    # Create data directory if it doesn't exist
    os.makedirs('../data/raw', exist_ok=True)
    
    generated_data = []

    print(f"Starting synthetic data generation for {NUM_CAMPAIGNS_TO_GENERATE} campaigns ({CAMPAIGNS_PER_INDUSTRY} per industry)...")

    campaign_counter = 0
    for selected_industry in industries:
        for j in range(CAMPAIGNS_PER_INDUSTRY):
            campaign_counter += 1

            selected_region = random.choice(regions)
            selected_audience = random.choice(target_audiences)

            # Randomly assign a performance tier
            performance_tier_name = random.choice(list(performance_tiers.keys()))
            tier_ranges = performance_tiers[performance_tier_name]

            # Construct the prompt with dynamic performance ranges
            prompt = f"""
            Generate a highly effective Google Ads campaign in JSON format for the following scenario:
            - **Industry:** {selected_industry}
            - **Geographical Region:** {selected_region}
            - **Target Audience:** {selected_audience}
            - **Desired Performance Level:** {performance_tier_name} (Generate metrics within these approximate ranges)

            The JSON output should contain:
            {{
                "industry": "{selected_industry}",
                "region": "{selected_region}",
                "target_audience": "{selected_audience}",
                "performance_tier": "{performance_tier_name}",
                "ad_headlines": [
                    "Headline 1 (max 30 chars)",
                    "Headline 2 (max 30 chars)"
                ],
                "ad_descriptions": [
                    "Description 1 (max 90 chars)",
                    "Description 2 (max 90 chars)",
                    "Description 3 (max 90 chars)"
                ],
                "keywords": [
                    "keyword phrase 1",
                    "keyword phrase 2",
                    "keyword phrase 3",
                    "keyword phrase 4",
                    "keyword phrase 5"
                ],
                "simulated_performance": {{
                    "impressions": "(number between {tier_ranges['impressions'][0]} and {tier_ranges['impressions'][1]})",
                    "clicks": "(number between {tier_ranges['clicks'][0]} and {tier_ranges['clicks'][1]})",
                    "conversions": "(number between {tier_ranges['conversions'][0]} and {tier_ranges['conversions'][1]})",
                    "cost": "(number between {tier_ranges['cost'][0]} and {tier_ranges['cost'][1]})"
                }}
            }}
            Ensure all string values are enclosed in double quotes. Do not include any explanation or extra text outside the JSON.
            Make sure the simulated performance numbers are actual integers or floats, not text like '(number between X and Y)'.
            """

            try:
                response = model.generate_content(
                    prompt,
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )

                # Parse the JSON response
                try:
                    json_string = response.text.strip()
                    if json_string.startswith("```json"):
                        json_string = json_string[len("```json"):].strip()
                    if json_string.endswith("```"):
                        json_string = json_string[:-len("```")].strip()

                    campaign_data = json.loads(json_string)

                    # --- Ensure the performance tier is captured ---
                    campaign_data["performance_tier"] = performance_tier_name

                    # For simulated metrics, ensure they are numbers for later use
                    if "simulated_performance" in campaign_data:
                        perf = campaign_data["simulated_performance"]

                        # Convert string numbers to actual numbers, handle errors
                        for key in ["impressions", "clicks", "conversions", "cost"]:
                            if isinstance(perf.get(key), str) and perf[key].replace('.', '', 1).isdigit():
                                perf[key] = float(perf[key]) # Use float to preserve decimals for cost
                                if key != "cost": # For impressions/clicks/conversions, ensure they are ints
                                    perf[key] = int(perf[key])
                            elif not isinstance(perf.get(key), (int, float)):
                                # Fallback to random within the TIER's range if parsing fails or key is missing
                                range_min, range_max = tier_ranges[key]
                                if key == "cost":
                                    perf[key] = round(random.uniform(range_min, range_max), 2)
                                else:
                                    perf[key] = random.randint(range_min, range_max)

                        # Calculate CTR and Conversion Rate (handle division by zero)
                        impressions = perf.get("impressions", 0)
                        clicks = perf.get("clicks", 0)
                        conversions = perf.get("conversions", 0)
                        cost = perf.get("cost", 0)

                        campaign_data["simulated_performance"]["ctr"] = (clicks / impressions) * 100 if impressions > 0 else 0
                        campaign_data["simulated_performance"]["conversion_rate"] = (conversions / clicks) * 100 if clicks > 0 else 0
                        campaign_data["simulated_performance"]["cpc"] = (cost / clicks) if clicks > 0 else 0
                        campaign_data["simulated_performance"]["cpa"] = (cost / conversions) if conversions > 0 else 0

                    generated_data.append(campaign_data)
                    print(f"Successfully generated campaign {campaign_counter}/{NUM_CAMPAIGNS_TO_GENERATE} (Industry: {selected_industry}, Tier: {performance_tier_name})")

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for campaign {campaign_counter}: {e}")
                    print(f"Raw response text: {response.text}")
                except Exception as e:
                    print(f"An unexpected error occurred during parsing for campaign {campaign_counter}: {e}")
                    print(f"Raw response text: {response.text}")

            except Exception as e:
                print(f"API call failed for campaign {campaign_counter}: {e}")
                # Implement retry logic if needed, or simply skip and log

            # Pause to respect rate limits and free tier
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\nGeneration complete. Total campaigns generated: {len(generated_data)}")

    # --- Save the Data ---
    if generated_data:
        # Flatten the dictionary for CSV export
        flattened_data = []
        for item in generated_data:
            s_perf = item["simulated_performance"]
            flat_item = {
                "industry": item["industry"],
                "region": item["region"],
                "target_audience": item["target_audience"],
                "performance_tier": item["performance_tier"], # Added performance tier
                "headline_1": item["ad_headlines"][0] if len(item["ad_headlines"]) > 0 else "",
                "headline_2": item["ad_headlines"][1] if len(item["ad_headlines"]) > 1 else "",
                "description_1": item["ad_descriptions"][0] if len(item["ad_descriptions"]) > 0 else "",
                "description_2": item["ad_descriptions"][1] if len(item["ad_descriptions"]) > 1 else "",
                "description_3": item["ad_descriptions"][2] if len(item["ad_descriptions"]) > 2 else "",
                "keywords": ", ".join(item["keywords"]), # Join keywords into a single string
                "simulated_impressions": s_perf["impressions"],
                "simulated_clicks": s_perf["clicks"],
                "simulated_conversions": s_perf["conversions"],
                "simulated_cost": round(s_perf["cost"], 2),
                "simulated_ctr": round(s_perf["ctr"], 2),
                "simulated_conversion_rate": round(s_perf["conversion_rate"], 2),
                "simulated_cpc": round(s_perf["cpc"], 2),
                "simulated_cpa": round(s_perf["cpa"], 2)
            }
            flattened_data.append(flat_item)

        df = pd.DataFrame(flattened_data)
        df.to_csv(OUTPUT_FILE_NAME, index=False)
        print(f"Data saved to {OUTPUT_FILE_NAME}")
    else:
        print("No data was generated.")

if __name__ == "__main__":
    generate_data()
