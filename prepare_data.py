import pandas as pd
import requests
import io

print("Downloading real Amazon reviews (approx 127MB)...")
url = "https://huggingface.co/datasets/datahiveai/Amazon-Reviews-Dataset/resolve/main/reviews.csv"

# Stream the download to show progress
response = requests.get(url, stream=True)
response.raise_for_status()
total_length = int(response.headers.get('content-length', 0))

downloaded = 0
chunks = []

for chunk in response.iter_content(chunk_size=8192):
    chunks.append(chunk)
    downloaded += len(chunk)
    if total_length > 0:
        done = int(50 * downloaded / total_length)
        print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded / (1024*1024):.1f} MB / {total_length / (1024*1024):.1f} MB", end='')

print("\nDownload complete! Processing data...")

# Load into pandas
content = b"".join(chunks)
df = pd.read_csv(io.BytesIO(content))
print(f"Downloaded {len(df)} rows.")

# Sample 300 rows to keep API costs low
df_sample = df.sample(n=300, random_state=42).copy()

# Map the actual columns to what backend.py expects
column_mapping = {
    'body': 'review_text',
    'stars': 'rating',
    'asin': 'product',
    'id': 'review_id'
}
df_sample = df_sample.rename(columns=column_mapping)

# Ensure we have the required columns
required = ['review_id', 'product', 'rating', 'review_text']
df_final = df_sample[required]

# Clean up the text (remove any nulls)
df_final = df_final.dropna(subset=['review_text'])
df_final['review_text'] = df_final['review_text'].astype(str)

# Save the reshaped data
df_final.to_csv("data/reviews_real.csv", index=False)
print(f"\nCreated data/reviews_real.csv with {len(df_final)} rows.")
print("\nHere is a preview of your real data:")
print(df_final.head(3))