import pandas as pd
import json
import os

# Load Excel
df = pd.read_excel('Positive and Negative Word List.xlsx')

# Extract and clean (drop NaN, lowercase, strip)
neg_words = df['Negative Sense Word List'].dropna().astype(str).str.lower().str.strip().tolist()
pos_words = df['Positive Sense Word List'].dropna().astype(str).str.lower().str.strip().tolist()

# Save to JSON
with open('models/negative_words.json', 'w') as f:
    json.dump(neg_words, f)
    
with open('models/positive_words.json', 'w') as f:
    json.dump(pos_words, f)

print(f"Saved {len(neg_words)} negative words and {len(pos_words)} positive words.")
