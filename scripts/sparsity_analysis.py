import pandas as pd
import numpy as np

# Load the merged data
df = pd.read_csv("../results/full_joined.csv")

print("📊 DATA SPARSITY ANALYSIS")
print("=" * 50)

# Extract user_id from the data - look for columns that might contain user info
user_col = None
for col in df.columns:
    if 'user' in col.lower():
        user_col = col
        break

# If no user column found, use flight_id as proxy
if user_col is None:
    print("⚠️ No user_id column found. Using flight_id_pour as proxy for users")
    user_col = 'flight_id_pour'

# Basic statistics
print(f"\nTotal Ratings: {len(df)}")
print(f"Unique Users/Flights: {df[user_col].nunique()}")
print(f"Unique Whiskeys: {df['whiskey_id'].nunique()}")

# Count ratings per user and per whiskey
user_counts = df[user_col].value_counts()
whiskey_counts = df['whiskey_id'].value_counts()

print(f"\n📈 RATING DISTRIBUTION:")
print(f"Average ratings per user: {user_counts.mean():.2f}")
print(f"Median ratings per user: {user_counts.median():.0f}")
print(f"Average ratings per whiskey: {whiskey_counts.mean():.2f}")
print(f"Median ratings per whiskey: {whiskey_counts.median():.0f}")

# Sparsity metrics
print(f"\n⚠️ DATA COVERAGE:")
print(f"Users with < 3 tastings: {(user_counts < 3).sum()} ({(user_counts < 3).sum() / len(user_counts) * 100:.1f}%)")
print(f"Users with < 5 tastings: {(user_counts < 5).sum()} ({(user_counts < 5).sum() / len(user_counts) * 100:.1f}%)")
print(f"Whiskeys with < 3 ratings: {(whiskey_counts < 3).sum()} ({(whiskey_counts < 3).sum() / len(whiskey_counts) * 100:.1f}%)")
print(f"Whiskeys with < 5 ratings: {(whiskey_counts < 5).sum()} ({(whiskey_counts < 5).sum() / len(whiskey_counts) * 100:.1f}%)")

# Calculate sparsity
total_possible_ratings = df[user_col].nunique() * df['whiskey_id'].nunique()
actual_ratings = len(df)
sparsity = 1 - (actual_ratings / total_possible_ratings)

print(f"\n📊 SPARSITY METRICS:")
print(f"Total possible user-whiskey pairs: {total_possible_ratings:,}")
print(f"Actual ratings: {actual_ratings:,}")
print(f"Data Sparsity: {sparsity:.2%}")
print(f"Data Density: {(1-sparsity):.2%}")

# Rating score analysis if available
if 'rating' in df.columns:
    print(f"\n⭐ RATING SCORES:")
    print(f"Average rating: {df['rating'].mean():.2f}")
    print(f"Median rating: {df['rating'].median():.2f}")
    print(f"Rating std dev: {df['rating'].std():.2f}")
    print(f"Rating range: {df['rating'].min():.1f} - {df['rating'].max():.1f}")

# Recommendations for modeling
print(f"\n💡 RECOMMENDATIONS:")
if sparsity > 0.99:
    print("- ⚠️ EXTREMELY SPARSE DATA (>99% sparse)")
    print("- Consider using content-based filtering primarily")
    print("- Collaborative filtering will likely struggle")
    print("- Focus on whiskey attributes rather than user patterns")
elif sparsity > 0.95:
    print("- ⚠️ VERY SPARSE DATA (>95% sparse)")
    print("- Hybrid approach recommended (content + collaborative)")
    print("- Use regularization to prevent overfitting")
    print("- Consider matrix factorization techniques")
else:
    print("- Data density is reasonable for collaborative filtering")
    print("- Can use standard recommendation algorithms")

# Save analysis results
results = {
    'total_ratings': len(df),
    'unique_users': df[user_col].nunique(),
    'unique_whiskeys': df['whiskey_id'].nunique(),
    'sparsity': sparsity,
    'avg_ratings_per_user': user_counts.mean(),
    'avg_ratings_per_whiskey': whiskey_counts.mean()
}

# Save to file
with open('../results/sparsity_analysis_results.txt', 'w') as f:
    f.write("SPARSITY ANALYSIS RESULTS\n")
    f.write("=" * 50 + "\n\n")
    for key, value in results.items():
        f.write(f"{key}: {value}\n")

print("\n✅ Results saved to sparsity_analysis_results.txt")