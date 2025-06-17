import pandas as pd
import os

# CSV data path
data_path = "../data/WhiskeyHubMySQL_6_13_2025_pt2/"

# Read CSV files
try:
    flights = pd.read_csv(os.path.join(data_path, "flights.csv"), encoding='utf-8-sig')
    print(f"✅ Loaded flights: {len(flights)} rows")
except Exception as e:
    print(f"❌ Error loading flights: {e}")
    flights = pd.DataFrame()

try:
    pours = pd.read_csv(os.path.join(data_path, "flight_pours.csv"), encoding='utf-8-sig')
    print(f"✅ Loaded flight_pours: {len(pours)} rows")
except Exception as e:
    print(f"❌ Error loading flight_pours: {e}")
    pours = pd.DataFrame()

try:
    notes = pd.read_csv(os.path.join(data_path, "flight_notes.csv"), encoding='utf-8-sig')
    print(f"✅ Loaded flight_notes: {len(notes)} rows")
except Exception as e:
    print(f"❌ Error loading flight_notes: {e}")
    notes = pd.DataFrame()

try:
    whiskeys = pd.read_csv(os.path.join(data_path, "whishkeys.csv"), encoding='utf-8-sig')
    print(f"✅ Loaded whiskeys: {len(whiskeys)} rows")
except Exception as e:
    print(f"❌ Error loading whiskeys: {e}")
    whiskeys = pd.DataFrame()

# Check if we have data to merge
if len(pours) > 0 and len(notes) > 0:
    # Merge tables
    print("\n🔄 Merging tables...")
    
    # First merge pours with notes
    df = pours.merge(notes, left_on="id", right_on="flight_pour_id", how="inner", suffixes=('_pour', '_note'))
    print(f"  - Pours + Notes: {len(df)} rows")
    
    # Print columns to debug
    print(f"  - Columns after first merge: {list(df.columns)[:10]}...")
    
    # Then merge with flights - use the flight_id from the correct table
    if len(flights) > 0:
        # Check which column to use
        flight_col = 'flight_id_pour' if 'flight_id_pour' in df.columns else 'flight_id'
        df = df.merge(flights, left_on=flight_col, right_on="id", how="left", suffixes=('', '_flight'))
        print(f"  - + Flights: {len(df)} rows")
    
    # Finally merge with whiskeys
    if len(whiskeys) > 0:
        df = df.merge(whiskeys, left_on="whiskey_id", right_on="id", how="left", suffixes=('', '_whiskey'))
        print(f"  - + Whiskeys: {len(df)} rows")
    
    # Save merged data
    df.to_csv("../results/full_joined.csv", index=False)
    print(f"\n✅ Merged data saved to full_joined.csv ({len(df)} rows, {len(df.columns)} columns)")
    
    # Show sample of columns
    print(f"\nColumns in merged data: {list(df.columns)[:10]}... (showing first 10)")
else:
    print("\n❌ Not enough data to perform merge. Check if CSV files are loaded correctly.")