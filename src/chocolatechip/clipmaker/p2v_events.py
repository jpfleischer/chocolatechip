import pandas as pd
import sys
import os

from chocolatechip.MySQLConnector import MySQLConnector

# instantiate your connector
connector = MySQLConnector()

# open a connection (using your default “testdb” / or swap use_testdb=True)
myoutputdb = connector._connect()
mycursor   = myoutputdb.cursor()

# Get intersection ID from arguments
iid = sys.argv[1]

# Create necessary directories
p2v_dir = f"{iid}/p2v"
v2v_dir = f"{iid}/v2v"
os.makedirs(p2v_dir, exist_ok=True)
os.makedirs(v2v_dir, exist_ok=True)

# Initialize DataFrames
p2v_df = pd.DataFrame()
v2v_df = pd.DataFrame()

# Process each timestamp pair
for i in range(2, len(sys.argv), 2):
    start_time = pd.to_datetime(sys.argv[i])
    end_time = pd.to_datetime(sys.argv[i + 1])

    # Query for `p2v = 1`
    p2v_query = f"""
        SELECT * FROM TTCTable 
        WHERE intersection_id = '{iid}' 
        AND include_flag = 1 
        AND timestamp BETWEEN '{start_time}' AND '{end_time}' 
        AND p2v = 1
    """
    p2v_df = pd.concat([p2v_df, pd.read_sql(p2v_query, myoutputdb)])

    # Query for `p2v = 0`
    v2v_query = f"""
        SELECT * FROM TTCTable 
        WHERE intersection_id = '{iid}' 
        AND include_flag = 1 
        AND timestamp BETWEEN '{start_time}' AND '{end_time}' 
        AND p2v = 0
    """
    v2v_df = pd.concat([v2v_df, pd.read_sql(v2v_query, myoutputdb)])

# Remove duplicates
p2v_df = p2v_df.drop_duplicates(subset=['unique_ID1', 'unique_ID2']).reset_index(drop=True)
v2v_df = v2v_df.drop_duplicates(subset=['unique_ID1', 'unique_ID2']).reset_index(drop=True)

# Sort by time column
p2v_df = p2v_df.sort_values(by='timestamp')
v2v_df = v2v_df.sort_values(by='timestamp')

# Save to CSV
p2v_df.to_csv(f"{p2v_dir}/p2v_events.csv", index=True, index_label='index')
v2v_df.to_csv(f"{v2v_dir}/v2v_events.csv", index=True, index_label='index')

print(f"✅ Saved {len(p2v_df)} p2v events to {p2v_dir}/p2v_events.csv")
print(f"✅ Saved {len(v2v_df)} v2v events to {v2v_dir}/v2v_events.csv")
