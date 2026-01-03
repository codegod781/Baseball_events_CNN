import pandas as pd

def count_events(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, sep=',')

    throws = 0
    swings = 0
    wait = 0
    
    with open('event_log.txt', 'w') as f:
        for i, row in df.iterrows():
            # If we're in waiting period, decrement wait and skip this row
            if wait > 0:
                wait -= 1
                continue
                
            # Check for throws
            if row['prob_2'] > 0.75:
                throws += 1
                f.write(f"{row['timestamp']}: Throw\n")
                wait = 10
                continue
            
            # Check for swings
            if row['prob_5'] > 0.75:
                swings += 1
                f.write(f"{row['timestamp']}: Swing\n")
                wait = 10
                continue
            
            # Check for additional throws
            if row['prob_1'] > 0.15:
                for j in range(i+1, min(i+7, len(df))):  # Added min to prevent index out of bounds
                    if df.iloc[j]['prob_2'] > 0.15:
                        throws += 1
                        f.write(f"{df.iloc[j]['timestamp']}: Throw\n")
                        wait = 10
                        break
            
            # Check for additional swings
            if row['prob_4'] > 0.15:
                for j in range(i+1, min(i+7, len(df))):  # Added min to prevent index out of bounds
                    if df.iloc[j]['prob_5'] > 0.15:
                        swings += 1
                        f.write(f"{df.iloc[j]['timestamp']}: Swing\n")
                        wait = 10
                        break
    
    return throws, swings

# Example usage
throws, swings = count_events('computed_probs.csv')

