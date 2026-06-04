import os
import pandas as pd
from pathlib import Path
from collections import Counter

def get_majority_vote(votes):
    """
    Returns the majority vote from a list of votes. 
    If there is a tie (e.g., all 3 models give unique labels, or no valid votes), 
    we flag it as 'Needs Human Review'.
    """
    # Remove NaN values and empty strings to ensure we are voting on actual labels
    valid_votes = [v for v in votes if pd.notna(v) and str(v).strip() != ""]
    if not valid_votes:
        return "Needs Human Review"
    
    counts = Counter(valid_votes)
    most_common = counts.most_common(2)
    
    # If there is a tie for the most common label
    # (e.g., 3 different labels, so the top 2 both have count 1)
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return "Needs Human Review"
        
    return most_common[0][0]

def main():
    # Base directory containing the annotations
    base_dir = Path(r"c:\Users\khati\OneDrive\Documents\PluralBench-NP\data\annotations\label")
    
    # The models to include in majority voting
    models = ["GPT-5.2", "Gemini 3.5 flash", "Sonet 4.6"]
    
    # Directory to save the final majority voting annotations
    output_dir = base_dir / "Majority_Voting"
    
    # Use GPT-5.2 as a reference for folder structure and files
    reference_model_dir = base_dir / models[0]
    
    if not reference_model_dir.exists():
        print(f"Reference directory {reference_model_dir} does not exist.")
        return
        
    for file_path in reference_model_dir.rglob("*.csv"):
        rel_path = file_path.relative_to(reference_model_dir)
        
        # Read the CSV from each model
        dfs = []
        for model in models:
            model_file = base_dir / model / rel_path
            if model_file.exists():
                try:
                    dfs.append(pd.read_csv(model_file))
                except Exception as e:
                    print(f"Error reading {model_file}: {e}")
            else:
                print(f"Warning: {model_file} not found.")
                
        if not dfs:
            continue
            
        # Determine the minimum row count in case there's any mismatch between files
        min_len = min(len(df) for df in dfs)
        if any(len(df) != min_len for df in dfs):
            print(f"Warning: Row counts mismatch for {rel_path}. Using minimum row count {min_len}.")
            
        # Base the final dataframe entirely on the first model's data
        # This guarantees the exact same structure (columns and rows) as the original file
        final_df = dfs[0].copy().head(min_len)
        
        # Identify annotation columns (everything except the 'comment' column)
        annotation_cols = [col for col in final_df.columns if col != 'comment']
        
        # Perform majority voting exactly row by row, column by column
        for idx in range(min_len):
            for col in annotation_cols:
                # Gather the votes from all models for this exact row index and column
                votes = [df.at[idx, col] for df in dfs if col in df.columns]
                
                # Apply the majority vote rule, assigning ties to 'Needs Human Review'
                final_df.at[idx, col] = get_majority_vote(votes)
                    
        # Save to the output directory, maintaining the subfolder structure
        out_file = output_dir / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        final_df.to_csv(out_file, index=False)
        print(f"Processed and saved: {out_file}")

if __name__ == "__main__":
    main()
