import pandas as pd
import numpy as np
import os
import sys

def install_required_packages():
    """Install required packages if missing"""
    try:
        import openpyxl
    except ImportError:
        print("Installing required package: openpyxl")
        os.system(f"{sys.executable} -m pip install openpyxl")
        print("Package installed successfully!")

def handle_nulls_and_duplicates(df, df_name):
    """Handle null values and duplicates in dataframe"""
    print(f"\n--- Handling Nulls and Duplicates for {df_name} ---")
    
    # Check for null values
    print("Null values before handling:")
    print(df.isnull().sum())
    
    # Handle null values based on column type
    for column in df.columns:
        if df[column].isnull().any():
            null_count = df[column].isnull().sum()
            if df[column].dtype in ['float64', 'int64']:
                # For numerical columns, fill with median
                median_val = df[column].median()
                df[column].fillna(median_val, inplace=True)
                print(f"Filled {null_count} nulls in {column} with median: {median_val}")
            elif df[column].dtype == 'object':
                # For categorical columns, fill with mode
                mode_val = df[column].mode()[0] if not df[column].mode().empty else 'Unknown'
                df[column].fillna(mode_val, inplace=True)
                print(f"Filled {null_count} nulls in {column} with mode: {mode_val}")
    
    # Check for duplicates
    initial_rows = len(df)
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed {duplicates} duplicate rows")
    
    print(f"Rows after cleaning: {len(df)} (removed {initial_rows - len(df)})")
    return df

def fix_data_types(df, df_name):
    """Fix data types for optimal processing"""
    print(f"\n--- Fixing Data Types for {df_name} ---")
    
    print("Data types before fixing:")
    print(df.dtypes)
    
    # Convert date columns
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted {col} to datetime")
    
    # Convert ID columns to appropriate types
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    for col in id_columns:
        if col in df.columns and df[col].dtype in ['float64', 'object']:
            # Try to convert to integer, if not possible keep as string
            try:
                df[col] = df[col].astype('Int64')  # Uses nullable integer type
                print(f"Converted {col} to integer")
            except (ValueError, TypeError):
                df[col] = df[col].astype(str)
                print(f"Converted {col} to string")
    
    # Convert categorical columns
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 50]
    for col in categorical_columns:
        df[col] = df[col].astype('category')
        print(f"Converted {col} to category ({df[col].nunique()} unique values)")
    
    print("Data types after fixing:")
    print(df.dtypes)
    
    return df

def justify_join_logic(transactions_df, social_profiles_df):
    """Analyze and justify the join logic"""
    print("\n--- Join Logic Justification ---")
    
    # Analyze customer ID mapping
    transactions_customers = set(transactions_df['customer_id_legacy'].astype(int).astype(str))
    social_customers = set(social_profiles_df['customer_id_new'].str.replace('A', '').astype(int).astype(str))
    
    common_customers = transactions_customers.intersection(social_customers)
    only_transactions = transactions_customers - social_customers
    only_social = social_customers - transactions_customers
    
    print(f"Total unique customers in transactions: {len(transactions_customers)}")
    print(f"Total unique customers in social profiles: {len(social_customers)}")
    print(f"Common customers (can be merged): {len(common_customers)}")
    print(f"Customers only in transactions: {len(only_transactions)}")
    print(f"Customers only in social profiles: {len(only_social)}")
    
    # Calculate overlap percentages
    trans_overlap = len(common_customers) / len(transactions_customers) * 100
    social_overlap = len(common_customers) / len(social_customers) * 100
    
    print(f"\nOverlap analysis:")
    print(f"{trans_overlap:.1f}% of transaction customers have social profiles")
    print(f"{social_overlap:.1f}% of social profile customers have transactions")
    
    # Justify inner join choice
    print(f"\nJoin Type Justification:")
    print("INNER JOIN selected because:")
    print("  - We want only customers who exist in both datasets")
    print("  - This ensures complete customer profiles (both transaction and social data)")
    print("  - Avoids null values for key customer attributes")
    print("  - Maintains data integrity for analysis")
    
    if len(common_customers) == 0:
        print("\nWARNING: No common customers found! Check ID mapping logic.")
        return False
    
    return True

def perform_post_merge_checks(merged_df, original_transactions, original_social):
    """Perform comprehensive checks after merge"""
    print("\n--- Post-Merge Quality Checks ---")
    
    # Basic shape and size checks
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Expected columns: {len(original_transactions.columns) + len(original_social.columns) - 1}")  # -1 for common ID
    
    # Check for null values in merged data
    print("\nNull values in merged dataset:")
    null_summary = merged_df.isnull().sum()
    for col, null_count in null_summary.items():
        if null_count > 0:
            print(f"  {col}: {null_count} nulls ({null_count/len(merged_df)*100:.1f}%)")
    
    # Check for duplicates in merged data
    merged_duplicates = merged_df.duplicated().sum()
    print(f"\nDuplicate rows in merged data: {merged_duplicates}")
    
    # Verify key relationships are maintained
    unique_customers_merged = merged_df['customer_id_common'].nunique()
    print(f"Unique customers in merged data: {unique_customers_merged}")
    
    # Check data integrity for key columns
    print("\nData Integrity Checks:")
    print(f"All customer_id_common values are strings: {merged_df['customer_id_common'].dtype == 'object'}")
    print(f"No negative purchase amounts: {(merged_df['purchase_amount'] >= 0).all()}")
    
    # Validate date ranges if available
    date_cols = [col for col in merged_df.columns if 'date' in col.lower()]
    for col in date_cols:
        if col in merged_df.columns:
            print(f"{col} range: {merged_df[col].min()} to {merged_df[col].max()}")
    
    # Check rating ranges
    if 'customer_rating' in merged_df.columns:
        print(f"Customer rating range: {merged_df['customer_rating'].min()} to {merged_df['customer_rating'].max()}")
    
    if 'engagement_score' in merged_df.columns:
        print(f"Engagement score range: {merged_df['engagement_score'].min()} to {merged_df['engagement_score'].max()}")

def merge_datasets():
    """Merge customer_transactions and customer_social_profiles datasets with comprehensive data cleaning"""
    
    # Install required packages first
    install_required_packages()
    
    # Create dataset folder if it doesn't exist
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    try:
        # Load the datasets
        print("Loading datasets...")
        transactions_df = pd.read_excel('dataset/customer_transactions.xlsx')
        social_profiles_df = pd.read_excel('dataset/customer_social_profiles.xlsx')
        
        print(f"Transactions dataset shape: {transactions_df.shape}")
        print(f"Social profiles dataset shape: {social_profiles_df.shape}")
        
        # Step 1: Handle nulls and duplicates
        transactions_clean = handle_nulls_and_duplicates(transactions_df.copy(), "Transactions")
        social_clean = handle_nulls_and_duplicates(social_profiles_df.copy(), "Social Profiles")
        
        # Step 2: Fix data types
        transactions_clean = fix_data_types(transactions_clean, "Transactions")
        social_clean = fix_data_types(social_clean, "Social Profiles")
        
        # Step 3: Justify join logic
        if not justify_join_logic(transactions_clean, social_clean):
            return
        
        # Prepare datasets for merging
        transactions_clean['customer_id_common'] = transactions_clean['customer_id_legacy'].astype(int).astype(str)
        social_clean['customer_id_common'] = social_clean['customer_id_new'].str.replace('A', '').astype(int).astype(str)
        
        # Step 4: Perform inner join
        print("\n--- Performing Merge ---")
        merged_df = pd.merge(
            transactions_clean,
            social_clean,
            on='customer_id_common',
            how='inner',
            validate='many_to_many'  # Allows checking relationship cardinality
        )
        
        print(f"Merged dataset shape: {merged_df.shape}")
        
        # Step 5: Perform post-merge checks
        perform_post_merge_checks(merged_df, transactions_clean, social_clean)
        
        # Step 6: Save the merged dataset
        merged_file_path = 'dataset/merged_customer_data.xlsx'
        merged_df.to_excel(merged_file_path, index=False)
        
        print(f"\nMerge completed successfully!")
        print(f"Merged dataset saved to: {merged_file_path}")
        
        # Final summary
        print(f"\nFinal Summary:")
        print(f"Original transactions: {len(transactions_df)} records")
        print(f"After cleaning: {len(transactions_clean)} records")
        print(f"Original social profiles: {len(social_profiles_df)} records")
        print(f"After cleaning: {len(social_clean)} records")
        print(f"Final merged dataset: {len(merged_df)} records")
        print(f"Merge efficiency: {len(merged_df)/len(transactions_clean)*100:.1f}% of cleaned transactions matched")
        
        # Display sample of merged data
        print(f"\nSample of merged data (first 5 rows):")
        print(merged_df.head(5).to_string())
        
    except FileNotFoundError as e:
        print(f"Error: Required files not found in dataset folder: {e}")
        print("Please ensure both customer_transactions.xlsx and customer_social_profiles.xlsx are in the dataset folder")
    except Exception as e:
        print(f"Error during merge process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    merge_datasets()