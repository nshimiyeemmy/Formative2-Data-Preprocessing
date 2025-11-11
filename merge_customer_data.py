import pandas as pd
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

def merge_datasets():
    """Merge customer_transactions and customer_social_profiles datasets"""
    
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
        
        # Display column names to understand the structure
        print("\nTransactions columns:", transactions_df.columns.tolist())
        print("Social profiles columns:", social_profiles_df.columns.tolist())
        
        # First, let's see the unique customer IDs in both datasets
        transactions_customers = set(transactions_df['customer_id_legacy'].astype(int).astype(str))
        social_customers = set(social_profiles_df['customer_id_new'].str.replace('A', '').astype(int).astype(str))
        
        print(f"\nUnique customers in transactions: {len(transactions_customers)}")
        print(f"Unique customers in social profiles: {len(social_customers)}")
        
        # Find common customers (remove 'A' prefix from social profiles for matching)
        common_customers = transactions_customers.intersection(social_customers)
        print(f"Common customers: {len(common_customers)}")
        
        if len(common_customers) == 0:
            print("Warning: No common customers found between datasets!")
            print("Sample transaction customers:", list(transactions_customers)[:5])
            print("Sample social profile customers:", list(social_customers)[:5])
            return
        
        # Prepare datasets for merging
        # Create a common customer ID in both datasets
        transactions_df['customer_id_common'] = transactions_df['customer_id_legacy'].astype(int).astype(str)
        social_profiles_df['customer_id_common'] = social_profiles_df['customer_id_new'].str.replace('A', '').astype(int).astype(str)
        
        # Perform inner join on common customer IDs
        print("\nMerging datasets...")
        merged_df = pd.merge(
            transactions_df,
            social_profiles_df,
            on='customer_id_common',
            how='inner'
        )
        
        print(f"Merged dataset shape: {merged_df.shape}")
        
        # Save the merged dataset
        merged_file_path = 'dataset/merged_customer_data.xlsx'
        merged_df.to_excel(merged_file_path, index=False)
        
        print(f"\nMerged dataset saved to: {merged_file_path}")
        print("Merge completed successfully!")
        
        # Display some statistics about the merged data
        print(f"\n--- Merge Statistics ---")
        print(f"Original transactions records: {len(transactions_df)}")
        print(f"Original social profiles records: {len(social_profiles_df)}")
        print(f"Merged records: {len(merged_df)}")
        print(f"Merge success rate: {len(merged_df)/len(transactions_df)*100:.2f}% of transactions matched")
        
        # Display first few rows of merged data
        print(f"\nFirst 5 rows of merged data:")
        print(merged_df.head())
        
    except FileNotFoundError as e:
        print(f"Error: Required files not found in dataset folder: {e}")
        print("Please ensure both customer_transactions.xlsx and customer_social_profiles.xlsx are in the dataset folder")
    except Exception as e:
        print(f"Error during merge process: {e}")

if __name__ == "__main__":
    merge_datasets()