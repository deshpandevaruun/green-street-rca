#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:57:42 2024

@author: varuundeshpande
"""

import pandas as pd
import numpy as np
import blosc
import brotli
import pickle
import sys, os
import linktransformer as lt
import scipy
from fuzzywuzzy import fuzz
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from itertools import product
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def readCompressedPickle(path, file):
   
    with open(os.path.join(path, file), "rb") as f:
        compressed_pickle = f.read()
 
    if file.endswith('.blc'):
        depressed_pickle = blosc.decompress(compressed_pickle)
    elif file.endswith('.blt'):
        depressed_pickle = brotli.decompress(compressed_pickle)
    else:
        raise ValueError
    data = pickle.loads(depressed_pickle)
    return data 

def split_buyer_seller_names(green_street, column_name):
    # Split the values in 'column_name' based on semicolon and create new columns dynamically

    green_street_buyer_split = green_street[column_name].str.split(';', expand=True)
    # Rename the columns to Column_name_1, ...
    green_street_buyer_split.columns = [f'{column_name}_{i+1}' for i in range(green_street_buyer_split.shape[1])]

    # Concatenate the original DataFrame with the split DataFrame
    green_street = pd.concat([green_street, green_street_buyer_split], axis=1)
    green_street = green_street.drop(column_name, axis=1)

    return green_street

#measure semantics score 
def semantics_score(s1, s2):
    if pd.isna(s1) or pd.isna(s2):
        return np.nan
    else:
    
        embedding_1= model.encode(s1, convert_to_tensor=True)
        embedding_2 = model.encode(s2, convert_to_tensor=True)
    
        score = util.pytorch_cos_sim(embedding_1, embedding_2)
        return score.item()


#Fuzzy Wuzy method
def method_fuzzywuzzy_2(rca, row):
    '''
    Function 1
    calculate fuzzy score between the passed property name from green street and every name in RCA and get >0.95 values
    '''
    col_name = 'Name_FuzzyScore'
    rca[col_name] = rca['propertyname_raw'].apply(lambda x: fuzz.ratio(str(x), str(row)))
    mask = rca[col_name] > 0.95  
    indices = rca.index[mask].tolist()
    return indices

def method_fuzzywuzzy(rca, green_street):

    green_street['highest_match_index_loc'] = green_street['Portfolio / Property Name'].apply(lambda row: method_fuzzywuzzy_2(rca, row))
    print(green_street)
    # Step 1: Explode the 'keys' column in green_street to create a long DataFrame
    green_street_exploded = green_street.explode('highest_match_index_loc')
    
    # Step 2: Convert rca's index to a column if it's not already a column, for merging
    rca_reset = rca.reset_index()
    
    # Merge A_exploded with B based on the exploded 'keys' column and B's index
    merged_df = pd.merge(green_street_exploded, rca_reset, left_on='highest_match_index_loc', right_on='index', how='left')
    
    return merged_df


#sentence transformer
def method_sentence_transformer_2(rca, row):
    '''
    Function 2
    calculate fuzzy score between the passed property name from green street and every name in RCA and get >0.95 values
    '''
    col_name = 'Name_Score'
    rca[col_name] = rca['propertyname_raw'].apply(lambda x: cosine_similarity(model.encode([str(x), str(row)]))[0][1])
    mask = rca[col_name] > 0.9
    indices = rca.index[mask].tolist()
    return indices

def method_sentence_transformer(rca, green_street):

    green_street['highest_match_index_loc'] = green_street['Portfolio / Property Name'].apply(lambda row: method_sentence_transformer_2(rca, row))
    print(green_street)
    # Step 1: Explode the 'keys' column in green_street to create a long DataFrame
    green_street_exploded = green_street.explode('highest_match_index_loc')
    
    # Step 2: Convert rca's index to a column if it's not already a column, for merging
    rca_reset = rca.reset_index()
    
    # Merge A_exploded with B based on the exploded 'keys' column and B's index
    merged_df = pd.merge(green_street_exploded, rca_reset, left_on='highest_match_index_loc', right_on='index', how='left')
    
    return merged_df


def method_link_transformer(rca, green_street):
    '''

    Parameters
    ----------
    rca : TYPE
        DESCRIPTION.
    green_street : TYPE
        DESCRIPTION: Merge rca and green_street on name using link_transformer. Merge n times such that after every iteration, those transactions are 
        removed from rca and the merge is done on this new rca. Keep only those transactions in merge that have score more than 90%
    n: maximum number of iterations
    Returns: merged dataset
    -------

    '''
    nn = 10
    rca_copy = rca.copy()
    rca_columns = rca_copy.columns
    merged = pd.DataFrame()
    for ii in range(nn):
        df_lm_matched = lt.merge(green_street, rca_copy, merge_type='1:m', model="dell-research-harvard/lt-wikidata-comp-en", left_on=['Portfolio / Property Name'], right_on=['propertyname_raw'])    
        
        #get rca transactions in merged dataframe
        matched_rca = df_lm_matched[rca_columns]
        
        #remove above transactions from rca_copy
        merged_removal = pd.merge(rca_copy, matched_rca, indicator=True, how='outer')
        rca_copy = merged_removal[merged_removal['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        #remove where match is less than 0.85
        df_lm_matched = df_lm_matched[df_lm_matched['score'] >= 0.85]
        
        merged = pd.concat([merged, df_lm_matched], ignore_index=True)
    
    return merged
        
def date_adjustment(merged):
    '''
    

    Parameters
    ----------
    merged : TYPE
        DESCRIPTION: gets 1:m dataframe and removes those matches that are more than 30 days and 90 days apart.

    Returns dataframe that are within 30 and 90 days apart
    -------
    None.

    '''
    
    t1 = 30
    t2 = 90
    # Calculate the absolute difference between the two timestamp columns
    merged['date_diff'] = (merged['Sale Date'] - merged['status_dt']).abs()
    
    # Filter rows where the difference is 30 days or less
    merged_30_days = merged[merged['date_diff'] <= pd.Timedelta(days=t1)]
    
    # Drop the 'date_diff' column if you no longer need it
    merged_30_days = merged_30_days.drop(columns=['date_diff'])
    
    # Filter rows where the difference is 90 days or less
    merged_90_days = merged[merged['date_diff'] <= pd.Timedelta(days=t2)]
    
    # Drop the 'date_diff' column if you no longer need it
    merged_90_days = merged_90_days.drop(columns=['date_diff'])
    
    return merged_30_days, merged_90_days


def find_max_per_row(row, comparison_results):
    '''
    

    Parameters
    ----------
    row : TYPE
        DESCRIPTION.
    comparison_results : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # Split the column names to separate col1 and col2, and initialize a dictionary to store max values and column names
    max_values = {col.split('_vs_')[0]: {"MaxValue": None, "Col2": None} for col in comparison_results.columns}
    
    # Iterate over each column in the row to update the dictionary with max values and corresponding col2 names
    for col, value in row.iteritems():
        col1, col2 = col.split('_vs_')
        if max_values[col1]["MaxValue"] is None or value > max_values[col1]["MaxValue"]:
            max_values[col1]["MaxValue"] = value
            max_values[col1]["Col2"] = col2
    
    # Convert the dictionary into a Series to return
    return pd.Series({col1: f'{max_info["Col2"]} ({max_info["MaxValue"]})' for col1, max_info in max_values.items()})


def buyer_seller_match(merged, similarity_function):
    '''
    

    Parameters
    ----------
    merged : TYPE
        DESCRIPTION.

    Returns : reduced dataframes with buyers and sellers matching atleast 80%
    -------
    None.

    '''
    rca_buyers = ['buyername1', 'buyername2', 'buyername3', 'buyername4']
    
    green_street_buyers = ['Buyer_1', 'Buyer_2', 'Buyer_3', 'Buyer_4']
    
    column_pairs_buyers = list(product(green_street_buyers, rca_buyers))
    
    rca_sellers = ['sellername1', 'sellername2', 'sellername3', 'sellername4']
    green_street_sellers = ['Seller_1','Seller_2', 'Seller_3', 'Seller_4']
    column_pairs_sellers = list(product(green_street_sellers, rca_sellers))
    
    comparison_results_buyers = pd.DataFrame(index=merged.index)
    
    comparison_results_sellers = pd.DataFrame(index=merged.index)

    # Apply the sematics_score/Jaccard Score for each pair across all rows
    for col1, col2 in column_pairs_buyers:
        comparison_results_buyers[f'{col1}_vs_{col2}'] = merged.apply(lambda row: similarity_function(row[col1], row[col2]), axis=1)
        
    for col1, col2 in column_pairs_sellers:
        comparison_results_sellers[f'{col1}_vs_{col2}'] = merged.apply(lambda row: similarity_function(row[col1], row[col2]), axis=1) 
    
    max_comparison_buyers = pd.DataFrame(comparison_results_buyers.apply(lambda row:find_max_per_row(row, comparison_results_buyers), axis=1))
    
    max_comparison_sellers = pd.DataFrame(comparison_results_sellers.apply(lambda row:find_max_per_row(row, comparison_results_sellers), axis=1))
    
    
    def split_match_columns(df):
        score_cols = []
        for col in df.columns:
            # Split the 'Buyer (Score)' format into two columns: 'BuyerName' and 'Score'
            df[[f'{col}', f'{col}_score']] = df[col].str.extract(r'([^\(]+)\s+\(([^)]+)\)')
            # Convert score column to numeric
            df[f'{col}_score'] = pd.to_numeric(df[f'{col}_score'], errors='coerce')
            # Keep track of score columns for average calculation
            score_cols.append(f'{col}_score')
            # Drop the original combined column
            df.drop(col, axis=1, inplace=True)
        # Calculate average score excluding NaN values
        df['Average_Score'] = df[score_cols].mean(axis=1, skipna=True)
        return df
    
    # Splitting the match results into separate columns for both buyers and sellers
    max_comparison_buyers = split_match_columns(max_comparison_buyers)
    max_comparison_sellers = split_match_columns(max_comparison_sellers)
    
    #Average mathcing greater than 80%
    max_comparison_buyers = max_comparison_buyers[(max_comparison_buyers['Average_Score'] >= 0.8) | (max_comparison_buyers['Average_Score'].isna())]
    max_comparison_sellers = max_comparison_sellers[(max_comparison_sellers['Average_Score'] >= 0.8) | (max_comparison_sellers['Average_Score'].isna())]
    
    #Get union of indices of both the above
    combined_indices = max_comparison_buyers.index.union(max_comparison_sellers.index)
    
    merged_reduced = merged.loc[merged.index.intersection(combined_indices)]
    return merged_reduced
    
    
def Jaccard_Similarity(doc1, doc2): 
  if pd.isna(doc1) or pd.isna(doc2):
      return np.nan
  # List the unique words in a document
  else:
      words_doc1 = set(doc1.lower().split()) 
      words_doc2 = set(doc2.lower().split())
      
      # Find the intersection of words list of doc1 & doc2
      intersection = words_doc1.intersection(words_doc2)
    
      # Find the union of words list of doc1 & doc2
      union = words_doc1.union(words_doc2)
          
      # Calculate Jaccard similarity score 
      # using length of intersection set divided by length of union set
      return float(len(intersection)) / len(union)  


#def zip_code_match(merged):
    
def main():
    data_path = "/Users/varuundeshpande"
    
    final_data = readCompressedPickle(data_path, 'RCA_O1B.blc')
    columns = ['propertykey_id', 'statuspriceadjustedusd_amt_cpiNOTadj', 'price_cpiNOTadj', 'propertykey_id_typeadj', 'propertytype_adj', 'propertyname_raw',\
               'status_dt', 'intconveyed_nb', 'intconveyed','address' , 'buyername1','buyername2','buyername3',
     'buyername4',
     'sellername1',
     'sellername2',
     'sellername3',
     'sellername4', 'zip_cd', 'psf/ppu', 'sqft_nb_OG', 'units_OG' ]
    final_data_reduced = final_data[columns]
    green_street = pd.read_csv("green_street_2019.csv")
    final_data_reduced['status_dt'] = pd.to_datetime(final_data_reduced['status_dt'])
    final_data_reduced.drop_duplicates(inplace=True)

    #get 2019 dta RCA
    rca_2019 = final_data_reduced[final_data_reduced['status_dt'].dt.year == 2019]
    #get 2019 green street data
    green_street['Sale Date'] = pd.to_datetime(green_street['Sale Date'])
    green_street_2019 = green_street[green_street['Sale Date'].dt.year == 2019]
    
    #Split buyers
    green_street_2019 = split_buyer_seller_names(green_street_2019, 'Buyer') 

    #Split Sellers
    green_street_2019 = split_buyer_seller_names(green_street_2019, 'Seller')
   
  
    
    #Step 1, match 1:m on names
    step_1_merged = method_link_transformer(rca_2019, green_street_2019)
    step_1_merged.to_csv("step_1_green_street_match.csv")
    
    
    #Step 2, get transactions that are within 30 days and 90 days of each other
    step_1_merged = pd.read_csv("step_1_green_street_match.csv")
    step_1_merged['Sale Date'] = pd.to_datetime(step_1_merged['Sale Date'])
    step_1_merged['status_dt'] = pd.to_datetime(step_1_merged['status_dt'])
    
    step_2_merged_30_days, step_2_merged_90_days  = date_adjustment(step_1_merged)
    step_2_merged_30_days.to_csv('step_2_merged_30_days.csv')
    step_2_merged_90_days.to_csv('step_2_merged_90_days.csv')
    
    
    #Step 3, match on buyers and sellers
    step_2_merged_30_days = pd.read_csv("step_2_merged_30_days.csv")
    step_3_merged_buyers_sellers_reduced_30_days = buyer_seller_match(step_2_merged_30_days, semantics_score) #Semantics score
    step_3_merged_buyers_sellers_reduced_30_days.to_csv('step_3_30_days.csv')
    
    step_3_merged_buyers_sellers_reduced_30_days_jaccard = buyer_seller_match(step_2_merged_30_days, Jaccard_Similarity) #Jaccard
    step_3_merged_buyers_sellers_reduced_30_days_jaccard.to_csv("step_3_30_days_jaccard.csv")
    
    step_2_merged_90_days = pd.read_csv("step_2_merged_90_days.csv")
    step_3_merged_buyers_sellers_reduced_90_days = buyer_seller_match(step_2_merged_90_days, semantics_score) #semantics score
    step_3_merged_buyers_sellers_reduced_90_days.to_csv('step_3_90_days.csv')
    
    step_3_merged_buyers_sellers_reduced_90_days_jaccard = buyer_seller_match(step_2_merged_90_days, Jaccard_Similarity) #Jaccard
    step_3_merged_buyers_sellers_reduced_90_days_jaccard.to_csv("step_3_90_days_jaccard.csv")    
    
if __name__ == '__main__':
    main()