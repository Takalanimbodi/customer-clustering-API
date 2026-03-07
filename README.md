# Clustering API

This project provides a **Python API** to perform clustering  from clean views for both aggregated model and behavioral model and store the results on PostgreSQL database-supabase depending on the view which was loaded. 
If the dataset cantains features for both aggregated model and behavioral model then all the views will be loaded with data depending on the features for models.
This api performs feature engineering for selecting which model to use depending on confidence on features mostly for membership_years feature for behavioral model if the membership_confidence => 'low' then it fallback on aggregated model since it does not require feature derivations.
Results are stored on clustered_results table(aggreageted/behavioral).
It supports both **Aggregated Model (Model A)** and **Behavioral Model (Model B)**including membership normalization.

