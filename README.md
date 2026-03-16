Customer Segmentation Clustering API

A customer segmentation microservice that performs clustering on prepared database views and stores segmentation results in a cloud database.The system exposes a REST API built with FastAPI that loads customer datasets, performs feature engineering, selects the appropriate clustering model, and stores the resulting clusters in a PostgreSQL database hosted on Supabase.The API supports two clustering strategies:
Aggregated Model (Model A) – segmentation based on high-level revenue and tenure metrics
Behavioral Model (Model B) – segmentation based on customer purchasing behaviour

Model Selection Logic

The API automatically selects which clustering model to apply based on feature availability and confidence levels.The behavioral model requires a reliable membership duration feature (membership_years).If the system detects low confidence in this feature (membership_confidence = 'low'), the API automatically falls back to the aggregated model, which does not depend on derived behavioral features.This dynamic routing ensures segmentation can still be performed even when behavioural data quality is insufficient.

Exploratory Data Analysis (EDA)

The preprocessing stage included:
Checking null values,Validating data types, Assessing feature skewness, Separating numeric and categorical features, Missing values were handled as follows:Numeric features → median imputation (chosen due to right-skewed distributions) and Categorical features → mode imputation.

Feature Engineering

Three behavioural signals were derived:

Purchases_Per_Year
(Total purchases ÷ membership years), Spend_Per_Purchase
(Customer lifetime value ÷ number of purchases), Recency_Score
(1 ÷ days since last purchase).These features capture purchase intensity, customer value, and recency behaviour, to address heavy right-skew, all behavioural features were log-transformed. 

The aggregated model uses higher-level financial indicators:

log-transformed monthly_fee, log-transformed total_revenue, raw tenure_months.These features allow segmentation even when behavioral signals are unavailable. 

API => https://customer-clustering-api-4.onrender.com/docs
