# Executive Summary
This report details the preparation, cleaning, and initial exploratory data analysis of a dataset containing car service and accident records. The dataset, comprising 99,817 entries across 22 columns, underwent a thorough cleaning process to address missing values and data type inconsistencies. Feature engineering and scaling were applied to enrich the dataset for further analysis. Key findings indicate significant variations in car prices based on fuel type, non-normal distribution of car prices, and specific service types leading to higher costs. Anomalies in repair cost ratios related to accident severity were also identified.

## Data
The initial dataset `df_raw` contained 99,817 rows and 22 columns, with a mix of object, float64, and int64 data types.

### Initial Observations:
*   **Dimensions:** (99817, 22)
*   **Missing Values:** Significant missing data in 9 columns, primarily related to service and accident records:
    *   `ServiceID`, `Date_of_Service`, `ServiceType`, `Cost_of_Service`: ~20.97% missing.
    *   `AccidentID`, `Date_of_Accident`, `Description`, `Cost_of_Repair`, `Severity`: ~14.03% missing.
*   **No Duplicate Rows:** The dataset contained no exact duplicate rows.
*   **Numeric Ranges:** Numeric columns like `Engine size`, `Year_of_Manufacturing`, `Mileage`, `Price`, `Cost_of_Service`, and `Cost_of_Repair` showed broad and reasonable ranges.
*   **Categorical Issues:** Unexpected category levels were found in `Manufacturer` (['VW', 'Porsche']), `Fuel_Type` (['Hybrid']), and `Severity` (['Severe']), indicating potential data entry inconsistencies or deviations from expected values.

## Cleaning
The data cleaning process aimed to ensure data quality and consistency without reducing the number of observations:

*   **No Row Dropping:** No rows were removed due to high missing values or exact duplication, maintaining the initial 99,817 records.
*   **Data Type Conversion:**
    *   `Date_of_Service` and `Date_of_Accident` were converted to datetime objects.
    *   `Cost_of_Service` and `Cost_of_Repair` were converted to numeric (float64).
    *   `ServiceID` and `AccidentID` were ensured to be object type, handling string 'nan' values.
*   **Missing Value Imputation:** Missing values were handled based on column type:
    *   **Categorical/ID Columns (mode imputation):** `ServiceID` (S00192), `ServiceType` (Suspension Check), `AccidentID` (A00108), `Description` (Side mirror broken), `Severity` (Moderate).
    *   **Date Columns (mode imputation):** `Date_of_Service` (2024-05-21), `Date_of_Accident` (2022-11-06).
    *   **Numeric Columns (median imputation):** `Cost_of_Service` (275.00), `Cost_of_Repair` (2536.00).
*   **Outlier Handling:** No capping of extreme numeric outliers was performed, as no specific issues were reported.

## EDA
Exploratory Data Analysis revealed several patterns and relationships:

*   **Feature Engineering:**
    *   `Service_Lead_Time`: Days between `Date_of_Accident` and `Date_of_Service`.
    *   `Repair_Cost_Ratio`: `Cost_of_Repair` divided by `Cost_of_Service`.
    *   `Service_Month`: Extracted from `Date_of_Service`.
*   **Scaling:** Key numeric features (`Cost_of_Service`, `Cost_of_Repair`, `Service_Lead_Time`, `Repair_Cost_Ratio`) were scaled using MinMaxScaler, transforming their values to a range between 0 and 1.
*   **One-Hot Encoding:** Categorical features `ServiceType`, `Description`, and `Severity` were one-hot encoded, creating 12, 22, and 4 new columns respectively.
*   **Dropped Columns:** Original `ServiceID`, `Date_of_Service`, `AccidentID`, and `Date_of_Accident` columns were dropped after transformations.
*   **Descriptive Statistics Highlights:**
    *   `Cost_of_Service`: Mean 274.69, Std 115.84 (Range: 50.00-500.00).
    *   `Cost_of_Repair`: Mean 2542.73, Std 1313.66 (Range: 100.00-5000.00).
    *   `Service_Lead_Time`: Mean 341.47, Std 423.73 (Range: 0.00-1810.00).
    *   `Repair_Cost_Ratio`: Mean 12.30, Std 11.70 (Range: 0.21-99.10).
    *   `Price`: Mean 13794.03, Std 16326.66 (Range: 76.00-167774.00).
*   **Visualizations:**
    *   Distributions of `Cost_of_Repair` and `Car Price`.
    *   A Correlation Heatmap of key numeric variables.
    *   A Box Plot of `Cost_of_Repair` to visualize outliers.

## Statistics

*   **Normality Test (Shapiro-Wilk) for Price:**
    *   p-value: 0.0000
    *   **Interpretation:** The car `Price` distribution is significantly different from a normal distribution.
*   **Group Comparison Test (ANOVA) for Price by Fuel_Type:**
    *   p-value: 0.0000
    *   **Interpretation:** There is a statistically significant difference in the average car prices across different `Fuel_Type` categories.

## Key Insights

*   **Service and Accident Data Completeness:** A significant portion of the dataset (14-21%) initially lacked service and accident records, necessitating imputation. The choice of mode and median for imputation reflects a strategy to preserve the overall distribution characteristics for categorical and numeric data, respectively.
*   **Impact of Service Type on Cost:** "Major Service" commands a notably higher average cost compared to other service types, indicating its substantial financial implication for car owners.
*   **Repair Cost Anomalies and Severity:** Incidents classified with 'Moderate' severity frequently exhibit the highest Repair_Cost_Ratios. This suggests that while not 'Severe', these events lead to disproportionately high repair expenses relative to the service cost, warranting further investigation into the nature of these 'moderate' accidents.
*   **Car Price and Fuel Type:** Car prices vary significantly based on the fuel type, suggesting `Fuel_Type` is a strong determinant of vehicle value.
*   **Feature Engineering Value:** The creation of `Service_Lead_Time` and `Repair_Cost_Ratio` provides valuable derived metrics that can enrich predictive models by capturing temporal relationships and cost efficiencies.
*   **Data Distribution:** The non-normal distribution of `Price` suggests that models assuming normality for this variable might be inappropriate, and robust alternatives should be considered.
*   **Correlation Trends (as described in the analysis):**
    *   A strong positive relationship is expected between `Cost_of_Service` and `Cost_of_Repair`, suggesting that higher service costs are generally associated with higher repair costs.
    *   Higher car prices tend to correlate with higher repair costs, possibly due to more expensive parts or specialized labor for premium vehicles.
    *   Higher mileage cars might exhibit longer service lead times, potentially indicating less frequent or delayed maintenance for vehicles with extensive use.