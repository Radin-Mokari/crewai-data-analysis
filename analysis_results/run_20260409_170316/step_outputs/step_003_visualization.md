```
The following visualizations provide a minimal analysis of `df_clean`, focusing on key distributions and relationships as requested.

### 1. Distribution of ORIGINAL_Median House Value
This histogram shows the distribution of housing prices. It appears skewed to the left (negatively skewed) with a long tail towards higher values, indicating that most houses are concentrated at lower to mid-range values, but there are a significant number of houses with very high values, including a cap at $500,000.

### 2. Distribution of ORIGINAL_Median Income
This histogram illustrates the distribution of median income in the districts. It shows a right-skewed (positively skewed) distribution, meaning most districts have lower median incomes, with fewer districts having very high incomes. There is a clear peak at lower income ranges.

### 3. ORIGINAL_Median Income vs ORIGINAL_Median House Value
This scatter plot reveals a strong positive correlation between `ORIGINAL_median_income` and `ORIGINAL_median_house_value`. As median income increases, the median house value generally increases. There's a visible ceiling on `ORIGINAL_median_house_value` at $500,000, where data points accumulate, suggesting potential data capping or truncation for the highest values.

### 4. Distribution of ORIGINAL_Ocean Proximity
This bar chart displays the frequency of different `ORIGINAL_ocean_proximity` categories. The majority of districts are 'NEAR BAY' or '<1H OCEAN', followed by 'INLAND'. 'NEAR OCEAN' and 'ISLAND' are less common, with 'ISLAND' being very rare.

```