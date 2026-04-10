```
I have plotted the correlation matrix of all numeric columns in `df_clean` to show their interrelationships. Additionally, I have visualized outliers for each numeric feature using box plots, which clearly highlight data points falling outside the typical range.

Here are the visualizations:

1.  **Correlation Matrix of Numeric Columns**: This heatmap provides a quick overview of how strongly each pair of numeric variables is correlated. Positive correlations are shown in warmer colors, and negative correlations in cooler colors.

2.  **Outlier Visualization with Box Plots**: For each numeric column, a box plot is generated. Box plots are effective in displaying the distribution of a dataset based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum. Any data points that fall outside 1.5 times the interquartile range (IQR) from Q1 or Q3 are considered potential outliers and are plotted individually.
```