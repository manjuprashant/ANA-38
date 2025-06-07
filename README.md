# ANA-38

Exploratory Data Analysis:
  This tasks involves thorough exploratory data analysis of underlying 
  patterns, relationships and characteristics of sales data. The analysis 
  must go beyond basic statistics to uncover actionsble insights about what 
  distinguishes winning deals from losing deals, how different activity 
  types contribute to deal success, what temporal patterns exist in
  successful sales patterns.
  
  EDA should provide a foundation for informed feature engineering
  decisions and model architectural choices. It must examine both
  individual activity patterns and complex multi activity interactions 
  that might indicate successful deal progression. The analysis should 
  also identify potential data quality issues, outliers and edge cases
  that need special handling in the pre-processing pipeline. 
  
  Files:
  1. activities_train.json
  2. won_deals_train.json
  3. lost_deals_train.json
  
  Detailed Technical Requirements:
  
  1. Deal Outcome Analysis:
     1. Distribution analysis of won vs lost deals with statistical 
        significance testing.
        
     2. Deal value analysis and correlation with activities.
     
     3. Deal duration analysis and relationship to outcome.
     
     4. Success rate analysis by deal chatacteristics(size, source, industry if available)
     
  2. Activity Pattern Analysis:
    
       1. Comprehensive analysis of all 5 activity types (Email, meetings, 
          Tasks, Notes, Calls).
   
       2. Activity frequency patterns for won vs lost deals.
       
       3. Activity timing and sequence analysis.
       
       4. Activity intensity patterns throghout deal lifecycle.
       
       5. Cross-activity correlation analysis.
       
  3. Temporal Pattern Discovery:
      
       1. Deal lifecycle duration analysis 
       
       2. Activity clustering by time periods.
       
       3. Seasonal Patterns in deal closure.
       
       4. Activity velocity analysis.(rate of activity over time).
       
       5. Critical time period identification (most important phases).
       
   4. Text Analytics(for Notes and recommendations):
   
       1. Sentiment analysis of communication content.
       
       2. Topic modelling to identify common themes.
       
       3. Communication style analysis.
       
       4. Key phrase extraction and frequency analysis.
  
    5. Advanced Statistical Analysis:
    
       1. Correlation matrices with significance testing.
       
       2. Principal Component Analysis for dimensionality understanding 
       
       3. Clustering analysis to identify deal archetypes.
  
       4. Outlier detection and analysis.
       
       5. Missing data pattern analysis.
       
    Deliverables:
    
    1. Interactive Analysis Notebooks:
       Complete Jupyter Notebooks with:
       1. Executive summary sections for business stakeholders.
       
       2. Detailed technical analysis with statistical validation.
       
       3. Interactive visualizations using plotly/bokeh for exploration.
       
       4. Code documention and explanation of methodologies.
       
    2. Statistical Analysis Report:
    
       Comprehensive statistical findings including:
       
       1. Hypothesis testing results for key business questions 
       
       2. Effect size calculations for significant relationships.
        
       3. Confidence intervals for key metrics.
       
       4. Power analysis for sample size adequency.
       
    3. Visualization Portfolio:  
     
     
       Professional quality visualizations including:
       
       1. Distribution plots for all key vasriables.
       
       2. Correlation heatmaps with clustering.
       
       3. Time Series plots for temporal patterns.
       
  
    Detailed Acceptance Criteria:
    
    1. EDA covers comprehensive analysis of all 5 activity types with
       statistical validation.
       
    2. Statistical significance is calculated and documented for all claimed 
       patterns.
       
    3. Visualizations clearly comnunicate findings and are publication- ready.
    
    4. Missing data patterns are identified and documented with impact 
       assessment.
       
    5. Outliers are identified , analysed and categorized. (valid verses data errors)
     
    6. Text analysis provides meaningful insights from unstructured data.
    
    7. Temporal patterns are analysed with appropriate time series techniques.
    
    8. Cross activity relationships are explored and quantified.
    
    9. Business implications of all findings are clearly articulated.
    
    10. Code is well documented and reproducible.
    
    11. Analysis methodology is sound and follows statistical beat practices.
    
    12. Recommendations for feature engineering are data driven and specific. 
    
