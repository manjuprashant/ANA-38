
#1. Data Loading and Initial Assessment

import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import missingno as msno

# Load datasets
def load_json_to_df(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return pd.json_normalize(data)

activities = load_json_to_df('activities_train.json')
won_deals = load_json_to_df('won_deals_train.json')
lost_deals = load_json_to_df('lost_deals_train.json')

# Initial data assessment
def assess_data(df, name):
    print(f"\n{name} Data Assessment:")
    print(f"Shape: {df.shape}")
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
    print("\nSample Data:")
    return df.head()

assess_data(activities, "Activities")
assess_data(won_deals, "Won Deals")
assess_data(lost_deals, "Lost Deals")

# 2. Deal Outcome Analysis
#2.1 Distribution Analysis with Statistical Testing

# Combine won and lost deals with outcome flag
won_deals['outcome'] = 'won'
lost_deals['outcome'] = 'lost'
all_deals = pd.concat([won_deals, lost_deals], ignore_index=True)

# Distribution visualization
fig = px.pie(all_deals, names='outcome', title='Deal Outcome Distribution')
fig.show()

# Statistical significance testing
n_won = len(won_deals)
n_lost = len(lost_deals)
chi2, p = stats.chisquare([n_won, n_lost])
print(f"Chi-square test for outcome distribution: p-value={p:.4f}")

#2.2 Deal Value Analysis

# Compare deal values
plt.figure(figsize=(10,6))
sns.boxplot(x='outcome', y='deal_value', data=all_deals)
plt.title('Deal Value by Outcome')
plt.yscale('log')  # If values span multiple orders of magnitude
plt.show()

# Statistical test
t_stat, p_val = stats.ttest_ind(
    won_deals['deal_value'].dropna(),
    lost_deals['deal_value'].dropna(),
    equal_var=False
)
print(f"Welch's t-test for deal value: p-value={p_val:.4f}")
print(f"Mean won deal value: {won_deals['deal_value'].mean():.2f}")
print(f"Mean lost deal value: {lost_deals['deal_value'].mean():.2f}")

#2.3 Deal Duration Analysis

# Calculate deal duration (assuming we have created_at and updated_at/closed_at)
all_deals['duration_days'] = (all_deals['closed_at'] - all_deals['created_at']).dt.days

# Visualization
fig = px.histogram(
    all_deals, 
    x='duration_days', 
    color='outcome', 
    barmode='overlay',
    title='Deal Duration Distribution by Outcome'
)
fig.show()

# Survival analysis (time-to-event)
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
for outcome in ['won', 'lost']:
    mask = all_deals['outcome'] == outcome
    kmf.fit(all_deals[mask]['duration_days'], label=outcome)
    kmf.plot_survival_function()
plt.title('Survival Analysis of Deal Duration')
plt.show()

#2.4 Success Rate by Deal Characteristics

# Example for industry (adjust for available columns)
if 'industry' in all_deals.columns:
    cross_tab = pd.crosstab(all_deals['industry'], all_deals['outcome'], normalize='index')
    cross_tab['success_rate'] = cross_tab['won'] / (cross_tab['won'] + cross_tab['lost'])
    
    fig = px.bar(
        cross_tab.sort_values('success_rate', ascending=False),
        x=cross_tab.index,
        y='success_rate',
        title='Success Rate by Industry'
    )
    fig.show()
    
    # Chi-square test for independence
    contingency_table = pd.crosstab(all_deals['industry'], all_deals['outcome'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Industry vs Outcome chi-square test: p-value={p:.4f}")
    
   # 3. Activity Pattern Analysis
#3.1 Activity Type Analysis

# Merge activities with deal outcomes
activities_with_outcome = activities.merge(
    all_deals[['deal_id', 'outcome']],
    on='deal_id',
    how='left'
)

# Activity type distribution
activity_counts = activities_with_outcome.groupby(['outcome', 'activity_type']).size().unstack()
activity_percentages = activity_counts.div(activity_counts.sum(axis=1), axis=0)

fig = px.bar(
    activity_percentages.T,
    barmode='group',
    title='Activity Type Distribution by Outcome'
)
fig.show()

# Statistical testing for each activity type
for activity in activities['activity_type'].unique():
    won_acts = activities_with_outcome[
        (activities_with_outcome['outcome'] == 'won') & 
        (activities_with_outcome['activity_type'] == activity)
    ].shape[0]
    
    lost_acts = activities_with_outcome[
        (activities_with_outcome['outcome'] == 'lost') & 
        (activities_with_outcome['activity_type'] == activity)
    ].shape[0]
    
    oddsratio, pvalue = stats.fisher_exact([[won_acts, lost_acts], 
                                           [total_won_acts-won_acts, total_lost_acts-lost_acts]])
    print(f"{activity}: OR={oddsratio:.2f}, p={pvalue:.4f}")
    
    
#3.2 Activity Frequency Patterns 

# Activities per deal
acts_per_deal = activities_with_outcome.groupby(['deal_id', 'outcome']).size().reset_index(name='activity_count')

fig = px.box(
    acts_per_deal,
    x='outcome',
    y='activity_count',
    title='Activity Count per Deal by Outcome'
)
fig.show()

# Statistical test
t_stat, p_val = stats.mannwhitneyu(
    acts_per_deal[acts_per_deal['outcome'] == 'won']['activity_count'],
    acts_per_deal[acts_per_deal['outcome'] == 'lost']['activity_count']
)
print(f"Mann-Whitney U test for activity count: p-value={p_val:.4f}")

#3.3 Activity Timing and Sequence Analysis

# Convert activity timestamps
activities_with_outcome['activity_date'] = pd.to_datetime(activities_with_outcome['created_at'])

# Calculate days since deal creation
activities_with_outcome = activities_with_outcome.merge(
    all_deals[['deal_id', 'created_at']],
    on='deal_id',
    how='left',
    suffixes=('_activity', '_deal')
)
activities_with_outcome['days_since_creation'] = (
    activities_with_outcome['activity_date'] - activities_with_outcome['created_at_deal']
).dt.days

# Activity timing distribution
fig = px.histogram(
    activities_with_outcome,
    x='days_since_creation',
    color='outcome',
    facet_row='activity_type',
    nbins=30,
    title='Activity Timing Distribution by Type and Outcome'
)
fig.show()

# Sequence mining (example for last activity types)
last_activities = activities_with_outcome.sort_values('activity_date').groupby('deal_id').last()
last_activity_dist = pd.crosstab(last_activities['activity_type'], last_activities['outcome'], normalize='index')
print("\nLast Activity Type Before Outcome:")
print(last_activity_dist)

#4. Temporal Pattern Discovery
#4.1 Deal Lifecycle Duration Analysis

# Already covered in 2.3, but can add more granularity
all_deals['duration_weeks'] = all_deals['duration_days'] / 7

# Optimal duration analysis
bins = [0, 1, 2, 4, 8, 12, 26, 52, float('inf')]
labels = ['<1w', '1-2w', '2-4w', '1-2m', '2-3m', '3-6m', '6-12m', '>1y']
all_deals['duration_bin'] = pd.cut(all_deals['duration_days'], bins=bins, labels=labels)

duration_success = pd.crosstab(all_deals['duration_bin'], all_deals['outcome'], normalize='index')
duration_success['success_rate'] = duration_success['won'] / (duration_success['won'] + duration_success['lost'])

fig = px.line(
    duration_success,
    y='success_rate',
    title='Success Rate by Deal Duration'
)
fig.show()

#4.2 Activity Clustering by Time Periods
# Weekly activity patterns
activities_with_outcome['week'] = activities_with_outcome['activity_date'].dt.isocalendar().week

weekly_activity = activities_with_outcome.groupby(['outcome', 'week', 'activity_type']).size().unstack().fillna(0)
weekly_activity = weekly_activity.reset_index().melt(id_vars=['outcome', 'week'], value_name='count')

fig = px.line(
    weekly_activity,
    x='week',
    y='count',
    color='activity_type',
    facet_row='outcome',
    title='Weekly Activity Patterns by Outcome'
)
fig.show()

#5. Text Analytics (for Notes)

from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Filter notes activities
notes = activities_with_outcome[activities_with_outcome['activity_type'] == 'Note']

# Sentiment analysis
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

notes['sentiment'] = notes['content'].apply(get_sentiment)

# Compare sentiment by outcome
fig = px.box(
    notes,
    x='outcome',
    y='sentiment',
    title='Note Sentiment by Deal Outcome'
)
fig.show()

# Word clouds
for outcome in ['won', 'lost']:
    text = ' '.join(notes[notes['outcome'] == outcome]['content'].astype(str))
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud)
    plt.title(f'Word Cloud for {outcome} deals')
    plt.axis('off')
    plt.show()

# Topic modeling
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
notes_counts = vectorizer.fit_transform(notes['content'].astype(str))
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(notes_counts)

# Display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lda, vectorizer.get_feature_names_out(), 10)

#6. Advanced Statistical Analysis
#6.1 Correlation Analysis

# Create activity features per deal
activity_features = pd.get_dummies(activities_with_outcome, columns=['activity_type'])
activity_features = activity_features.groupby(['deal_id', 'outcome']).agg({
    'activity_type_Email': 'sum',
    'activity_type_Meeting': 'sum',
    'activity_type_Call': 'sum',
    'activity_type_Task': 'sum',
    'activity_type_Note': 'sum',
    'days_since_creation': ['mean', 'std', 'max']
}).reset_index()
activity_features.columns = ['_'.join(col).strip() for col in activity_features.columns.values]

# Merge with deal value
analysis_df = activity_features.merge(all_deals[['deal_id', 'deal_value', 'duration_days']], on='deal_id')

# Correlation matrix
corr_matrix = analysis_df.corr()
fig = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    title='Feature Correlation Matrix'
)
fig.show()

#6.2 Principal Component Analysis

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Prepare data
X = analysis_df.select_dtypes(include=[np.number]).dropna()
X = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
pca_df['outcome'] = analysis_df['outcome_'].values

# Visualize
fig = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color='outcome',
    title='PCA of Sales Activities and Deal Characteristics'
)
fig.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

