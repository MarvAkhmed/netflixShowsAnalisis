import kaggle
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Authenticate with Kaggle API
kaggle.api.authenticate()

# Download the Netflix dataset from Kaggle
kaggle.api.dataset_download_files('ariyoomotade/netflix-data-cleaning-analysis-and-visualization', path='.', unzip=True)

# Fetch metadata of the dataset
kaggle.api.dataset_metadata('ariyoomotade/netflix-data-cleaning-analysis-and-visualization', path='.')

# Get the file paths in the dataset directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the dataset into a DataFrame
df = pd.read_csv('/Users/marwa_awwad_mohamed_awwad/Desktop/python/machineLearningHw/Netflix/netflix1.csv')

# Drop unnecessary columns
df.drop(columns=['show_id'], inplace=True)

# Handle data quality issues
df['type'] = df['type'].astype('category')
df['date_added'] = pd.to_datetime(df['date_added'])
df['release_year'] = df['release_year'].astype(int)

# Handle structural and consistency issues
all_index_of_season = df[df['duration'].str.contains('Season')].index.to_list()
tem_df = df.drop(all_index_of_season).reset_index(drop=True)
tem_df['duration'] = tem_df['duration'].str.split(' ').str.get(0).astype(int)

# Visualize the top 10 movie directors
tem_df['director'].value_counts().head(10).plot.barh(title='Top 10 Movies Directors', grid=True, color='black')
plt.show()

# Visualize the types of shows in the dataset
types_of_shows = df['type'].value_counts()
fig = px.bar(types_of_shows, color=types_of_shows.index, color_discrete_sequence=["#8a12f9", "#FFFFFF"], text_auto=True, template='plotly_dark')
fig.show()

# Visualize the number of movies and TV shows released every year
fig = px.area(df['release_year'].value_counts(), color_discrete_sequence=["#a8f53d"], template='plotly_dark')
fig.show()

# Show ratings from lowest to highest
rating_counts = df['rating'].value_counts(sort=False)
sorted_rating_counts = rating_counts.sort_index()
print(sorted_rating_counts)

# Visualize ratings frequency
fig = px.bar(data_frame=df['rating'].value_counts(), template='plotly_dark',
             x=df['rating'].value_counts(), y=df['rating'].value_counts().index,
             orientation='h', title='Most ratings', labels={"x": "frequency"})
fig.show()

# Use t-SNE for dimensionality reduction and visualization
df.dropna(subset=['country'], inplace=True)
countries = df['country'].value_counts()[:10]
df_top_countries = df[df['country'].isin(countries.index)]
df_encoded = pd.get_dummies(df_top_countries['country'])
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(df_scaled)

# Plot the t-SNE results
plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
plt.title('t-SNE Visualization of Top 10 Countries')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# Drop missing values
df.dropna(inplace=True)

# Visualize the top 5 directors
fig = px.bar(data_frame=df['director'].value_counts()[0:5], x=df['director'].value_counts()[0:5],
             y=df['director'].value_counts()[0:5].index, color=df['director'].value_counts()[0:5].index,
             text_auto=True, orientation="h", template='plotly_dark')
fig.show()

# Visualize the top 10 countries by frequency
countries = df['country'].value_counts()[:10]
plt.figure(figsize=(10, 6))
countries.plot(kind='bar')
plt.title('Top 10 Countries by Frequency')
plt.xlabel('Country')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()
