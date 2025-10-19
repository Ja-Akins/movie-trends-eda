import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter

# ---------------------------------------------------
# 1. Page Setup
# ---------------------------------------------------
st.set_page_config(
    page_title="Movie Trends Dashboard",
    layout="wide",
    page_icon="🎬"
)

st.title("🎥 Exploratory Data Analysis of Movie Trends")
st.markdown("Analyze movie performance by **genre**, **year**, and **financial metrics**.")

# Apply styling
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# ---------------------------------------------------
# 2. Load and Clean Dataset
# ---------------------------------------------------
@st.cache_data
def load_and_clean_data():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")
    
    # Merge datasets
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
    df = df[['budget','genres','original_title','overview','popularity','production_companies',
             'release_date','revenue','runtime','vote_average','vote_count','cast','crew']]
    
    # Clean data
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    df.dropna(subset=['runtime', 'release_date'], inplace=True)
    
    # Date conversion
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    
    # Extract genres
    df['genres'] = df['genres'].apply(ast.literal_eval).apply(
        lambda x: [d['name'] for d in x] if isinstance(x, list) else []
    )
    
    # Extract directors
    df['crew'] = df['crew'].apply(ast.literal_eval)
    df['director'] = df['crew'].apply(
        lambda x: next((d['name'] for d in x if d['job'] == 'Director'), None)
    )
    
    # Financial metrics
    df['profit'] = df['revenue'] - df['budget']
    df['ROI'] = (df['profit'] / df['budget']).replace([np.inf, -np.inf], 0)
    
    return df

try:
    df = load_and_clean_data()
    st.sidebar.success(f"✅ Data loaded successfully! {len(df)} movies")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please make sure you have the required CSV files: 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv'")
    st.stop()

# ---------------------------------------------------
# 3. Sidebar Filters
# ---------------------------------------------------
st.sidebar.header("🔍 Filter Options")

# Genre filter (multi-select from all available genres)
all_genres_list = [genre for sublist in df['genres'] for genre in sublist]
unique_genres = sorted(set(all_genres_list))

selected_genres = st.sidebar.multiselect(
    "Select Genre(s)", 
    unique_genres, 
    default=unique_genres[:3] if len(unique_genres) > 3 else unique_genres
)

# Filter dataframe based on selected genres
if selected_genres:
    df_filtered = df[df['genres'].apply(lambda x: any(genre in x for genre in selected_genres))]
else:
    df_filtered = df.copy()

# Year filter
if 'release_year' in df_filtered.columns:
    min_year = int(df_filtered['release_year'].min())
    max_year = int(df_filtered['release_year'].max())
    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_year, max_year, (min_year, max_year)
    )
    df_filtered = df_filtered[
        (df_filtered['release_year'] >= selected_years[0]) & 
        (df_filtered['release_year'] <= selected_years[1])
    ]

# Budget filter
budget_min = float(df_filtered['budget'].min() / 1e6)
budget_max = float(df_filtered['budget'].max() / 1e6)
budget_range = st.sidebar.slider(
    "Budget Range (Millions $)",
    budget_min,
    budget_max,
    (budget_min, budget_max)
)
df_filtered = df_filtered[
    (df_filtered['budget'] >= budget_range[0] * 1e6) & 
    (df_filtered['budget'] <= budget_range[1] * 1e6)
]

# Show filter summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Filters:**")
st.sidebar.markdown(f"- Genres: {len(selected_genres)} selected")
st.sidebar.markdown(f"- Years: {selected_years[0]} - {selected_years[1]}")
st.sidebar.markdown(f"- Movies: {len(df_filtered):,}")

# ---------------------------------------------------
# 4. KPIs
# ---------------------------------------------------
st.subheader("📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_movies = df_filtered.shape[0]
total_revenue = df_filtered['revenue'].sum()
avg_roi = df_filtered['ROI'].mean()
avg_rating = df_filtered['vote_average'].mean()

col1.metric("Total Movies", f"{total_movies:,}")
col2.metric("Total Revenue", f"${total_revenue/1e9:.2f}B")
col3.metric("Average ROI", f"{avg_roi:.2f}")
col4.metric("Average Rating", f"{avg_rating:.2f}/10")

# Additional KPIs
col5, col6, col7, col8 = st.columns(4)
total_budget = df_filtered['budget'].sum()
total_profit = df_filtered['profit'].sum()
success_rate = (df_filtered['profit'] > 0).mean() * 100
avg_runtime = df_filtered['runtime'].mean()

col5.metric("Total Budget", f"${total_budget/1e9:.2f}B")
col6.metric("Total Profit", f"${total_profit/1e9:.2f}B")
col7.metric("Success Rate", f"{success_rate:.1f}%")
col8.metric("Avg Runtime", f"{avg_runtime:.1f} min")

# ---------------------------------------------------
# 5. Main Dashboard - Tabs
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Financial Analysis", "🎭 Genre Analysis", "📅 Time Trends", "🏆 Top Performers"])

with tab1:
    st.subheader("💰 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Budget vs Revenue Scatter Plot
        st.markdown("**Budget vs Revenue**")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_filtered['budget']/1e6, df_filtered['revenue']/1e6, 
                           alpha=0.6, c=df_filtered['vote_average'], cmap='viridis')
        ax.set_xlabel("Budget (Millions $)")
        ax.set_ylabel("Revenue (Millions $)")
        ax.set_title("Budget vs Revenue (Color: Rating)")
        plt.colorbar(scatter, ax=ax, label='Rating')
        st.pyplot(fig)
        
        # Correlation info
        budget_revenue_corr = df_filtered['budget'].corr(df_filtered['revenue'])
        st.info(f"**Correlation between Budget and Revenue:** {budget_revenue_corr:.3f}")
    
    with col2:
        # ROI Distribution
        st.markdown("**Return on Investment Distribution**")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Filter extreme outliers for better visualization
        roi_filtered = df_filtered[df_filtered['ROI'].between(df_filtered['ROI'].quantile(0.05), 
                                                             df_filtered['ROI'].quantile(0.95))]
        ax.hist(roi_filtered['ROI'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_xlabel("Return on Investment (ROI)")
        ax.set_ylabel("Number of Movies")
        ax.set_title("Distribution of ROI")
        st.pyplot(fig)
        
        # Profit distribution
        st.markdown("**Profit Distribution**")
        fig, ax = plt.subplots(figsize=(10, 6))
        profit_filtered = df_filtered[df_filtered['profit'].between(df_filtered['profit'].quantile(0.05), 
                                                                  df_filtered['profit'].quantile(0.95))]
        ax.hist(profit_filtered['profit']/1e6, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel("Profit (Millions $)")
        ax.set_ylabel("Number of Movies")
        ax.set_title("Distribution of Profits")
        st.pyplot(fig)

with tab2:
    st.subheader("🎭 Genre Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most Common Genres
        st.markdown("**Most Common Genres**")
        genre_counts = Counter([genre for sublist in df_filtered['genres'] for genre in sublist])
        genre_counts_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values('Count', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=genre_counts_df.head(10), x='Count', y='Genre', ax=ax)
        ax.set_title("Top 10 Most Common Genres")
        st.pyplot(fig)
    
    with col2:
        # Average Revenue by Genre
        st.markdown("**Average Revenue by Genre**")
        genre_revenue = {}
        for genre in unique_genres:
            genre_movies = df_filtered[df_filtered['genres'].apply(lambda x: genre in x)]
            if len(genre_movies) > 0:
                genre_revenue[genre] = genre_movies['revenue'].mean()
        
        genre_revenue_df = pd.DataFrame(genre_revenue.items(), columns=['Genre', 'Avg_Revenue']).sort_values('Avg_Revenue', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=genre_revenue_df.head(10), x='Avg_Revenue', y='Genre', ax=ax)
        ax.set_title("Top 10 Genres by Average Revenue")
        ax.set_xlabel("Average Revenue ($)")
        st.pyplot(fig)
    
    # Genre ROI Analysis
    st.markdown("**Genre ROI Analysis**")
    genre_roi = {}
    for genre in unique_genres:
        genre_movies = df_filtered[df_filtered['genres'].apply(lambda x: genre in x)]
        if len(genre_movies) > 3:  # Minimum 3 movies for meaningful average
            genre_roi[genre] = genre_movies['ROI'].mean()
    
    if genre_roi:
        genre_roi_df = pd.DataFrame(genre_roi.items(), columns=['Genre', 'Avg_ROI']).sort_values('Avg_ROI', ascending=False)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=genre_roi_df.head(10), x='Avg_ROI', y='Genre', ax=ax)
            ax.set_title("Top 10 Genres by Average ROI")
            ax.set_xlabel("Average ROI")
            st.pyplot(fig)
        
        with col4:
            # Show top and bottom genres
            st.markdown("**Best & Worst Performing Genres**")
            top_5_genres = genre_roi_df.head(5)
            bottom_5_genres = genre_roi_df.tail(5)
            
            st.write("**Highest ROI Genres:**")
            for _, row in top_5_genres.iterrows():
                st.write(f"- {row['Genre']}: {row['Avg_ROI']:.2f}")
            
            st.write("**Lowest ROI Genres:**")
            for _, row in bottom_5_genres.iterrows():
                st.write(f"- {row['Genre']}: {row['Avg_ROI']:.2f}")

with tab3:
    st.subheader("📅 Time Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue Trends Over Time
        st.markdown("**Financial Trends Over Time**")
        yearly_stats = df_filtered.groupby('release_year').agg({
            'revenue': 'mean',
            'budget': 'mean',
            'profit': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(yearly_stats['release_year'], yearly_stats['revenue']/1e6, 
                marker='o', label='Average Revenue', linewidth=2)
        ax.plot(yearly_stats['release_year'], yearly_stats['budget']/1e6, 
                marker='s', label='Average Budget', linewidth=2)
        ax.plot(yearly_stats['release_year'], yearly_stats['profit']/1e6, 
                marker='^', label='Average Profit', linewidth=2)
        ax.set_xlabel("Year")
        ax.set_ylabel("Amount (Millions $)")
        ax.set_title("Financial Trends Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Runtime and Rating Trends
        st.markdown("**Runtime & Rating Trends Over Time**")
        trends = df_filtered.groupby('release_year').agg({
            'runtime': 'mean',
            'vote_average': 'mean'
        }).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Average Runtime (min)', color=color)
        ax1.plot(trends['release_year'], trends['runtime'], color=color, marker='o', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Average Rating', color=color)
        ax2.plot(trends['release_year'], trends['vote_average'], color=color, marker='s', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Runtime and Rating Trends Over Time')
        st.pyplot(fig)
    
    # Movies per year
    st.markdown("**Movies Released Per Year**")
    movies_per_year = df_filtered.groupby('release_year').size().reset_index(name='count')
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(movies_per_year['release_year'], movies_per_year['count'], color='skyblue', alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Movies Released Per Year")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab4:
    st.subheader("🏆 Top Performers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Grossing Movies
        st.markdown("**Top Grossing Movies**")
        top_n = st.slider("Number of top movies to show", 5, 20, 10, key="top_movies")
        top_movies = df_filtered.nlargest(top_n, 'revenue')[['original_title', 'release_year', 'revenue', 'budget', 'profit', 'vote_average']]
        top_movies = top_movies.rename(columns={'original_title': 'Title', 'release_year': 'Year', 
                                              'revenue': 'Revenue', 'budget': 'Budget', 
                                              'profit': 'Profit', 'vote_average': 'Rating'})
        
        # Format currency columns
        display_df = top_movies.copy()
        display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f"${x/1e6:.1f}M")
        display_df['Budget'] = display_df['Budget'].apply(lambda x: f"${x/1e6:.1f}M")
        display_df['Profit'] = display_df['Profit'].apply(lambda x: f"${x/1e6:.1f}M")
        display_df['Rating'] = display_df['Rating'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(display_df, use_container_width=True)
    
    with col2:
        # Top Directors by Average Revenue
        st.markdown("**Top Directors by Average Revenue**")
        director_stats = df_filtered.groupby('director').agg({
            'revenue': 'mean',
            'vote_average': 'mean',
            'original_title': 'count'
        }).rename(columns={'original_title': 'movie_count'})
        
        top_directors = director_stats[director_stats['movie_count'] >= 2].nlargest(10, 'revenue')
        
        if not top_directors.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top_directors.reset_index(), x='revenue', y='director', ax=ax)
            ax.set_xlabel("Average Revenue ($)")
            ax.set_title("Top Directors by Average Revenue (Min. 2 movies)")
            st.pyplot(fig)
        else:
            st.info("No directors meet the minimum movie count criteria with current filters.")
    
    # Best ROI Movies
    st.markdown("**Best Return on Investment (ROI)**")
    roi_movies = df_filtered[df_filtered['budget'] > 1e6]  # Filter very low budget movies
    top_roi = roi_movies.nlargest(10, 'ROI')[['original_title', 'release_year', 'budget', 'revenue', 'ROI']]
    top_roi = top_roi.rename(columns={'original_title': 'Title', 'release_year': 'Year', 
                                    'budget': 'Budget', 'revenue': 'Revenue', 'ROI': 'ROI'})
    
    display_roi = top_roi.copy()
    display_roi['Budget'] = display_roi['Budget'].apply(lambda x: f"${x/1e6:.1f}M")
    display_roi['Revenue'] = display_roi['Revenue'].apply(lambda x: f"${x/1e6:.1f}M")
    display_roi['ROI'] = display_roi['ROI'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_roi, use_container_width=True)

# ---------------------------------------------------
# 6. Correlation Heatmap
# ---------------------------------------------------
st.subheader("🔗 Correlation Heatmap")

# Calculate correlation matrix
numerical_cols = ['budget', 'revenue', 'profit', 'runtime', 'vote_average', 'vote_count', 'ROI']
correlation_data = df_filtered[numerical_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', ax=ax)
ax.set_title("Correlation Heatmap of Movie Metrics")
st.pyplot(fig)

# ---------------------------------------------------
# 7. Key Insights
# ---------------------------------------------------
st.subheader("💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top Insights:**")
    
    # Most profitable genre
    genre_profit = {}
    for genre in unique_genres:
        genre_movies = df_filtered[df_filtered['genres'].apply(lambda x: genre in x)]
        if len(genre_movies) > 5:  # Minimum 5 movies for meaningful average
            genre_profit[genre] = genre_movies['profit'].mean()
    
    if genre_profit:
        most_profitable_genre = max(genre_profit, key=genre_profit.get)
        st.write(f"• **Most Profitable Genre:** {most_profitable_genre}")
    
    # Budget-Revenue relationship
    budget_revenue_corr = df_filtered['budget'].corr(df_filtered['revenue'])
    st.write(f"• **Budget-Revenue Correlation:** {budget_revenue_corr:.3f}")
    
    # Rating insight
    high_rated = df_filtered[df_filtered['vote_average'] >= 7.5]
    low_rated = df_filtered[df_filtered['vote_average'] < 5.0]
    
    if not high_rated.empty and not low_rated.empty:
        high_rated_revenue = high_rated['revenue'].mean()
        low_rated_revenue = low_rated['revenue'].mean()
        st.write(f"• **High-rated movies (≥7.5)** earn ${high_rated_revenue/1e6:.1f}M on average")
        st.write(f"• **Low-rated movies (<5.0)** earn ${low_rated_revenue/1e6:.1f}M on average")
    
    # ROI insights
    high_roi_movies = df_filtered[df_filtered['ROI'] > 5]  # ROI > 500%
    if not high_roi_movies.empty:
        st.write(f"• **{len(high_roi_movies)} movies** achieved ROI > 500%")

with col2:
    st.markdown("**Dataset Summary:**")
    st.write(f"• **Time Period:** {int(df_filtered['release_year'].min())} - {int(df_filtered['release_year'].max())}")
    st.write(f"• **Total Budget Analyzed:** ${df_filtered['budget'].sum()/1e9:.2f}B")
    st.write(f"• **Average Runtime:** {df_filtered['runtime'].mean():.1f} minutes")
    st.write(f"• **Success Rate (Profit > 0):** {(df_filtered['profit'] > 0).mean()*100:.1f}%")
    st.write(f"• **Highest Grossing Movie:** {df_filtered.loc[df_filtered['revenue'].idxmax(), 'original_title']} (${df_filtered['revenue'].max()/1e6:.1f}M)")
    st.write(f"• **Highest Budget Movie:** {df_filtered.loc[df_filtered['budget'].idxmax(), 'original_title']} (${df_filtered['budget'].max()/1e6:.1f}M)")

# ---------------------------------------------------
# 8. Footer
# ---------------------------------------------------
st.markdown("---")
st.markdown("👨‍🔬 *Built by Benjamin Akingbade — Data Science & Movie Analytics Project*")
st.markdown("*Data Source: TMDB 5000 Movie Dataset*")