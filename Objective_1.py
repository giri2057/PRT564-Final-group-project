import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
file_path = '/content/retractions35215.csv'
df = pd.read_csv(file_path)

# Convert dates to datetime
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], errors='coerce')
df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], errors='coerce')

# Extract years from dates
df['RetractionYear'] = df['RetractionDate'].dt.year
df['OriginalPaperYear'] = df['OriginalPaperDate'].dt.year

# Split subjects and reasons into lists
df['Subject'] = df['Subject'].apply(lambda x: x.split(';') if pd.notna(x) else [])
df['Reason'] = df['Reason'].apply(lambda x: x.split(';') if pd.notna(x) else [])

# Expand the lists into separate rows
df_subjects = df.explode('Subject')
df_reasons = df.explode('Reason')

# Visualization by Subject (Top 10)
top_subjects = df_subjects['Subject'].value_counts().head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_subjects.values, y=top_subjects.index)
plt.title('Top 10 Retractions by Subject')
plt.xlabel('Count')
plt.ylabel('Subject')
plt.show()

# Visualization by Year
plt.figure(figsize=(12, 8))
sns.countplot(x='RetractionYear', data=df)
plt.title('Retractions by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualization by Reason (Top 10)
top_reasons = df_reasons['Reason'].value_counts().head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_reasons.values, y=top_reasons.index)
plt.title('Top 10 Retractions by Reason')
plt.xlabel('Count')
plt.ylabel('Reason')
plt.show()

# Visualization by Country (Top 10)
top_countries = df['Country'].value_counts().head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 10 Retractions by Country')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()

# Visualization by Year with Logarithmic Scale
plt.figure(figsize=(12, 8))
sns.countplot(x='RetractionYear', data=df)
plt.yscale('log')
plt.title('Retractions by Year (Logarithmic Scale)')
plt.xlabel('Year')
plt.ylabel('Count (Log Scale)')
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap for Numerical Columns
# Select numerical columns
numerical_cols = ['RetractionYear', 'OriginalPaperYear', 'RetractionPubMedID', 'OriginalPaperPubMedID', 'CitationCount']

# Calculate correlation matrix
corr_matrix = df[numerical_cols].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
