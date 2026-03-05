# Exploratory Data Analysis (EDA) for Fraud Detection

## 1. Start With Dataset Overview

First understand **what the data looks like**.

### Key questions

- How many rows and columns?
- What are the data types?
- Are there missing values?
- What features exist?

### What to write in report

- Dataset contains **X transactions** and **Y features**
- Feature types include **numeric, categorical**
- Initial statistics show **transaction amount ranges from ...**

---

## 2. Check Class Distribution (Very Important)

Fraud datasets are **highly imbalanced**.

### Code

```python
df['is_fraud'].value_counts()

df['is_fraud'].value_counts(normalize=True) * 100
```

### Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='is_fraud', data=df)
plt.title("Fraud vs Legitimate Transactions")
plt.show()
```

### What to analyze

- Fraud percentage
- imbalance ratio

Example insight:

> Fraud cases represent only **0.17% of total transactions**, confirming a highly imbalanced classification problem.

---

# 3. Missing Values

Check whether any columns have missing values.

### Code

```python
df.isnull().sum()
```

If missing exists:

```python
(df.isnull().sum() / len(df)) * 100
```

### What to analyze

- columns with missing values
- strategy (drop, fill, etc.)

Example insight:

> No missing values were found in the dataset.

---

# 4. Transaction Amount Analysis

Fraud often occurs in **different amount distributions**.

### Code

```python
sns.boxplot(x='is_fraud', y='amt', data=df)
plt.show()
```

Or

```python
sns.histplot(df['amt'], bins=50)
plt.show()
```

### Questions to answer

- Are fraud transactions larger?
- Are there outliers?

Example insight:

> Fraud transactions tend to occur at **higher transaction amounts compared to legitimate ones**.

---

# 5. Feature Distribution

Look at distribution of key features.

### Code

```python
df.hist(figsize=(12,10))
plt.show()
```

Or for specific columns:

```python
sns.histplot(df['amt'], kde=True)
```

### What to look for

- skewed distributions
- outliers
- unusual spikes

---

# 6. Correlation Analysis

This helps identify **important predictive features**.

### Code

```python
corr = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm")
plt.show()
```

Better: correlation with fraud only.

```python
corr_target = corr['is_fraud'].sort_values(ascending=False)
print(corr_target)
```

### Insight example

> Transaction amount and merchant category show moderate correlation with fraud occurrence.

---

# 7. Time Pattern Analysis (If dataset has time)

Fraud often occurs at certain times.

Example:

```python
sns.countplot(x='hour', hue='is_fraud', data=df)
```

Questions:

- Do frauds occur more at night?
- Are there temporal patterns?

---

# 8. Feature vs Fraud Comparison

Compare each important feature against fraud.

Example:

```python
sns.boxplot(x='is_fraud', y='amt', data=df)
```

or

```python
sns.violinplot(x='is_fraud', y='amt', data=df)
```

---

# 9. Outlier Detection

Outliers may represent fraud.

```python
Q1 = df['amt'].quantile(0.25)
Q3 = df['amt'].quantile(0.75)

IQR = Q3 - Q1
```

Look for extreme values.

---

# 10. Key Insights Section (Most Important)

Your EDA is not the plots — it's the **conclusions**.

Example:

### Key Findings

1. The dataset is highly imbalanced with only **X% fraud transactions**.
2. Fraudulent transactions tend to occur at **higher transaction amounts**.
3. Certain merchant categories appear more frequently in fraud cases.
4. Feature distributions show significant skewness requiring scaling.
5. No missing values were detected in the dataset.

---

# Recommended Notebook Structure

Create:

```
notebooks/01_eda.ipynb
```

Sections:

```
1. Dataset Overview
2. Class Distribution
3. Missing Values
4. Feature Distribution
5. Fraud vs Legitimate Comparison
6. Correlation Analysis
7. Key Insights
```

Example:

Bad EDA:

> Here is a histogram of transaction amount.

Good EDA:

> Fraud transactions appear concentrated in higher transaction amounts, suggesting transaction size may be an important predictive feature.
