# Loan Default Data Analysis

# Step 1: Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt;

# Step 2: Load the Dataset
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\Advance Python\Loan_Default.csv",low_memory = False)
# print(df.head())

# step 3 : Basic overview
# print(df.shape)
# print(df.info())

# # 4. Data Cleaning and Preparation
df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")

df = df.drop_duplicates()
# question 1 : Missing Values: Dataset ke har column mein kitni missing values hain? Unhe visualize karne ke liye ek bar chart banaiye.

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values>0]
# print(missing_values)

# plot bar garph
plt.figure()
missing_values.plot(kind = 'bar') # aa syntax no use kari ne darek column no bar garph banavi sakay
plt.title("Missing Values per Column")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")

plt.xticks(rotation=45)
plt.tight_layout()

# plt.show()


# handling null and missing values
'''Step 2: Column classify karo (MOST IMPORTANT)
  
Missing values ne 3 group ma divide karo:

🟢 A) Low missing (< 1–2%)

👉 Safe simple fill

Columns:

loan_limit
approv_in_adv
loan_purpose
neg_ammortization
term
age
submission_of_application
✔️ Handling:
categorical → mode
numeric (term) → median
🟡 B) Medium missing (5–15%)

👉 Careful fill required

Columns:

income
property_value
ltv
dtir1
✔️ Handling:
numeric → median (BEST)
OR grouped median (advanced)
🔴 C) High missing (20–35%)

👉 Special treatment

Columns:

rate_of_interest
interest_rate_spread
upfront_charges
✔️ Handling options:

✔️ Option 1 (BEST for ML):

median + add missing flag column

✔️ Option 2:

drop if not important (depends model)'''

# 1. Categorical columns fill(mode)
df['loan_limit'] = df['loan_limit'].fillna(df['loan_limit'].mode()[0])
df['approv_in_adv'] = df['approv_in_adv'].fillna(df['approv_in_adv'].mode()[0])
df['loan_purpose'] = df['loan_purpose'].fillna(df['loan_purpose'].mode()[0])
df['submission_of_application'] = df['submission_of_application'].fillna(df['submission_of_application'].mode()[0])
df['neg_ammortization'] = df['neg_ammortization'].fillna(df['neg_ammortization'].mode()[0])
df['age'] = df['age'].fillna(df['age'].mode()[0])

# numerical value(median)

df['term'] = df['term'].fillna(df['term'].median())
df['income'] = df['income'].fillna(df['income'].median())
df['property_value'] = df['property_value'].fillna(df['property_value'].median())
df['ltv'] = df['ltv'].fillna(df['ltv'].median())
df['dtir1'] = df['dtir1'].fillna(df['dtir1'].median())

# high missing value
df['rate_of_interest_missing'] = df['rate_of_interest'].isnull().astype(int) # aa ek new column banave che jema(0/1) store thay che 0 means value and 1 means null value
df['rate_of_interest'] = df['rate_of_interest'].fillna(df['rate_of_interest'].median())
df['interest_rate_spread_missing'] = df['interest_rate_spread'].isnull().astype(int) 
df['interest_rate_spread'] = df['interest_rate_spread'].fillna(df['interest_rate_spread'].median())
df['upfront_charges_missing'] = df['upfront_charges'].isnull().astype(int)
df['upfront_charges'] = df['upfront_charges'].fillna(df['upfront_charges'].median())

# print(df.info())

# 5. Answering Business Questions with Analysis

'''# question 1: Loan Status Distribution: Kitne logo ne loan default kiya hai (Status column) aur kitne logo ne nahi? Iska count aur percentage nikiye.'''
counts = df['status'].value_counts()
percentage = df['status'].value_counts(normalize = True)*100

default = counts.get(1,0)
non_default = counts.get(0,0)

default_per = percentage.get(1,0)
non_default_per = percentage.get(0,0)

print(f" Loan Default : {default} ({default_per:.2f}%)")
print(f"No Default : {non_default} ({non_default_per:.2f}%)")
'''
# question 2:Income vs Loan Amount: Kya income aur loan_amount ke beech koi correlation hai? Correlation matrix banaiye aur heatmap se show kijiye.'''

corr = df[['income','loan_amount']].corr()
print(corr)

plt.figure()
corr = df[['income','loan_amount']].corr()
sns.heatmap(corr,annot = True)
plt.title("correlation matrix of income and loan amount")
# plt.show()

#   correlation ni range -1 ≤ r ≤ +1
print("As shown in heatmap,correlation between income and loan_amount is 0.44 which indicates a moderate positive relationship")

'''# question 3:Alag-alag loan_type ke liye average rate_of_interest kya hai?'''
diff_type = df.groupby('loan_type')['rate_of_interest'].mean()
print(diff_type)

'''# question 4:Default karne walo (Status=1) aur default na karne walo (Status=0) ka average Credit_Score compare kijiye.'''
default = df.groupby('status')['credit_score'].mean()
print(default)

# comparision ke liye hum bar chart create karenge
plt.figure()
default.plot(kind = 'bar')
plt.title("Average Credit Score by Loan Status")
plt.xlabel("Status (0 = No Default, 1 = Default)")
plt.ylabel("Average Credit Score")

# plt.show()

'''# question 5:Kis gender category mein sabse zyada loan default cases dekhne ko milte hain? Ek countplot banaiye.'''
default1 = df[df['status'] == 1].groupby('gender')['status'].count()
gender1 = default1.idxmax()
max_default =default1.max()
print(f"gender category mai sabse jayda loan default cases --> {gender1} : {max_default}")

# countplot
plt.figure()
sns.countplot(x = 'gender',hue = 'status',data = df)
plt.title("Loan default by gender")
plt.xlabel("gender")
plt.ylabel("count")
# plt.show()

'''question 6:Region ke hisaab se business_or_commercial loans ki distribution kya hai? Iske liye ek grouped bar chart banaiye.'''
distri = df.groupby(['region','business_or_commercial']).size().unstack()
'''👉 size() = “ketla individual cases(b/c or non b/c) chhe ae count kare?”
👉 unstack() = “table banaavi ne side-by-side dekhaad”'''
print(distri)

# grouped bar chart(countplot = bar chart with automatic counting)
plt.close('all') # uppar na bhada chart ne close karva use thay che
plt.figure()
sns.countplot(x= 'region',hue = 'business_or_commercial',data = df)
plt.title("bar chart of bussiness or commerical loans by region")
plt.xlabel("region")
# plt.show()

'''question 7:Kaunse age group (age column) ke log sabse zyada loan ke liye apply karte hain?'''
age_count = df['age'].value_counts()
# print(age_count)
max_age = age_count.idxmax()
count = age_count.max()
print(F"age group jisne sabse jyada loan ke liye apply kiya hai --> {max_age} : {count}")

'''question 8:loan_amount aur income columns ke liye boxplot banaiye taaki outliers ka pata chal sake.'''
plt.close('all')
plt.figure()

# loan_amount
plt.subplot(1,2,1)
sns.boxplot(x = df['loan_amount'])
plt.title("box plot of loan amount")

# income
plt.subplot(1,2,2)
sns.boxplot(x = df['income'])
plt.title("box plot of income")

plt.suptitle("boxplot to detect outliers")
# plt.show()
print("Both loan_amount and income show a right-skewed distribution with several outliers. Loan amount has extreme high values indicating few large loans, while income also contains high-income outliers. This suggests that the dataset is not normally distributed and contains significant variability.")

'''question 10 : property_value aur LTV (Loan-to-Value) ka relation scatter plot ke zariye dikhaiye'''
plt.close('all')
plt.figure()
sns.scatterplot(x='property_value', y='ltv', data=df)

plt.title("Relationship between Property Value and LTV")
plt.xlabel("Property Value")
plt.ylabel("LTV (Loan-to-Value)")

plt.show()

'''Data Quality Issue: Dataset ma 2000 thi vadhare LTV (Loan-to-Value) na outliers che, je data entry ni bhul (error) darshave che.

Inverse Relation: Jem Property Value vadhe che, tem LTV niche jay che, jeno arth che ke vadhare kimat ni property par loan nu risk praman ma ochhu hoy che.

Risk Pattern: Credit Score vadhare hoy to pan default risk (~25%) samanya loko jetlo j rahe che, etle ke khali score parthi default predict na kari shakay.'''

# overall insights
'''1)Loan Status Distribution: 36,639 borrowers (24.64%) defaulted, while 112,031 (75.36%) did not default.
2)Income vs Loan Amount Correlation: There is a moderate positive correlation (0.44) between income and loan amount.
3)Loan Type vs Interest Rate: Type1 loans have the highest average interest rate (4.08), followed by Type2 (3.97) and Type3 (3.70).
4)Credit Score vs Default Status: Non-defaulters (0) and defaulters (1) have very similar average credit scores (~699–700), showing weak separation.
5)Gender with Most Defaults: Male category has the highest number of loan defaults (11,091 cases).
6)Region vs Business/Commercial Loans: North and South regions have the highest loan counts, with non-business loans dominating in all regions.
7)Age Group with Most Applications: The 45–54 age group applied for the most loans (34,920 applications).
8)Outliers in Loan Amount & Income: Both variables show strong right-skewness with significant outliers, especially high loan amounts and incomes.
9)Property Value vs LTV Relationship: Higher property values generally correspond to lower LTV, indicating reduced risk for expensive properties.
10)Overall Insight: Loan default is influenced more by multiple factors together rather than a single variable like credit score, income, or property value alone.'''