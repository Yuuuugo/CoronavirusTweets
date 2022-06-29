import pandas as pd
import cleaning


df = pd.read_csv("data/Corona_NLP_train.csv", encoding="latin-1")
df.rename(columns={"OriginalTweet": "text"}, inplace=True)
df = cleaning.cleaning_df(df)
print(df["clean_text"].head())
print(df.columns)
