import pandas as pd
def load_dataset(datapath):
    # next load the data
    df = pd.read_csv(datapath)
    print(df.head())
    print(df.shape)
    print("Check data Type:")
    print(df.dtypes)
    return df

