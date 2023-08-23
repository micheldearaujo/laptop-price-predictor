import sys

sys.path.insert(0,'.')

from src.config import *

prices_df = pd.read_csv("./data/raw/laptop_price.csv", sep=',', encoding='latin1')

def convert_to_gb(string):
    """Converts Hard Drive string memory to float."""
    number = float(string[:-2])

    if "TB" in string:
        number = number*1024
    
    return number

def build_features(raw_dataframe: pd.DataFrame):

    """
    Performs Feature Engineering to build up the features

    Ags:
        raw_dataframe (pd.DataFrame): The raw DataFrame holding the original features
    """
    
    companies_to_agg = ["Razer", "Mediacom", "Microsoft", "Xiaomi", "Vero", "Chuwi", "Google", 'Fujitsu', 'LG','Huawei']
    raw_dataframe["Company"] = raw_dataframe["Company"].apply(lambda x: "Other" if x in companies_to_agg else x)

    # Screen Resolution - A resolução é sempre no final da string, então é fácil
    raw_dataframe["ScreenResolution"] = raw_dataframe["ScreenResolution"].str.split(" ").apply(lambda x: x[-1])
    raw_dataframe["ScreenWidth"] =  raw_dataframe["ScreenResolution"].str.split("x").apply(lambda x: x[0]).astype("int")
    raw_dataframe["ScreenHeight"] =  raw_dataframe["ScreenResolution"].str.split("x").apply(lambda x: x[1]).astype("int")

    # CPU é mesma coisa, a frequencia do CPU é sempre a ultima e a marca é a primeira
    raw_dataframe["CPU_BRAND"] =  raw_dataframe["Cpu"].str.split(" ").apply(lambda x: x[0])
    raw_dataframe["CPU_FREQUENCY"] =  raw_dataframe["Cpu"].str.split(" ").apply(lambda x: x[-1])
    raw_dataframe["CPU_FREQUENCY"] =  raw_dataframe["CPU_FREQUENCY"].apply(lambda x: x[:-3]).astype("float")
    # RAM
    raw_dataframe["Ram"] =  raw_dataframe["Ram"].apply(lambda x: x[:-2]).astype("int")
    # Memory
    raw_dataframe["Memory_Size"] = raw_dataframe["Memory"].str.split(" ").apply(lambda x: x[0])
    raw_dataframe["Memory_Size"] = raw_dataframe["Memory_Size"].apply(lambda x: convert_to_gb(x))
    raw_dataframe["Memory_Type"] = raw_dataframe["Memory"].str.split(" ").apply(lambda x: x[1])

    # Weight
    raw_dataframe["Weight"] = raw_dataframe["Weight"].apply(lambda x: x[:-2]).astype("float")
    # GPU
    raw_dataframe["GPU_BRAND"] = raw_dataframe["Gpu"].str.split(" ").apply(lambda x: x[0])

    # Get Dummies das variaveis categóricas finais
    raw_dataframe = raw_dataframe.join(pd.get_dummies(raw_dataframe["Company"], prefix="company", dtype="int"))
    raw_dataframe = raw_dataframe.join(pd.get_dummies(raw_dataframe["TypeName"], prefix="typeName", dtype="int"))
    raw_dataframe = raw_dataframe.join(pd.get_dummies(raw_dataframe["CPU_BRAND"], prefix="CPU_BRAND", dtype="int"))
    raw_dataframe = raw_dataframe.join(pd.get_dummies(raw_dataframe["GPU_BRAND"], prefix="GPU_BRAND", dtype="int"))
    raw_dataframe = raw_dataframe.join(pd.get_dummies(raw_dataframe["OpSys"], prefix="OpSys", dtype="int"))
    raw_dataframe = raw_dataframe.join(pd.get_dummies(raw_dataframe["Memory_Type"], prefix="Memory_Type", dtype="int"))

    processed_df = raw_dataframe.drop(["laptop_ID", "Product", "Company", "TypeName", "ScreenResolution", "Cpu", "Memory", "Gpu", "OpSys", "GPU_BRAND", "Memory_Type", "CPU_BRAND"], axis=1)

    return processed_df


def remove_weak_features(clean_dataframe: pd.DataFrame):
    """
    Select the top 20 features based on Linear Correlation

    Args:
        clean_dataframe (pd.DataFrame): The cleaned dataframe, out of the build_features function.

    Returns:
        pd.DataFrame: Cleaned Dataframe only with the top performing features.
    """

    top_20_features = ['OpSys_No OS',
    'company_MSI',
    'CPU_BRAND_AMD',
    'CPU_BRAND_Intel',
    'GPU_BRAND_Intel',
    'GPU_BRAND_AMD',
    'company_Acer',
    'Weight',
    'Memory_Type_Flash',
    'typeName_Workstation',
    'typeName_Ultrabook',
    'GPU_BRAND_Nvidia',
    'typeName_Gaming',
    'Memory_Type_HDD',
    'CPU_FREQUENCY',
    'Memory_Type_SSD',
    'typeName_Notebook',
    'ScreenHeight',
    'ScreenWidth',
    'Ram',
    'Price_euros']

    model_dataframe = clean_dataframe[top_20_features]

    return model_dataframe


# Execute the function
processed_df = build_features(prices_df)

# save the clean DF
processed_df.to_csv("./data/processed/clean_prices.csv", index=False)

# Execute the function
model_df = remove_weak_features(processed_df)

# save the model DF
model_df.to_csv("./data/processed/model_df.csv", index=False)