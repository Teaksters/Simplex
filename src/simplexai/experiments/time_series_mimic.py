import os

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

pd.set_option("display.max_columns", None)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = '../Data/preprocessed'


class TimeSeriesProstateCancerDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        self.X = X
        self.y = y.astype(int)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> tuple:
        data = torch.tensor(self.X[i], dtype=torch.float32)
        target = torch.tensor(self.y[i], dtype=torch.float32)
        return data, target

def load_from_preprocessed(dir):
    # Reads and concatenates the train and test data into one dataframe
    data = [os.path.join(dir, sub_dir, 'listfile.csv')
                  for sub_dir in os.listdir(dir)]
    dfs = [pd.read_csv(x) for x in data]
    df = pd.concat(dfs)
    return df

def load_age():
    full_df = pd.read_csv(os.path.join(DATA_DIR, 'all_stays.csv'))
    full_df.sort_values(['SUBJECT_ID', 'ICUSTAY_ID'])
    # Adhere to data format of other dataframes
    general_df = full_df[full_df.duplicated('SUBJECT_ID') == False]
    general_df['SUBJECT_ID'] = general_df['SUBJECT_ID'].astype(str) + "_episode1_timeseries.csv"

    duplicates_df = full_df[full_df.duplicated('SUBJECT_ID') == True]
    i = 2
    while not duplicates_df.empty:
        # update general df
        episode_str = "_episode" + str(i) + "_timeseries.csv"
        temp_df = duplicates_df[duplicates_df.duplicated('SUBJECT_ID') == False]
        temp_df['SUBJECT_ID'] = temp_df['SUBJECT_ID'].astype(str) + episode_str
        general_df = pd.concat([general_df, temp_df])
        # prepare for next round
        duplicates_df = duplicates_df[duplicates_df.duplicated('SUBJECT_ID') == True]
        i += 1

    general_df.rename(columns={'SUBJECT_ID': 'stay'}, inplace=True)
    return general_df


def load_tabular_mimic(random_seed: int = 42) -> tuple:
    # Load MIMIC-III data into panda dataframes
    label_dir = os.path.join(DATA_DIR, 'in-hospital-mortality')
    label_df = load_from_preprocessed(label_dir)
    feature_dir = os.path.join(DATA_DIR, 'phenotyping')
    feature_df = load_from_preprocessed(feature_dir)
    age_df = load_age()

    # Merge data into workable complete format
    data_df = pd.merge(label_df, feature_df, on='stay')
    data_df = pd.merge(age_df, data_df, on='stay')

    ##################### OPTIONAL ######################################
    ### Balance data set for even amount of survivors and mortalities ###
    #####################################################################
    # mask = data_df[label] is True
    # df_dead = data_df[mask]
    # df_survive = data_df[~mask]
    # data_df = pd.concat(
    #     [
    #         df_dead.sample(2500, random_state=random_seed),
    #         df_survive.sample(2500, random_state=random_seed),
    #     ]
    # )
    ############################################################################

    # df = sklearn.utils.shuffle(data_df, random_state=random_seed)
    # df = df.reset_index(drop=True)
    return data_df

def generate_paths(dir):
    file_paths = os.path.abspath(
                     os.path.join(DATA_DIR, dir)
    )
    file_paths = [os.path.join(file_paths, sub_dir)
                      for sub_dir in os.listdir(file_paths)]
    file_paths = [os.path.join(path, file)
                      for path in file_paths
                          for file in os.listdir(path)
                              if file[-4:] == '.csv']
    return file_paths

########################### WORKING ######################################
def load_time_series_mimic(random_seed: int = 42) -> tuple:
    # Specify desired features and label
    temporal_features = [
        'Hours',
        'Capillary refill rate',
        'Diastolic blood pressure',
        'Fraction inspired oxygen',
        'Glascow coma scale eye opening',
        'Glascow coma scale motor response',
        'Glascow coma scale total',
        'Glascow coma scale verbal response',
        'Glucose',
        'Heart Rate',
        'Height',
        'Mean blood pressure',
        'Oxygen saturation',
        'Respiratory rate',
        'Systolic blood pressure',
        'Temperature',
        'Weight',
        'pH'
    ]
    constant_features = [
        'AGE'
    ]
    categorical_features = [
        'GENDER',
        'ETHNICITY',
        'Acute and unspecified renal failure',
        'Acute cerebrovascular disease',
        'Acute myocardial infarction',
        'Cardiac dysrhythmias',
        'Chronic kidney disease',
        'Chronic obstructive pulmonary disease and bronchiectasis',
        'Complications of surgical procedures or medical care',
        'Conduction disorders',
        'Congestive heart failure',
        'nonhypertensive',
        'Coronary atherosclerosis and other heart disease',
        'Diabetes mellitus with complications',
        'Diabetes mellitus without complication',
        'Disorders of lipid metabolism',
        'Essential hypertension	Fluid and electrolyte disorders',
        'Gastrointestinal hemorrhage',
        'Hypertension with complications and secondary hypertension',
        'Other liver diseases',
        'Other lower respiratory disease',
        'Other upper respiratory disease',
        'Pleurisy',
        'pneumothorax',
        'pulmonary collapse',
        'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
        'Respiratory failure',
        'insufficiency',
        'arrest (adult)',
        'Septicemia (except in labor)',
        'Shock',
    ]
    label = 'y_true'


    ######################## DONE TILL HERE ###########################
    # Load tabular data
    df = load_tabular_mimic()
    df = pd.concat([df, pd.DataFrame(columns=temporal_features, dtype='object')])

    # Define all paths to time serie data
    paths = generate_paths('in-hospital-mortality')
    for path in paths:
        stay = path.split('/')[-1]
        temp_df = pd.read_csv(path)
        for col in temporal_features:
            df.at[df.index[df['stay'] == stay][0], col] = temp_df[col].tolist()
            print(df.loc[df.index[df['stay'] == stay][0], col])
    print(df)
    exit()
    ####################### WORKING TILL HERE #########################
    max_time_points = temporal_df["New ID"].value_counts().max()
    const_df = pd.read_csv(
        os.path.abspath(
            os.path.join(
                ROOT_DIR,
                "./data/Time series Prostate Cancer/baseline.csv",
            )
        )
    )
    y = const_df[label].to_numpy()

    df = temporal_df.merge(const_df, on="New ID", how="left")
    # categorical_df = df[categorical_features]

    # Get dummies
    all_features = constant_features + temporal_features + ["New ID"]
    df = pd.get_dummies(df[all_features], columns=categorical_features)
    all_features = [col for col in df.columns if col != "New ID"]

    # Scaling
    rescale_dict = df[all_features].max().to_dict()
    scaler = MinMaxScaler()
    df[all_features] = scaler.fit_transform(df[all_features])

    # Balance the dataset (not in use - use weights instead)
    # grouped_df = df.groupby(by="New ID")
    # df = pd.concat([df for idx, df in grouped_df][:200])

    # Limit columns to features
    df = df[all_features + ["New ID"]]

    group_df = df.groupby(by="New ID").cumcount()
    mux = pd.MultiIndex.from_product([df["New ID"].unique(), group_df.unique()])
    X = (
        df.set_index(["New ID", group_df])
        .reindex(mux, fill_value=0)
        .groupby(level=0)
        .apply(lambda x: x.values)
        .to_numpy()
    )

    # mask = const_df[label] == 1
    # const_df = sklearn.utils.shuffle(const_df, random_state=random_seed)
    # const_df = const_df.reset_index(drop=True)

    return X, y, all_features, max_time_points, rescale_dict


if __name__=='__main__':
    load_time_series_mimic()
