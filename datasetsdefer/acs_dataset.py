from pathlib import Path
import pandas as pd
import numpy as np
import folktables
from .basedataset import dataset


ACS_TASK = "ACSIncome"
SEED = 42
EPS = 1e-6
data_dir = Path("~").expanduser()\
    / "data" / "folktables" / "train=0.6_test=0.2_validation=0.2_max-groups=4"


ACS_CATEGORICAL_COLS = {
    'COW',  # class of worker
    'MAR',  # marital status
    'OCCP',  # occupation code
    'POBP',  # place of birth code
    'RELP',  # relationship status
    'SEX',
    'RAC1P',  # race code
    'DIS',  # disability
    'ESP',  # employment status of parents
    'CIT',  # citizenship status
    'MIG',  # mobility status
    'MIL',  # military service
    'ANC',  # ancestry
    'NATIVITY',
    'DEAR',
    'DEYE',
    'DREM',
    'ESR',
    'ST',
    'FER',
    'GCL',
    'JWTR',
    # 'PUMA',
    # 'POWPUMA',
}


def split_X_Y_S(data, label_col: str, sensitive_col: str,
                ignore_cols=None, unawareness=False) -> tuple:
    ignore_cols = ignore_cols or []
    ignore_cols.append(label_col)
    if unawareness:
        ignore_cols.append(sensitive_col)

    feature_cols = [c for c in data.columns if c not in ignore_cols]

    return (
        data[feature_cols],                           # X
        data[label_col].to_numpy().astype(int),       # Y
        data[sensitive_col].to_numpy().astype(int),   # S
    )


def load_ACS_data(dir_path: str, task_name: str,
                  sensitive_col: str = None) -> pd.DataFrame:
    """Loads the given ACS task data from pre-generated datasets.

    Returns
    -------
    dict[str, tuple]
        A list of tuples, each tuple composed of (features, label,
        sensitive_attribute).
        The list is sorted as follows" [<train data tuple>, <test data tuple>,
        <val. data tuple>].
    """
    # Load task object
    task_obj = getattr(folktables, task_name)

    # Load train, test, and validation data
    data = dict()
    for data_type in ['train', 'test', 'validation']:
        # Construct file path
        path = Path(dir_path) / f"{task_name}.{data_type}.csv"

        if not path.exists():
            print(f"Couldn't find data\
                  for '{path.name}' \
                  (this is probably expected).")
            continue

        # Read data from disk
        df = pd.read_csv(path, index_col=0)

        # Set categorical columns
        cat_cols = ACS_CATEGORICAL_COLS & set(df.columns)
        df = df.astype({col: "category" for col in cat_cols})

        data[data_type] = split_X_Y_S(
            df,
            label_col=task_obj.target,
            sensitive_col=sensitive_col or task_obj.group,
        )

    return data


def human_sim(y, s):
    """Simulate human error in the labels.
    
    Parameters
    ----------
    y : np.ndarray
        The true labels.
    """
    m = np.zeros(len(y))
    for i in range(len(y)):
        if s[i] == 0:
            random = np.random.uniform(0, 1)
            if random < 0.85:
                m[i] = y[i]
            else:
                m[i] = 1-y[i]
        else:
            random = np.random.uniform(0, 1)
            if random < 0.6:
                m[i] = y[i]
            else:
                m[i] = 1-y[i]
    m = m.astype(int)
    return m


def generate_ACS():
    Dataset = dataset()
    #  Load and pre-process data
    all_data = load_ACS_data(
        dir_path=data_dir, task_name=ACS_TASK,
    )
    # Unpack into features, label, and group
    all_sets = []
    if "validation" in all_data:
        all_sets = ['train', 'test', 'validation']
    else:
        all_sets = ['train', 'test']
    for data_type in all_sets:
        Dataset.X[data_type], Dataset.y[data_type], Dataset.s[data_type] =\
              all_data[data_type]
        Dataset.M[data_type] =\
            human_sim(Dataset.y[data_type], Dataset.s[data_type])
        Dataset.MY[data_type] =\
            np.where(Dataset.y[data_type] == Dataset.M[data_type], 1, 0)
    Dataset.finalize()
    return Dataset
