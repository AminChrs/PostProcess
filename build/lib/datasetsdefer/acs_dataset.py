from pathlib import Path
import pandas as pd
import numpy as np
import folktables
ACS_TASK = "ACSIncome"
SEED = 42
EPS = 1e-6
data_dir = Path("~").expanduser() / "data" / "folktables" / "train=0.6_test=\
    0.2_validation=0.2_max-groups=4"


class dataset:
    def __init__():
        pass


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


def human_sim(y_train, y_test, y_val, s_train, s_test, s_val, val=True):
    """Simulate human error in the labels.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Testing labels.
    y_val : np.ndarray
        Validation labels.
    s_train : np.ndarray
        Training sensitive attributes.
    s_test : np.ndarray
        Testing sensitive attributes.
    s_val : np.ndarray
        Validation sensitive attributes.
    val : bool, optional
        Whether to simulate human error in the validation set, by default True.
    """
    M_test = np.zeros(len(y_test))
    M_val = np.zeros(len(y_val))

    def noisy(y, s):
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
    M_test = noisy(y_test, s_test)
    M_train = noisy(y_train, s_train)
    if val:
        M_val = noisy(y_val, s_val)
    else:
        print("No validation data.")
    return M_train, M_test, M_val


def generate():
    #  Load and pre-process data
    all_data = load_ACS_data(
        dir_path=data_dir, task_name=ACS_TASK,
    )
    # Unpack into features, label, and group
    X_train, y_train, s_train = all_data["train"]
    X_test, y_test, s_test = all_data["test"]
    if "validation" in all_data:
        X_val, y_val, s_val = all_data["validation"]
    else:
        print("No validation data.")

    actual_prevalence = np.sum(y_train) / len(y_train)
    print(f"Global prevalence: {actual_prevalence:.1%}")
    if "validation" in all_data:
        val_data = True
    M_train, M_test, M_val = human_sim(y_train, y_test, y_val, s_train, s_test,
                                       s_val,
                                       val=val_data)

    # find the new labels that are M==Y
    MY_train = np.where(y_train == M_train, 1, 0)
    MY_test = np.where(y_test == M_test, 1, 0)
    if "validation" in all_data:
        MY_val = np.where(y_val == M_val, 1, 0)
        # print("MY_val:", MY_val)
    else:
        print("No validation data.")
    Dataset = dataset()
    Dataset.X_train = X_train
    Dataset.y_train = y_train
    Dataset.s_train = s_train
    Dataset.X_test = X_test
    Dataset.y_test = y_test
    Dataset.s_test = s_test
    Dataset.X_val = X_val
    Dataset.y_val = y_val
    Dataset.s_val = s_val
    Dataset.M_train = M_train
    Dataset.M_test = M_test
    Dataset.M_val = M_val
    Dataset.MY_train = MY_train
    Dataset.MY_test = MY_test
    Dataset.MY_val = MY_val
    return Dataset
