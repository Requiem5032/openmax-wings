import glob
import re
import glob
import pandas as pd
import utils.constants as const

from sklearn.model_selection import StratifiedKFold


def get_species_list():
    data_dir = glob.glob('data/wings-2024/*')
    pattern_species = re.compile(r'(?:wings-2024/)([^/]*)$')
    species_list = []

    for data in data_dir:
        match_species = pattern_species.search(data)
        species_name = match_species.group(1)

        if species_name not in species_list:
            species_list.append(species_name)

    species_list = sorted(species_list)

    return species_list


def get_species_map():
    species_list = get_species_list()

    for species in const.UNKNOWN_SPECIES:
        species_list.remove(species)

    species_list.append('Unknown')
    species_labels = range(len(species_list))
    species_map = dict(zip(species_labels, species_list))

    return species_map


def get_data_df(species_map):
    df = pd.DataFrame(columns=['Path', 'Species_Name', 'Species_ID'])
    data_dir = glob.glob('data/wings-2024/*')
    pattern_species = re.compile(r'(?:wings-2024/)([^/]*)$')

    for species_dir in data_dir:
        temp_df = pd.DataFrame(columns=df.columns)
        specimen_list = glob.glob(species_dir + '/*')
        match_species = pattern_species.search(species_dir)
        species_name = match_species.group(1)

        if species_name in const.UNKNOWN_SPECIES:
            species_name = 'Unknown'

        for k, v in species_map.items():
            if species_name == v:
                species_label = k

        species_name_list = [species_name] * len(specimen_list)
        species_label_list = [species_label] * len(specimen_list)

        temp_df['Path'] = specimen_list
        temp_df['Species_Name'] = species_name_list
        temp_df['Species_ID'] = species_label_list

        df = pd.concat([df, temp_df], ignore_index=True)

    df = df.sort_values(by=['Species_ID']).reset_index(drop=True)
    return df


def get_comparison_df(aedes_only=True):
    df = pd.DataFrame(columns=['Path', 'Species_Name', 'Species_ID'])
    data_dir = glob.glob('data/wings-2024/*')
    pattern_species = re.compile(r'(?:wings-2024/)([^/]*)$')
    pattern_genus = re.compile(r'(^[a-zA-Z]{2})(?:_.*)$')
    aedes_label = 0

    for species_dir in data_dir:
        temp_df = pd.DataFrame(columns=df.columns)
        specimen_list = glob.glob(species_dir + '/*')

        match_species = pattern_species.search(species_dir)
        species_name = match_species.group(1)

        match_genus = pattern_genus.search(species_name)
        genus_name = match_genus.group(1)

        if aedes_only:
            if genus_name == 'Ae':
                species_label = aedes_label
                aedes_label += 1

            else:
                continue

        else:
            if genus_name == 'Ae':
                species_name = 'Aedes'
                species_label = 0
            else:
                species_name = 'Non-aedes'
                species_label = 1

        species_name_list = [species_name] * len(specimen_list)
        species_label_list = [species_label] * len(specimen_list)

        temp_df['Path'] = specimen_list
        temp_df['Species_Name'] = species_name_list
        temp_df['Species_ID'] = species_label_list

        df = pd.concat([df, temp_df], ignore_index=True)

    df = df.sort_values(by=['Species_ID']).reset_index(drop=True)
    return df


def filter_value(df, col, values, include=False):
    if include:
        return df[df[col].isin(values)]

    else:
        return df[~df[col].isin(values)]


def split_img_list(known_df, species_map):
    skf = StratifiedKFold(n_splits=5)
    X = known_df['Path'].to_list()
    y = known_df['Species_ID'].to_list()
    fold = 1

    for train, val in skf.split(X, y):
        train_df = pd.DataFrame(columns=known_df.columns)
        val_df = pd.DataFrame(columns=known_df.columns)

        train_path_list = []
        val_path_list = []

        train_name_list = []
        val_name_list = []

        train_id_list = []
        val_id_list = []

        for k in train:
            train_path_list.append(X[k])
            train_id_list.append(y[k])

        for k in val:
            val_path_list.append(X[k])
            val_id_list.append(y[k])

        for id in train_id_list:
            for k, v in species_map.items():
                if id == k:
                    train_name_list.append(v)
                    continue

        for id in val_id_list:
            for k, v in species_map.items():
                if id == k:
                    val_name_list.append(v)
                    continue

        train_df['Path'] = train_path_list
        train_df['Species_Name'] = train_name_list
        train_df['Species_ID'] = train_id_list
        train_df['Split'] = ['Train'] * len(train_id_list)

        val_df['Path'] = val_path_list
        val_df['Species_Name'] = val_name_list
        val_df['Species_ID'] = val_id_list
        val_df['Split'] = ['Val'] * len(val_id_list)

        split_df = pd.concat([train_df, val_df], ignore_index=True)
        split_df.to_csv(f'data/splits/data_fold_{fold}.csv', index=False)
        fold += 1


def split_comparison(aedes_only=True):
    data_df = get_comparison_df(aedes_only)

    if aedes_only:
        path_name = 'aedes_only'
        species_name_list = data_df['Species_Name'].unique()
        species_id_list = range(len(species_name_list))
        species_map = dict(zip(species_id_list, species_name_list))

    else:
        path_name = 'aedes_vs_non_aedes'
        species_name_list = data_df['Species_Name'].unique()
        species_id_list = range(len(species_name_list))
        species_map = dict(zip(species_id_list, species_name_list))

    skf = StratifiedKFold(n_splits=5)
    X = data_df['Path'].to_list()
    y = data_df['Species_ID'].to_list()
    fold = 1

    for train, val in skf.split(X, y):
        train_df = pd.DataFrame(columns=data_df.columns)
        val_df = pd.DataFrame(columns=data_df.columns)

        train_path_list = []
        val_path_list = []

        train_name_list = []
        val_name_list = []

        train_id_list = []
        val_id_list = []

        for k in train:
            train_path_list.append(X[k])
            train_id_list.append(y[k])

        for k in val:
            val_path_list.append(X[k])
            val_id_list.append(y[k])

        for id in train_id_list:
            for k, v in species_map.items():
                if id == k:
                    train_name_list.append(v)
                    continue

        for id in val_id_list:
            for k, v in species_map.items():
                if id == k:
                    val_name_list.append(v)
                    continue

        train_df['Path'] = train_path_list
        train_df['Species_Name'] = train_name_list
        train_df['Species_ID'] = train_id_list
        train_df['Split'] = ['Train'] * len(train_id_list)

        val_df['Path'] = val_path_list
        val_df['Species_Name'] = val_name_list
        val_df['Species_ID'] = val_id_list
        val_df['Split'] = ['Val'] * len(val_id_list)

        split_df = pd.concat([train_df, val_df], ignore_index=True)
        split_df.to_csv(f'data/comparison/{path_name}/data_fold_{fold}.csv', index=False)
        fold += 1
