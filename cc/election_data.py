# Data obtained from
# https://commonslibrary.parliament.uk/parliament-and-elections/elections-elections/constituency-data-election-results/

import numpy as np
import pandas as pd

np.random.seed(0) # important for the train/test split

def create_Y(num_parties):
    # Creates a data frame with the vote share for each constituency,
    # including a remainder category for smaller parties + independents
    election_file = "../in/uk_election_2019.csv"
    raw_data = pd.read_csv(election_file)

    # Sum total vote share by political party to find which are the largest parties
    res = raw_data.pivot_table(index='party_name', values='share', aggfunc=sum)
    res = res.sort_values(by='share', ascending=False)
    largest_parties = res.index[0:num_parties]

    # Separate the main parties from the remainder class
    split_idx = raw_data['party_name'].isin(largest_parties)
    main_frame = raw_data[split_idx]
    rest_frame = raw_data[~split_idx]
    rest_frame = rest_frame.pivot_table(index='constituency_name', values='share', aggfunc=sum)
    rest_frame = rest_frame.rename({'share': 'rest'}, axis=1)

    # Merge the main parties with the remainder into one frame
    res = main_frame.pivot_table(index='constituency_name', columns='party_name', values='share')
    res = pd.merge(res, rest_frame, left_index=True, right_index=True, how='outer')
    res = res.fillna(0.0)

    return res

def create_X():
    # Creates a data frame with the predictors for each constituency.
    # As our modeling here is for illustration of the CC vs Dirichlet,
    # we stick to the few predictors available in the original data source
    election_file = "../in/uk_election_2019.csv"
    raw_data = pd.read_csv(election_file)

    to_keep = ['constituency_name', 'country_name', 'constituency_type', 'electorate', 'turnout_2017']
    keep_frame = raw_data[to_keep].drop_duplicates()
    # Normalize the electorate
    keep_frame['electorate'] = (keep_frame['electorate'] - np.mean(keep_frame['electorate'])) / np.std(keep_frame['electorate'])
    # Convert our categorical variables to dummy indicator variables
    dummy1 = pd.get_dummies(keep_frame['constituency_type'])
    dummy2 = pd.get_dummies(keep_frame['country_name'])
    keep_frame = pd.concat([keep_frame, dummy1, dummy2], axis=1)
    # Drop the baseline classes from our dataframe to avoid collinearity
    res = keep_frame.drop(['country_name', 'constituency_type', 'County', 'England'], axis=1)
    res = res.set_index('constituency_name')

    return res

def create_election_data(num_parties=3):
    # Stick together the predictors and response into a single data frame
    Y = create_Y(num_parties)
    X = create_X()
    n = Y.shape[0]
    res = pd.merge(Y, X, left_index=True, right_index=True)

    # Set up training/test split
    res['test'] = np.random.binomial(1, 0.2, n)

    # Save data
    res.to_pickle('../in/election_data_{}party.pkl'.format(num_parties))

def load_election_data(num_parties=3):
    return pd.read_pickle('../in/election_data_{}party.pkl'.format(num_parties))

if __name__ == '__main__':
    create_election_data(3)
    create_election_data(4)
    create_election_data(5)
