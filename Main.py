import io
import string
from ctypes import Union

import pandas as pd
from pandas import Series, DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


numberOfMatches = 150
"""
players = pd.read_csv('players.csv', nrows=10 * numberOfMatches)
matches = pd.read_csv('match.csv', nrows=numberOfMatches)
heroes = pd.read_csv('hero_names.csv')
items = pd.read_csv('item_ids.csv')
"""


class PlayerFormatter:
    players: DataFrame

    def __init__(self, players_path: string, heroes_path: string, items_path: string):
        self.readFiles(players_path, heroes_path, items_path)

    def readFiles(self, players_path: string, heroes_path: string, items_path: string):
        self.players = pd.read_csv(players_path, nrows=10 * numberOfMatches)
        heroes = pd.read_csv(heroes_path)
        items = pd.read_csv(items_path)
        self.formatPlayerHeroes(heroes)
        self.formatPlayerItems(items)

    def formatPlayerHeroes(self, heroes_temp: DataFrame):
        hero_lookup = dict(zip(heroes_temp['hero_id'], heroes_temp['localized_name']))
        hero_lookup[0] = 'Unknown'
        self.players['hero'] = self.players['hero_id'].apply(lambda _id: hero_lookup[_id])

    def formatPlayerItems(self, items_temp: DataFrame):
        item_lookup = dict(zip(items_temp['item_id'], items_temp['item_name']))
        item_lookup[0] = 'Unknown'

        def find_item(_id):
            return item_lookup.get(_id, 'u_' + str(_id))

        self.players['item_0'] = self.players['item_0'].apply(find_item)
        self.players['item_1'] = self.players['item_1'].apply(find_item)
        self.players['item_2'] = self.players['item_2'].apply(find_item)
        self.players['item_3'] = self.players['item_3'].apply(find_item)
        self.players['item_4'] = self.players['item_4'].apply(find_item)
        self.players['item_5'] = self.players['item_5'].apply(find_item)


"""
hero_lookup = dict(zip(heroes['hero_id'], heroes['localized_name']))
hero_lookup[0] = 'Unknown'
players['hero'] = players['hero_id'].apply(lambda _id: hero_lookup[_id])

item_lookup = dict(zip(items['item_id'], items['item_name']))
item_lookup[0] = 'Unknown'


def find_item(_id):
    return item_lookup.get(_id, 'u_' + str(_id))


players['item_0'] = players['item_0'].apply(find_item)
players['item_1'] = players['item_1'].apply(find_item)
players['item_2'] = players['item_2'].apply(find_item)
players['item_3'] = players['item_3'].apply(find_item)
players['item_4'] = players['item_4'].apply(find_item)
players['item_5'] = players['item_5'].apply(find_item)
print(players['item_0'])
"""

players = PlayerFormatter("players.csv", "hero_names.csv", "item_ids.csv").players

player_heroes = pd.get_dummies(players['hero'])

item0 = pd.get_dummies(players['item_0'].fillna(0))
item1 = pd.get_dummies(players['item_1'].fillna(0))
item2 = pd.get_dummies(players['item_2'].fillna(0))
item3 = pd.get_dummies(players['item_3'].fillna(0))
item4 = pd.get_dummies(players['item_4'].fillna(0))
item5 = pd.get_dummies(players['item_5'].fillna(0))
player_items = item0 \
    .add(item1, fill_value=0) \
    .add(item2, fill_value=0) \
    .add(item3, fill_value=0) \
    .add(item4, fill_value=0) \
    .add(item5, fill_value=0)

radiant_cols = list(map(lambda s: 'radiant_' + s, player_heroes.columns.values))
dire_cols = list(map(lambda s: 'dire_' + s, player_heroes.columns.values))
radiant_items_cols = list(map(lambda s: 'radiant_' + str(s), player_items.columns.values))
dire_items_cols = list(map(lambda s: 'dire_' + str(s), player_items.columns.values))

X = None

# if isfile('mapped_match_hero_item.csv'):
#    X = pd.read_csv('mapped_match_hero_item.csv')
# else:

radiant_heroes = []
dire_heroes = []
radiant_items = []
dire_items = []

for _id, _index in players.groupby('match_id').groups.items():
    radiant_heroes.append(player_heroes.iloc[_index][:5].sum().values)
    dire_heroes.append(player_heroes.iloc[_index][5:].sum().values)
    radiant_items.append(player_items.iloc[_index][:5].sum().values)
    dire_items.append(player_items.iloc[_index][5:].sum().values)

radiant_heroes = pd.DataFrame(radiant_heroes, columns=radiant_cols)
dire_heroes = pd.DataFrame(dire_heroes, columns=dire_cols)
radiant_items = pd.DataFrame(radiant_items, columns=radiant_items_cols)
dire_items = pd.DataFrame(dire_items, columns=dire_items_cols)
X = pd.concat([radiant_heroes, radiant_items, dire_heroes, dire_items], axis=1)
matches = pd.read_csv('match.csv', nrows=numberOfMatches)
X.to_csv('mapped_match_hero_item.csv', index=False)
y = matches['radiant_win'].apply(lambda win: 1 if win else 0)
classes = ['Dire Win', 'Radiant Win']


# jemali = pd.Series(X).apply(lambda i: item_lookup[i]).value_counts().plot(kind='bar')
# plt.show()
# _ = pd.Series(y).apply(lambda i: classes[i]).value_counts().plot('bar')


def build_decision_tree(X, y):
    dt = DecisionTreeClassifier()
    print('CV score:', cross_val_score(estimator=dt, X=X, y=y).mean())
    dt.fit(X, y)
    return dt


dt = build_decision_tree(X=X, y=y)


def get_tree_stats(dts):
    feature_importances = dict(zip(X.columns, dts.feature_importances_))
    feature_importances = pd.Series(feature_importances).sort_values(ascending=False)
    return feature_importances.head(20)


stats = get_tree_stats(dts=dt)
stats.plot()
import matplotlib.pyplot as plt

plt.show()
