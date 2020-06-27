import string

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

numberOfMatches = 150


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


class RadiantDireData:
    radiant_heroes = []
    dire_heroes = []
    radiant_items = []
    dire_items = []

    def __init__(self, players: DataFrame):
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

        for _id, _index in players.groupby('match_id').groups.items():
            self.radiant_heroes.append(player_heroes.iloc[_index][:5].sum().values)
            self.dire_heroes.append(player_heroes.iloc[_index][5:].sum().values)
            self.radiant_items.append(player_items.iloc[_index][:5].sum().values)
            self.dire_items.append(player_items.iloc[_index][5:].sum().values)

        self.radiant_heroes = pd.DataFrame(self.radiant_heroes, columns=radiant_cols)
        self.dire_heroes = pd.DataFrame(self.dire_heroes, columns=dire_cols)
        self.radiant_items = pd.DataFrame(self.radiant_items, columns=radiant_items_cols)
        self.dire_items = pd.DataFrame(self.dire_items, columns=dire_items_cols)


class DecisionTree:
    X: DataFrame
    Y: DataFrame
    dt: DecisionTreeClassifier

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.build_decision_tree()

    def build_decision_tree(self):
        self.dt = DecisionTreeClassifier()
        self.dt.fit(self.X, self.Y)

    def print_cross_validation(self):
        print('CV score:', cross_val_score(estimator=self.dt, X=self.X, y=self.Y).mean())

    def get_tree_stats(self):
        feature_importance = dict(zip(X.columns, self.dt.feature_importances_))
        feature_importance = pd.Series(feature_importance).sort_values(ascending=False)
        return feature_importance.head(20)


players = PlayerFormatter("players.csv", "hero_names.csv", "item_ids.csv").players
data = RadiantDireData(players)

X = pd.concat([data.radiant_heroes, data.radiant_items, data.dire_heroes, data.dire_items], axis=1)
X.to_csv('mapped_match_hero_item.csv', index=False)

matches = pd.read_csv('match.csv', nrows=numberOfMatches)
Y = matches['radiant_win'].apply(lambda win: 1 if win else 0)

dt = DecisionTree(X, Y)
stats = dt.get_tree_stats()

stats.plot()

plt.show()
