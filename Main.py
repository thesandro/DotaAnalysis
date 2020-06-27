import string
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import os.path
from os import path

numberOfMatches = 150

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtCore import pyqtSlot
import pandas as pd

"""
    თავდაპირველად player.csv შეიცავს id-ებს hero-ს და item-ის სახელის მაგივრად,
    ამიტომ საჭიროებს დაფორმატებას.
    კლასს კონსტრუქტორში გადაეწოდება სამი ფაილის მისამართი string ტიპად და ხდება შემდგომ DataFrame-ს დამუშავება.
"""


class PlayerFormatter:
    players: DataFrame

    # კონსტრუქტორში ხდება ფაილების წაკითხვა და შემდგომ დაფორმატებული player DataFrame-ს დასეტვა.
    def __init__(self, players_path: string, heroes_path: string, items_path: string):
        # ვიძახებთ ფაილების წამკითხველ მეთოდს რომელსაც გადავცემთ ფაილების მდებარეობას.
        self.readFiles(players_path, heroes_path, items_path)

    # კითხულოვ ფაილებს pd.read_csv-ს მეშვეობით და შემდგომ იძახებს player-ის დაფორმატების მეთოდებს
    # სადაც პირველად player-ი შეიცავს id-ებს სახელების მაგივრად
    def readFiles(self, players_path: string, heroes_path: string, items_path: string):
        self.players = pd.read_csv(players_path, nrows=10 * numberOfMatches)
        heroes = pd.read_csv(heroes_path)
        items = pd.read_csv(items_path)
        self.formatPlayerHeroes(heroes)
        self.formatPlayerItems(items)

    # გადაეცემა heroes და შემდგომ ხდება მის მიხედვით player-ის დაფორმატება
    def formatPlayerHeroes(self, heroes_temp: DataFrame):
        # თავიდან გარდავქმნი tuple-დ რადგან პირდაპირ გადაწოდება არ არის შესაძლებელი
        # შემდგომ გადაგვყავს dict-ში
        hero_lookup = dict(zip(heroes_temp['hero_id'], heroes_temp['localized_name']))
        # ვანიჭებთ Unknown-ს იმ შემთხვევაში თუ hero-ს სახელი არ მოიძებნება
        hero_lookup[0] = 'Unknown'
        # ვცვლით hero_id-ს hero-ს სახელით რადგან მოხდეს შემდეგში მონაცამების მარტივი ვიზუალიზაცია
        self.players['hero'] = self.players['hero_id'].apply(lambda _id: hero_lookup[_id])

    # გადაეცემა items და შემდგომ ხდება მის მიხედვით player-ის დაფორმატება
    def formatPlayerItems(self, items_temp: DataFrame):
        # თავიდან გარდავქმნი tuple-დ რადგან პირდაპირ გადაწოდება არ არის შესაძლებელი
        # შემდგომ გადაგვყავს dict-ში
        item_lookup = dict(zip(items_temp['item_id'], items_temp['item_name']))
        # ვანიჭებთ Unknown-ს იმ შემთხვევაში თუ item-ის სახელი არ მოიძებნება
        item_lookup[0] = 'Unknown'

        # ვსაზღვრავთ ფუნქციას რომ აღარ გავიმეოროთ კოდი ბევრჯერ
        def find_item(_id):
            # თუ ვერ იპოვა item-ის სახელი მაშინ ხდება u_ id -ის მიხედვით სახელის მინიჭება
            return item_lookup.get(_id, 'u_' + str(_id))

        # რადგან მოთაშეს აქვს 6 ნივთი, ვიმეორებთ ამას 6-ივე ადგილისთვის
        self.players['item_0'] = self.players['item_0'].apply(find_item)
        self.players['item_1'] = self.players['item_1'].apply(find_item)
        self.players['item_2'] = self.players['item_2'].apply(find_item)
        self.players['item_3'] = self.players['item_3'].apply(find_item)
        self.players['item_4'] = self.players['item_4'].apply(find_item)
        self.players['item_5'] = self.players['item_5'].apply(find_item)


# ვყოფთ Player-ის გმირს და ნივთებს გუნდების მიხედით
class RadiantDireData:
    radiant_heroes = []
    dire_heroes = []
    radiant_items = []
    dire_items = []

    #   კონსტრუქტორში გადავცემთ დამუშავებულ player DataFrame-ს
    def __init__(self, players: DataFrame):
        # გადმოგვყავს კატეგორიული ცვლადი მატრიცულ ტიპად
        # მაგ: თუ არის 0 index-ზე Hero-ს სახელი Rubick მაშინ შემდგომ შედგენილ მატრიცაში იქნება 1
        # და სხვა ყველგან 0-ები
        player_heroes = pd.get_dummies(players['hero'])
        item0 = pd.get_dummies(players['item_0'].fillna(0))
        item1 = pd.get_dummies(players['item_1'].fillna(0))
        item2 = pd.get_dummies(players['item_2'].fillna(0))
        item3 = pd.get_dummies(players['item_3'].fillna(0))
        item4 = pd.get_dummies(players['item_4'].fillna(0))
        item5 = pd.get_dummies(players['item_5'].fillna(0))

        # ვაკეთებთ ერთ dataFrame-ში ყველა ნივთის გაერთიანებას
        player_items = item0 \
            .add(item1, fill_value=0) \
            .add(item2, fill_value=0) \
            .add(item3, fill_value=0) \
            .add(item4, fill_value=0) \
            .add(item5, fill_value=0)
        print(player_items)
        # ვქმნით სვეტებს ორივე გუნდის ტიპისთვის (Dire და Radiant)
        radiant_cols = list(map(lambda s: 'radiant_' + s, player_heroes.columns.values))
        dire_cols = list(map(lambda s: 'dire_' + s, player_heroes.columns.values))
        radiant_items_cols = list(map(lambda s: 'radiant_' + str(s), player_items.columns.values))
        dire_items_cols = list(map(lambda s: 'dire_' + str(s), player_items.columns.values))

        # ვალაგებთ match_id-ით და ვახდენთ მოთამაშეების დაყოფას იმის მიხედვით თუ რომელ გუნდშია
        # გუნდის პირველი 5-თი მოთამაშე ლოგიკურად რადიანტშია და დანარჩენი 5 მოთამაშე dire-ში
        for _id, _index in players.groupby('match_id').groups.items():
            self.radiant_heroes.append(player_heroes.iloc[_index][:5].sum().values)
            self.dire_heroes.append(player_heroes.iloc[_index][5:].sum().values)
            self.radiant_items.append(player_items.iloc[_index][:5].sum().values)
            self.dire_items.append(player_items.iloc[_index][5:].sum().values)

        # ვანიჭებთ სახელებს სვეტების მიხედვით
        self.radiant_heroes = pd.DataFrame(self.radiant_heroes, columns=radiant_cols)
        self.dire_heroes = pd.DataFrame(self.dire_heroes, columns=dire_cols)
        self.radiant_items = pd.DataFrame(self.radiant_items, columns=radiant_items_cols)
        self.dire_items = pd.DataFrame(self.dire_items, columns=dire_items_cols)


# კლასში ხდება ერთად გაერთიანებული სხვადასხვა გუნდის Item-ების და Hero-ების გადაწოდება X: DataFrame
# მეორე პარამეტრად გადაეწოდება radiant-ის მოგების და წაგების DataFrame
class DecisionTree:
    X: DataFrame
    Y: DataFrame
    dt: DecisionTreeClassifier

    # ვუსეტავთ ორივე გადაწოდებულ ცვლადს და ვიწყებთ გადაწყვეეტილებების ხის აგებას
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.build_decision_tree()

    # ვქმნით ხის კლასის instance-ს და შემდგომ ვახდენთ ხის დამუშავებას X და y ცვლადებით
    def build_decision_tree(self):
        self.dt = DecisionTreeClassifier()
        self.dt.fit(self.X, self.Y)

    # ვახდენთ ვალიდაციას თუ რამდენად ახდენს თამაშძე განსხვავებული გმირები, ნივთები და გუნდის ტიპი
    # თუ ქულა აღებატება 50-55% ეს ნიშნავს რომ ნამდვილად გავლენას ახდენს
    def print_cross_validation(self):
        print('CV score:', cross_val_score(estimator=self.dt, X=self.X, y=self.Y).mean())

    # ვახდენთ მონაცემების dict-ად გარდაქმნას და შემდგომ ვალაგებთ კლებადობით,
    # რომ გავარკვიოთ თუ რომელი ნივთი/გმირი გადაწყვეტს ყველაზე მეტად მოგებას
    # და ვაბრუნებთ როგორც Series

    def get_tree_stats(self):
        feature_importance = dict(zip(self.X.columns, self.dt.feature_importances_))
        feature_importance = pd.Series(feature_importance).sort_values(ascending=False)
        return feature_importance


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.textboxPlayers = QLineEdit(self)
        self.textboxMatches = QLineEdit(self)
        self.textboxHeroes = QLineEdit(self)
        self.textboxItems = QLineEdit(self)
        self.title = 'Dota 2 Analysis'
        self.left = 10
        self.top = 10
        self.width = 360
        self.initUI()

    def initUI(self):
        self.textboxPlayers.setText("players")
        self.textboxMatches.setText("match")
        self.textboxHeroes.setText("hero_names")
        self.textboxItems.setText("item_ids")
        self.setWindowTitle(self.title)
        # Create textbox Players
        y = 20
        self.textboxPlayers.move(20, y)
        self.textboxPlayers.resize(280, 40)
        y += 50
        self.textboxMatches.move(20, y)
        self.textboxMatches.resize(280, 40)
        y += 50
        self.textboxHeroes.move(20, y)
        self.textboxHeroes.resize(280, 40)
        y += 50
        self.textboxItems.move(20, y)
        self.textboxItems.resize(280, 40)

        y += 50
        # Create a button in the window

        self.button = QPushButton('Predict winning heroes/items', self)
        self.button.move(20, y)
        y += 50

        self.setGeometry(self.left, self.top, self.width, y)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        path_dict = dict()
        path_dict["players"] = self.textboxPlayers.text() + ".csv"
        path_dict["matches"] = self.textboxMatches.text() + ".csv"
        path_dict["heroes"] = self.textboxHeroes.text() + ".csv"
        path_dict["items"] = self.textboxItems.text() + ".csv"

        for key in path_dict.keys():
            if not path.exists(path_dict[key]):
                print("file not found: " + key)
                return

        # გადავცემთ ფაილების სახელებს კონსტრუქტორიში
        players = PlayerFormatter(path_dict["players"], path_dict["heroes"], path_dict["items"]).players
        # შემდგომ ვქმნით დანაწევრებულ მონაცემებს (გუნდის ტიპის მიხედვით)
        data = RadiantDireData(players)

        # ვაერთიანებთ ყველა მონაცემბს ერთ DataFrame-ში
        X = pd.concat([data.radiant_heroes, data.radiant_items, data.dire_heroes, data.dire_items], axis=1)
        # X.to_csv('mapped_match_hero_item.csv', index=False)

        # ვკითხულობთ match-ებს აქ არის აღნიშული თუ რომელი მატჩი მოიგო dire/radiant-მა
        matches = pd.read_csv('match.csv', nrows=numberOfMatches)
        # რადგან ფრე შეუძლებელია რომ მოხდებს,
        # ამიტომ გადაგვყავს უფრო მარტივ სტრუქტორაში 1 იმ შემთხევაშია თუ radiant-მა მოიგო lambda
        Y = matches['radiant_win'].apply(lambda win: 1 if win else 0)

        # ვაგებთ ხეს
        dt = DecisionTree(X, Y)
        # ვიღებთ სტატებს როგორც Series
        stats = dt.get_tree_stats()

        # ვახდენთ ვიზუალურად წარმოდგენას
        # თუ რომელი ნივთი/გმირი ახდენს მოგებაზე გავლენას

        # stats.plot()
        # self.show()


app = QApplication(sys.argv)
application = App()
application.show()
sys.exit(app.exec())
