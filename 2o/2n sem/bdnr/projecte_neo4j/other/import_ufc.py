import pandas as pd
from py2neo import Graph

#get fight data
fights = pd.read_csv('fight_hist.csv')
#ignore DQs
fights = fights[fights.method != 'DQ']

graph = Graph(password="Planoles")

#load data into graph
tx = graph.begin()
for index, row in fights[fights.result == 'L'].iterrows():
    tx.evaluate('''
       MERGE (a: fighter {name: $fighter})
       MERGE (b: fighter {name: $opponent})
       MERGE (b)-[r:win_to {date: $date, division: $division, method: $method}]->(a)
    ''', parameters = {'fighter': row['fighter'], 'opponent': row['opponent'], 'date':row['date'], 
                       'method':row['method'], 'division':row['division']})
tx.commit()
