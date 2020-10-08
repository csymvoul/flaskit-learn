from flask import Flask
from flaskit.flaskit_learn import clustering

dbscan = clustering()
res = dbscan.fit()
print(res)