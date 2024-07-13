import pandasql
import pandas as pd
from pandasql import sqldf
run_query = lambda query: sqldf(query, globals())
df = pd.DataFrame([{'test':1, 'test2':2}, {'test':3, 'test2':4}])
# Simple select query
query_1 = """
SELECT *
FROM df;
"""
result_1 = run_query(query_1)
print(result_1)