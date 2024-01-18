import logging
import pandas as pd
from cachetools import cached, TTLCache

# from utils.speedup import timing
from config import config
from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickhouseError

log = logging.getLogger('werkzeug')


class ClickhouseQueryExecutor:

    def __init__(self,
                 database: str = 'reliability'):

        self.config = config['CLICKHOUSE']
        self.database = database

    def init_client(self) -> Client:
        """ Returns a Clickhouse client"""

        return Client(host=self.config['HOST'],
                      port=self.config['PORT'],
                      user=self.config['CREDENTIALS']['USER'],
                      password=self.config['CREDENTIALS']['PASSWORD'],
                      database=self.database,
                      settings=self.config['SETTINGS'])

    # @timing
    @cached(cache=TTLCache(maxsize=1000000000, ttl=600),
            key=lambda *args, **kwargs: hash(args[1]) if not kwargs else hash(kwargs['query']))
    def execute(self, query: str) -> pd.DataFrame:
        client = self.init_client()
        with client as native:
            return native.query_dataframe(query)


# Some useful methods
def check_table_exists(ch_executor: ClickhouseQueryExecutor, table_name: str) -> bool:
    exists_table_query = f'EXISTS TABLE {table_name}'
    query_result = ch_executor.execute(exists_table_query)
    result = False
    if "result" not in query_result:
        raise ClickhouseError(f"The 'result' key not found in: {query_result}")
    if query_result["result"][0] == 1:
        result = True
    return result


if __name__ == '__main__':
    executor = ClickhouseQueryExecutor()
    # testing caching works as expected by looking at timing results
    executor.execute('SHOW TABLES')
    executor.execute(query='SHOW TABLES')
    executor.execute('SELECT * FROM assets LIMIT 10')
    executor.execute('SHOW TABLES')
    executor.execute('SELECT * FROM assets LIMIT 10')
