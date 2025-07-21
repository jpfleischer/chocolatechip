import os
import time
import json
import pandas as pd
import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from datetime import datetime, timedelta

from contextlib import contextmanager


class MySQLConnector:
    def __init__(self):
        # load config once
        script_dir = os.path.dirname(os.path.abspath(__file__))
        is_docker = bool(os.environ.get('in_docker', False))
        if is_docker:
            self.config = {
                "host": os.getenv("CC_host"),
                "user": os.getenv("CC_user"),
                "passwd": os.getenv("CC_passwd"),
                "db": os.getenv("CC_db"),
                "testdb": os.getenv("CC_testdb"),
                "port": int(os.getenv("CC_port", 3306))
            }
        else:
            env_path = os.path.join(script_dir, "login.env")
            if not os.path.exists(env_path):
                raise FileNotFoundError(f"Could not find {env_path}")
            load_dotenv(env_path)
            self.config = {
                "host": os.getenv("host"),
                "user": os.getenv("user"),
                "passwd": os.getenv("passwd"),
                "db": os.getenv("db"),
                "testdb": os.getenv("testdb"),
                "port": int(os.getenv("port", 3306))
            }

    def _connect(self, *, streaming: bool = False):
        """
        Centralize all pymysql.connect(...) calls.
        If streaming=True, we return a server-side cursor, with timeouts.
        """
        kwargs = {
            "host": self.config["host"],
            "user": self.config["user"],
            "passwd": self.config["passwd"],
            "db": self.config["testdb"],
            "port": self.config["port"]
        }
        if streaming:
            kwargs.update({
                "connect_timeout": 10,
                "read_timeout": 60,
                "write_timeout": 60,
                "cursorclass": pymysql.cursors.SSCursor
            })
        return pymysql.connect(**kwargs)
    

    @contextmanager
    def cursor(self, *, streaming: bool = False):
        """
        Yields a cursor (SSCursor if streaming=True), and closes the connection
        when the with-block exits.
        """
        conn = self._connect(streaming=streaming)
        try:
            cur = conn.cursor()
            yield cur
        finally:
            conn.close()


    def fetchPeriodicTrackCounts(self,
                                 intersec_id: int,
                                 cam_id:     int,
                                 start:      str,
                                 end:        str,
                                 pedestrian_counting: bool | None = None
                                 ) -> pd.DataFrame:
        """
        15-minute bins, DISTINCT track_id per bin.
        """
        sql = """
        SELECT
          FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(timestamp)/900)*900) AS period_start,
          COUNT(DISTINCT track_id)                     AS count
        FROM RealTrackProperties
        WHERE timestamp BETWEEN %s AND %s
          AND intersection_id = %s
          AND camera_id      = %s
          AND isAnomalous    = 0
        """
        params = [start, end, intersec_id, cam_id]

        if pedestrian_counting is True:
            sql += " AND `class` = %s"
            params.append("pedestrian")
        elif pedestrian_counting is False:
            sql += " AND `class` != %s"
            params.append("pedestrian")

        sql += " GROUP BY period_start ORDER BY period_start;"

        with self._connect() as conn:
            df = pd.read_sql(sql, con=conn, params=params)

        df['period_start'] = pd.to_datetime(df['period_start'])
        return df

    def fetchPeriodicTrackCountsPerMinute(self,
                                          intersec_id: int,
                                          cam_id:     int,
                                          start:      str,
                                          end:        str,
                                          pedestrian_counting: bool | None = None
                                          ) -> pd.DataFrame:
        """
        1-minute bins, DISTINCT track_id per minute.
        """
        sql = """
        SELECT
          FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(timestamp)/60)*60) AS period_start,
          COUNT(DISTINCT track_id)                     AS count
        FROM RealTrackProperties
        WHERE timestamp BETWEEN %s AND %s
          AND intersection_id = %s
          AND camera_id      = %s
          AND isAnomalous    = 0
        """
        params = [start, end, intersec_id, cam_id]

        if pedestrian_counting is True:
            sql += " AND `class` = %s"
            params.append("pedestrian")
        elif pedestrian_counting is False:
            sql += " AND `class` != %s"
            params.append("pedestrian")

        sql += " GROUP BY period_start ORDER BY period_start;"

        with self._connect() as conn:
            df = pd.read_sql(sql, con=conn, params=params)

        df['period_start'] = pd.to_datetime(df['period_start'])
        return df
    

    def fetchConflictRecords(self, intersec_id: int, p2v: int, start: str, end: str) -> pd.DataFrame:
        """
        Fetch raw conflict rows (timestamp, cluster1, cluster2) for a given intersection/p2v/time range,
        using a single pandas.read_sql call instead of streaming.
        """
        sql = """
        SELECT
          timestamp,
          cluster1,
          cluster2
        FROM TTCTable
        WHERE intersection_id = %s
          AND p2v           = %s
          AND include_flag  = 1
          AND timestamp BETWEEN %s AND %s
        """
        params = (intersec_id, p2v, start, end)
        # This executes in one shot, which is faster than handleRequest’s fetch‐many loop
        df = pd.read_sql(sql, con=self._connect(), params=params)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    

    def fetchConflictCoordinates(self, intersec_id: int, p2v: int, start: str, end: str) -> pd.DataFrame:
        """
        Fetch timestamp, numeric x/y and cluster codes for P2V or V2V.
        """
        sql = '''
        SELECT
          timestamp,
          cluster1,
          cluster2,
          conflict_x,
          conflict_y
        FROM TTCTable
        WHERE intersection_id = %s
          AND p2v           = %s
          AND include_flag  = 1
          AND timestamp BETWEEN %s AND %s
        '''
        df = pd.read_sql(sql, con=self._connect(), params=(intersec_id, p2v, start, end))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    

    def fetchHourlyConflictCounts(self,
                                  intersec_id: int,
                                  p2v:         int,
                                  start:       str,
                                  end:         str
                                  ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns [date, hour, count], where
        'count' = number of distinct (unique_ID1, unique_ID2) pairs
        in TTCTable for that intersection, within [start,end],
        with include_flag=1 and the given p2v flag.
        """
        sql = """
        SELECT
          DATE(timestamp)                    AS date,
          HOUR(timestamp)                    AS hour,
          COUNT(DISTINCT CONCAT(unique_ID1,'_',unique_ID2)) AS count
        FROM TTCTable
        WHERE intersection_id = %s
          AND p2v             = %s
          AND include_flag    = 1
          AND timestamp BETWEEN %s AND %s
        GROUP BY date, hour
        ORDER BY date, hour;
        """
        params = [intersec_id, p2v, start, end]

        with self._connect() as conn:
            df = pd.read_sql(sql, con=conn, params=params)

        # Make sure 'date' is a datetime.date and 'hour' is int
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['hour'] = df['hour'].astype(int)
        return df


    def handleRequest(self, params: dict, df_type: str) -> pd.DataFrame:
        """
        Raw-row streaming for conflicts or full track dump.
        """
        print(f"[DB] Connecting to {self.config['testdb']}...", end="", flush=True)
        t0 = time.time()
        conn = self._connect(streaming=True)
        print(f" done in {time.time()-t0:.2f}s")

        try:
            with conn.cursor() as cursor:
                if df_type == "conflict":
                    query = """
                    SELECT *
                      FROM TTCTable
                     WHERE intersection_id = %s
                       AND p2v             = %s
                       AND include_flag    = 1
                       AND timestamp BETWEEN %s AND %s
                    """
                    sql_params = (
                        params['intersec_id'],
                        params['p2v'],
                        params['start_date'],
                        params['end_date']
                    )
                else:
                    query = """
                    SELECT timestamp, track_id, class
                      FROM RealTrackProperties
                     WHERE timestamp BETWEEN %s AND %s
                       AND intersection_id = %s
                       AND camera_id       = %s
                       AND isAnomalous     = 0
                    """
                    sql_params = (
                        params['start_date'],
                        params['end_date'],
                        params['intersec_id'],
                        params['cam_id']
                    )

                print("[DB] Executing query...", end="", flush=True)
                t1 = time.time()
                cursor.execute(query, sql_params)
                print(f" done in {time.time()-t1:.2f}s")

                print("[DB] Fetching rows…", end="", flush=True)
                t2 = time.time()
                rows = []
                while batch := cursor.fetchmany(5000):
                    rows.extend(batch)
                print(f" done in {time.time()-t2:.2f}s; rows={len(rows)}")

                cols = [d[0] for d in cursor.description]

        finally:
            conn.close()

        df = pd.DataFrame(rows, columns=cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'track_id' in df.columns:
            df['hour_digit'] = df['timestamp'].dt.hour
        return df


    def countTracks(self,
                    intersection_id: int,
                    start: str,
                    end: str,
                    class_name: str = None) -> int:
        """
        Returns the number of rows in RealTrackProperties for the given intersection
        and time window, optionally filtered by class (e.g. 'car').
        """
        sql = """
        SELECT COUNT(*) 
          FROM RealTrackProperties
         WHERE timestamp BETWEEN %s AND %s
           AND intersection_id = %s
           AND isAnomalous     = 0
        """
        params = [start, end, intersection_id]
        if class_name:
            sql += " AND `class` = %s"
            params.append(class_name)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()[0] or 0


        # #
        # # excess, old code that we dont use 
        # # but keep just in case :)
        # # 
        # mycursor = mydb.cursor()
        # (iid,camid,dtstart,dtend,num_cameras)=json.loads(body)
        # stime = pd.to_datetime(dtstart)
        # ftime = pd.to_datetime(dtend)
        # query = "SELECT * FROM RealTrackProperties where timestamp between '{}' and '{}' and intersection_id = {} and camera_id = {} and isAnomalous=0".format(dtstart,dtend,iid,camid)
        # path_query = "SELECT * FROM Paths where source = {}".format(camid)
        # tracksdf = pd.read_sql(query, con=mydb)
        # pathdf = pd.read_sql(path_query, con=mydb)
        # for index, row in pathdf.iterrows():
        #     lane = row['name']
        #     print (camid, lane, len(tracksdf[tracksdf.lane==lane]))

    def fetch_latest_entry_from_table(self, table_name):
        mydb = pymysql.connect(host=self.config['host'], \
                user=self.config['user'], passwd=self.config['passwd'], \
                db=self.config['testdb'], port=int(self.config['port']))

        try:
            with mydb.cursor() as cursor:
                query = f"SELECT * FROM {table_name} ORDER BY start DESC LIMIT 1;"
                cursor.execute(query)
                result = cursor.fetchall()

                column_headers = [desc[0] for desc in cursor.description]

        finally:
            mydb.close()

        df = pd.DataFrame(result, columns=column_headers)
        df['start'] = pd.to_datetime(df['start'])
        return df
    
    
    def insert_new_entry_with_composite_id(self, offset='00:00:00.00'):
        table_name="VideoPropertiesNew"

         # Convert the offset to a timedelta
        hours, minutes, seconds = map(float, offset.split(':'))
        offset_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)


        composites = {
            31: 33,
            32: 33
        }
        # Fetch the latest entry
        df = self.fetch_latest_entry_from_table(table_name)
        print("retrieved:\n", df.to_string())

    # Subtract the offset from the timestamp, start, and end columns
        for column in ['start', 'end']:
            df[column] = pd.to_datetime(df[column]) - offset_delta

        print("fixed:\n", df.to_string())
        # Extract the camera_id from the latest entry
        camera_id = df['camera_id'].values[0]

        # Look up the associated id from the composites dictionary
        new_id = composites[camera_id]

        # Create a new entry with the new id
        new_entry = df.copy()
        new_entry['camera_id'] = new_id
        new_entry['path'] = "video/black-video.mp4"

        print("new\n", new_entry.to_string())

        # Connect to the database
        mydb = pymysql.connect(host=self.config['host'], \
                user=self.config['user'], passwd=self.config['passwd'], \
                db=self.config['testdb'], port=int(self.config['port']))

        try:
            with mydb.cursor() as cursor:
                # Prepare the insert query
                columns = ', '.join(new_entry.columns)
                placeholders = ', '.join(['%s'] * len(new_entry.columns))
                query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"

                # Execute the insert query
                # Execute the insert query
                cursor.execute(query, new_entry.values[0].tolist())

            # Commit the transaction
            mydb.commit()

        finally:
            mydb.close()

    def query_ttctable(self, id_tuples):
        """
        id_tuples is a list of tuples. Each tuple contains two IDs.
        """
        mydb = pymysql.connect(host=self.config['host'], \
                user=self.config['user'], passwd=self.config['passwd'], \
                db=self.config['testdb'], port=int(self.config['port']))

        df = pd.DataFrame()

        try:
            with mydb.cursor() as cursor:
                for id_tuple in id_tuples:
                    query = "SELECT * FROM TTCTable WHERE unique_ID1 = %s AND unique_ID2 = %s"
                    cursor.execute(query, id_tuple)
                    result = cursor.fetchall()

                    column_headers = [desc[0] for desc in cursor.description]
                    df_temp = pd.DataFrame(result, columns=column_headers)
                    df = pd.concat([df, df_temp])

        finally:
            mydb.close()

        return df
    
    def query_tracksreal(self, intersection_id: int, start_timestamp: str, end_timestamp: str, thru: bool = False):
        """
        Query the TracksReal table for data between specific timestamps for a given intersection.

        Args:
        intersection_id (int): The ID of the intersection.
        start_timestamp (str): The start timestamp in the format "YYYY-MM-DD HH:MM:SS.fff".
        end_timestamp (str): The end timestamp in the format "YYYY-MM-DD HH:MM:SS.fff".

        Returns:
        pd.DataFrame: A DataFrame containing the query results.
        """
        mydb = pymysql.connect(host=self.config['host'],
                            user=self.config['user'], passwd=self.config['passwd'],
                            db=self.config['testdb'], port=int(self.config['port']))

        try:
            with mydb.cursor() as cursor:
                # Define the SQL query to select data within the timestamp range for a specific intersection
                if thru: # used to filter for only thru traffic, when analyzing correlations between conflicts and speed
                    query = """
                    SELECT *
                    FROM TracksReal
                    WHERE intersection_id = %s AND
                        start_timestamp BETWEEN %s AND %s AND
                        (approach = 'NBT' OR approach = 'SBT' OR approach = 'EBT' OR approach = 'WBT')
                    ORDER BY start_timestamp;
                    """
                else:
                    query = """
                    SELECT *
                    FROM TracksReal
                    WHERE intersection_id = %s AND
                        start_timestamp BETWEEN %s AND %s
                    ORDER BY start_timestamp;
                    """
                # Execute the query with parameters
                cursor.execute(query, (intersection_id, start_timestamp, end_timestamp))

                # Fetch all results
                result = cursor.fetchall()

                # Extract column headers
                column_headers = [desc[0] for desc in cursor.description]

                # Convert results to a DataFrame
                df = pd.DataFrame(result, columns=column_headers)
                df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
                if 'end_timestamp' in df.columns:
                    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])

        finally:
            mydb.close()

        return df
    