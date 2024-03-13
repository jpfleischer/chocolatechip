import numpy as np
import pandas as pd
import pymysql
import json
import yaml
import os

from datetime import datetime, timedelta

class MySQLConnector:

    
    def __init__(self):
        # look in the same dir as mysqlconnector for config file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'login.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.config = config
        

    # def handleRequest(self, body, num_cameras, dual_cam_id={}):
    def handleRequest(self, params, df_type: str):
        """
        params is a dict with attributes
        start date, end date, intersec_id, camera_id
        df_type is a string that can be "conflict" or "track"
        """
        mydb = pymysql.connect(host=self.config['host'], \
                user=self.config['user'], passwd=self.config['passwd'], \
                db=self.config['testdb'], port=int(self.config['port']))
        
        try:
            with mydb.cursor() as cursor:
                # query = "SELECT * FROM RealTrackProperties WHERE timestamp BETWEEN %s AND %s AND intersection_id = %s AND camera_id = %s AND isAnomalous = 0"
                # path_query = "SELECT * FROM Paths WHERE source = %s"
                if df_type == "conflict":

                    query = "SELECT * FROM TTCTable WHERE " \
                        "intersection_id = %s AND p2v = %s " \
                        "AND include_flag=1 AND timestamp BETWEEN %s AND %s;"
                    cursor.execute(query, (params['intersec_id'], params['p2v'], params['start_date'], params['end_date']))
                elif df_type == "track":
                    query = "SELECT * FROM RealTrackProperties WHERE " \
                        "timestamp BETWEEN %s AND %s AND intersection_id = %s "\
                        "AND camera_id = %s AND isAnomalous = 0"
                    cursor.execute(query, (params['start_date'], params['end_date'], params['intersec_id'], params['cam_id']))

                
                result = cursor.fetchall()

                column_headers = [desc[0] for desc in cursor.description]
                # cursor.execute(path_query, (camid,))
                # path_result = cursor.fetchall()

        finally:
            mydb.close()
        # print('hey')
        # print(column_headers)

        df = pd.DataFrame(result, columns=column_headers)
        # Convert the 'timestamp' column to datetime type
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract hour digit and create a new column
        df['hour_digit'] = df['timestamp'].dt.hour
        return df


        #
        # excess, old code that we dont use 
        # but keep just in case :)
        # 
        mycursor = mydb.cursor()
        (iid,camid,dtstart,dtend,num_cameras)=json.loads(body)
        stime = pd.to_datetime(dtstart)
        ftime = pd.to_datetime(dtend)
        query = "SELECT * FROM RealTrackProperties where timestamp between '{}' and '{}' and intersection_id = {} and camera_id = {} and isAnomalous=0".format(dtstart,dtend,iid,camid)
        path_query = "SELECT * FROM Paths where source = {}".format(camid)
        tracksdf = pd.read_sql(query, con=mydb)
        pathdf = pd.read_sql(path_query, con=mydb)
        for index, row in pathdf.iterrows():
            lane = row['name']
            print (camid, lane, len(tracksdf[tracksdf.lane==lane]))

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