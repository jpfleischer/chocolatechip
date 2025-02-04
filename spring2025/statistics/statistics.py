""" from chocolatechip.src.chocolatechip.heatmap.heatmap import heatmap_generator 
from chocolatechip.src.chocolatechip.MySQLConnector import MySQLConnector
heatmap_generator(df_type = "track",
                        mean = True,
                        intersec_id = 3287,
                        p2v = False,
                        conflict_type = "thru",
                        pedestrian_counting = False,
                        return_agg = False) """


print("Hello World")


#docker build -t statsdocker --build-arg ssh_pub_key="$(cat ~/.ssh/id_rsa.pub)" .