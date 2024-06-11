import pymysql
import os
from pprint import pprint
import yaml
from yaspin import yaspin


def main():
    # Read from file and save it to a variable as a string
    # simple. before you start just ls > original_file_list.txt
    with open('original_file_list.txt') as fp:
        file_content = fp.read()

    all_originals = file_content.splitlines()
    all_originals = [x for x in all_originals if x]
    all_originals = [x for x in all_originals if x.endswith('.mp4')]

    current = os.listdir()
    current = [x for x in current if x.endswith('.mp4')]


    # print(invalid_conflicts)

    tania_valid_conflicts = [
        "001", "002", "004", "006", "009", "015", "017",
        "024", "026", "027", "028", "030", "031", "032", "035", "036", "037", "038", "040"
    ]
    

    # sort tania
    # tania_valid_conflicts.sort()

    tania_valid_conflicts = [str(x) for x in tania_valid_conflicts]

    tania_valid_conflicts.sort()

    # pprint(tania_valid_conflicts)
    
    invalid_conflicts = []
    for videofile in current:
        if videofile.split('_')[0] not in tania_valid_conflicts:
            invalid_conflicts.append(videofile)
    invalid_conflicts.sort()

    pprint(invalid_conflicts)
    #
    #hackysolution
    # delete all conflicts starting from 830 onward
    # invalid_conflicts = [x for x in invalid_conflicts if int(x[:3]) < 830]
    # additional = [839, 838, 837, 836, 835, 834, 833, 832, 830, 849, 848, 847, 844, 843, 841, 840, 859, 858, 857, 855, 853, 852, 851, 850, 869, 868, 867, 866, 863, 862, 860, 879, 878, 877, 876, 873, 871, 870, 889, 888, 887, 882, 881, 880, 894, 891, 890, 907, 905, 904, 903, 902, 901, 913, 912, 911]
    # for x in current:
        # if int(x[:3]) in additional:
            # invalid_conflicts.append(str(x))
    # end of hackysolution
    #

    # 0, 2, 3, 4 are bad i deleted them. 304 is bad too. these are examples
    
    # exit(0)

    jp_cred = '/home/jpf/chocolatechip/src/chocolatechip/login.yaml'
    with open(jp_cred, 'r') as stream:
        try:
            jp_creds = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    def ttc_unflag(connection, uniqueID1, uniqueID2, cameraID):
        with connection.cursor() as cursor:
            sql = "UPDATE TTCTable SET include_flag=0 WHERE unique_ID1=%s AND unique_ID2=%s AND camera_id=%s"
            cursor.execute(sql, (uniqueID1, uniqueID2, cameraID))
            connection.commit()

    connection = pymysql.connect(host=jp_creds['host'],
                                user=jp_creds['user'],
                                password=jp_creds['passwd'],
                                db=jp_creds['testdb'],
                                port=int(jp_creds['port']),
    )

    unflagged_row_count = 0
    with yaspin(text="unflagging rows", color="yellow") as spinner:
        length = len(invalid_conflicts)
        for index, bad in enumerate(invalid_conflicts):
            spinner.text = f"unflagging rows {index+1}/{length}. current unflags {unflagged_row_count}"
            delarray = bad.strip().split('_')
            camid = delarray[1]
            unique_ID2 = delarray[-1].split('.mp4')[0]
            unique_ID1 = delarray[-2].split('.mp4')[0]
            # print(unique_ID1, unique_ID2, camid)
            ttc_unflag(connection, unique_ID1, unique_ID2, camid)
            unflagged_row_count += 1
        spinner.ok("✅")

    print(f"unflagged rows: {unflagged_row_count} out of {length} conflicts")
