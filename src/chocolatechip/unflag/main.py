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
        '006', '007', '012', '020', '023', '039', '043', '044', '045', '048', '050', '051', '053', '056', '057', '060', '063', '078', '079', '080', '092', '099', '106', '115', '118', '119', '121', '132', '140', '145', '149', '151', '156', '157', '159', '160', '161', '164', '165', '168', '171', '174', '176', '177', '180', '181', '184', '185', '186', '194', '195', '198', '199', '200', '206', '207', '210', '211', '217', '218', '219', '224', '229', '232', '235', '238', '250', '251', '260', '263', '273', '284', '288', '296', '297', '298', '299', '300', '301', '302', '303', '304', '310', '312', '313', '316', '319', '320', '321', '324', '328', '329', '334', '338', '340', '341', '347', '348', '350', '351', '356', '357', '362', '363', '364', '365', '368', '369', '371', '372', '381', '384', '392', '396', '399', '401', '403', '405', '407', '410', '411', '416', '419', '423', '424', '425', '430', '432', '434', '439', '440', '443', '447', '449', '450', '454', '455', '458', '459', '460', '461', '462', '463', '464', '465', '466', '469', '470', '471', '472', '473', '474', '475', '476', '478', '480', '481', '482', '485', '486', '488', '489', '491', '492', '495', '496', '499', '500', '501', '502', '503', '504', '505', '507', '508', '510', '511', '512', '513', '514', '515', '516', '517', '518', '520', '521', '522', '523', '524', '525', '528', '529', '531', '532', '533', '535', '536', '537', '539', '541', '547', '548', '549', '550', '551', '558', '559', '565', '566', '567', '568', '571', '573', '575', '576', '580', '581', '587', '588', '589'
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
