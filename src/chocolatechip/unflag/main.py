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
'326', '327', '328', '331', '333', '335', '336', '337', '338', '340', '341', '343', '345', '346', '347', '350', '351', '353', '354', '355', '356', '357', '358', '359', '360', '365', '366', '367', '368', '370', '371', '372', '373', '375', '376', '377', '379', '380', '381', '382', '383', '385', '386', '387', '388', '389', '391', '392', '393', '394', '396', '397', '398', '399', '400', '401', '403', '407', '409', '410', '411', '412', '414', '416', '417', '418', '419', '420', '421', '422', '423', '424', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '439', '440', '441', '444', '445', '446', '447', '449', '450', '451', '453', '454', '455', '456', '457', '458', '459', '460', '461', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '474', '475', '476', '477', '478', '479', '480', '481', '483', '484', '485', '486', '487', '488'
'008', '037', '046', '052', '058', '065', '069', '077', '081', '084', '086', '091', '100', '102', '108', '111', '115', '122', '123', '125', '130', '137', '139', '140', '141', '142', '143', '144', '146', '147', '149', '152', '153'
'165', '169', '175', '182', '183', '184', '185', '186', '189', '195', '196', '197', '198', '200', '201', '202', '204', '206', '207', '208', '210', '212', '215', '216', '223', '230', '234', '235', '236', '237', '239', '244', '245', '252', '253', '255', '256', '257', '258', '263', '264', '265', '266', '269', '270', '275', '278', '280', '284', '285', '286', '287', '289', '290', '291', '294', '295', '296', '297', '298', '300', '301', '302', '303', '304', '305', '310', '314', '315', '316', '317', '318', '320', '321'
'489', '490', '508', '519', '522', '557', '592', '643'
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
