import time

def get_time_str():
    time_str = time.strftime('%Y%m%d_%H%M%S')
    year = time_str[2:4]
    month = time_str[4:6]
    day = time_str[6:8]
    hour = time_str[9:11]
    minute = time_str[11:13]
    second = time_str[13:15]
    return year + month + day + '_' + hour + minute + second