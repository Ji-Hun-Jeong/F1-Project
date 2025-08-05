import re
import pandas as pd

def extract_track_from_path(file_path):
    path_parts = file_path.split('/')
    session_folder = path_parts[1]
    
    # 정규식으로 년도_트랙이름_세션 패턴 매칭
    pattern = r'(\d{4})_(.+)_(FP[1-3]|Q|R|SS|SQ)$'
    match = re.match(pattern, session_folder)
    
    if match:
        year, track_name, session = match.groups()
        return track_name
    else:
        # 예외 처리
        return session_folder

def get_real_track_name(track_name: str) -> str:
    tokens = track_name.split("_")
    real_track_name = ""
    for token in tokens:
        if token == "Grand":
            break
        real_track_name += token
    return real_track_name

def timedelta_to_seconds(td):
    if isinstance(td, str):
        td = pd.to_timedelta(td)
    else:
        print("TimeIsZero", td)
        return 0.0
    return td.total_seconds()

def get_team_name(file_name: str) -> str:
    infos = file_name.split("/")
    if len(infos) < 4:
        return None

# def get_list_by_file_name(file_name: str):
#     infos = file_name.split("/")
#     if len(infos) < 4:
#         return None
    
#     session_folder = infos[1]
#     pattern = r'(\d{4})_(.+)_(FP[1-3]|Q|R|SS|SQ)$'
#     match = re.match(pattern, session_folder)

#     team_name = infos[2]

#     return 