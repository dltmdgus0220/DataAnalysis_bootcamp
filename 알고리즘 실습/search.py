# 선형 탐색 O(n), 정렬되지 않은 무작위 데이터를 찾을 때 용이
def linear_search(data_list, target):
    for i in range(len(data_list)):
        if data_list[i] == target:
            return i
    return -1

