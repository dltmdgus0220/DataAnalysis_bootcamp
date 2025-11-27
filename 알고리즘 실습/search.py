import time

# 선형 탐색 O(n), 정렬되지 않은 무작위 데이터를 찾을 때 용이
def linear_search(data_list, target):
    for i in range(len(data_list)):
        if data_list[i] == target:
            return i
    return -1

# 이진 탐색 O(logn), 정렬되어 있는 리스트에서 빠르게 찾을 수 있음
def binary_search(data_list, target):
    start, end = 0, len(data_list)-1

    while start <= end:
        mid = (start+end)//2
        # print('mid:',mid)
        if data_list[mid] == target:
            return mid
        if data_list[mid] > target:
            end = mid-1
        else:
            start = mid+1
    return -1


target_list = [5, 2, 8, 1, 9, 4]
print(target_list)
for i in range(8, 11):
    print(linear_search(target_list, i))

sorted_list = [1, 4, 8, 9, 11, 15, 20]
print(sorted_list)
print(binary_search(sorted_list, 4))
print(binary_search(sorted_list, 11))
print(binary_search(sorted_list, 10))


big_list = []
for i in range(100000000):
    big_list.append(i)

target = 99999998

start = time.time()
print(linear_search(big_list, target))
end = time.time()
print(f'Linear Search: {end-start:.6f}')

start = time.time()
print(binary_search(big_list, target))
end = time.time()
print(f'Binary Search: {end-start:.6f}')

# 항상 이진탐색이 좋은건아님. 
# 서비스에 따라서 매번 정렬하는게 코스트가 높다면 정렬하지 않고 
# 선형탐색하는게 이득일 수 있음.