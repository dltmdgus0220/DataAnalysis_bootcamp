def selection_sort(data_list):
    n = len(data_list)

    for i in range(n-1):
        target_index = i
        for j in range(i+1, n):
            if data_list[j] < data_list[target_index]:
                target_index = j
        data_list[i], data_list[target_index] = data_list[target_index], data_list[i]
        print(data_list)

def insertion_sort(data_list):
    n = len(data_list)

    for i in range(1, n):
        key = data_list[i]
        j = i - 1
        while j >= 0 and data_list[j] > key:
            data_list[j + 1] = data_list[j]
            j -= 1
        data_list[j + 1] = key
        print(data_list)

def bubble_sort(data_list):
    n = len(data_list)

    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if data_list[j] > data_list[j+1]:
                data_list[j], data_list[j+1] = data_list[j+1], data_list[j]
                swapped = True
        if not swapped:
            break
        print(data_list)

def quick_sort(data_list): # 재귀로
    if len(data_list) <= 1:
        return data_list
    
    pivot = data_list[0] # pivot 위치가 정렬된 위치
    rest_of_list = data_list[1:]

    # 피봇 기준
    # 왼쪽 : pivot보다 작은 값들
    left = [x for x in rest_of_list if x <= pivot]
    # 오른쪽 : pivot보다 큰 값들
    right = [x for x in rest_of_list if x >= pivot]
    
    return quick_sort(left) + [pivot] + quick_sort(right)

test = [5, 3, 4, 1, 2]
print('원본 :', test)
# selection_sort(test)
# insertion_sort(test)
# bubble_sort(test)
test = quick_sort(test)
print('결과 :', test)
