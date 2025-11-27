class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        # self.prev = None

class LinkedList:
    def __init__(self):
        self.head = None
        
    # 빈 리스트인지
    def isEmpty(self):
        return self.head is None
    
    # 전체 길이
    def length(self):
        count = 0
        cur = self.head
        while cur:
            count += 1
            cur = cur.next
        return count
    
    # 삽입
    def insert(self, index, data):
        new_node = Node(data)

        if index == 0:
            new_node.next = self.head
            self.head = new_node
            return
        prev = self.head
        for _ in range(index-1):
            if prev is None:
                raise IndexError('Index Out of Range')
            prev = prev.next
        new_node.next = prev.next
        prev.next = new_node
            
    # 제거
    def delete(self, index):
        if self.head is None: # 비어있는지
            raise IndexError('Index Out of Range')
        if index >= self.length(): # 인덱스 범위 벗어나는지
            raise IndexError('Index Out of Range')
        
        if index == 0:
            self.head = self.head.next
            return

        prev = self.head
        for _ in range(index-1):
            prev = prev.next
        prev.next = prev.next.next

    # 값 가져오기
    def get(self, index):
        if self.head is None: # 비어있는지
            raise IndexError('Index Out of Range')
        if index >= self.length(): # 인덱스 범위 벗어나는지
            raise IndexError('Index Out of Range')
        
        cur = self.head
        for _ in range(index):
            cur = cur.next
        return cur.data
    
