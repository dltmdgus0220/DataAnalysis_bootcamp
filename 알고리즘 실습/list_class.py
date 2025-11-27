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
    
