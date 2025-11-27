import heapq

def dijkstra(graph, start_node):
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    priority_queue = [(0, start_node)]

    while priority_queue:
        print('=' * 20)
        current_distance, current_node = heapq.heappop(priority_queue)
        print(current_distance, current_node)

        if current_distance > distances[current_node]:
            print('continue')
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                print('push', distance, neighbor)
        print(distances)
    return distances


graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': [('E', 3)],
    'E': []
}

start_node = 'A'
dis = dijkstra(graph, start_node)
print(dis)