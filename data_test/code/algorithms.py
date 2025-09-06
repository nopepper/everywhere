"""Classic algorithms and data structures implementations
"""

def fibonacci_recursive(n):
    """Recursive Fibonacci - exponential time complexity"""
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_memoized(n, memo={}):
    """Memoized Fibonacci - linear time complexity"""
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memoized(n-1, memo) + fibonacci_memoized(n-2, memo)
    return memo[n]

def quicksort(arr):
    """Quick sort implementation"""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)

def binary_search(arr, target):
    """Binary search in sorted array"""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def dfs_inorder(root):
    """In-order depth-first search"""
    if not root:
        return []
    return dfs_inorder(root.left) + [root.val] + dfs_inorder(root.right)

def dijkstra(graph, start):
    """Dijkstra's shortest path algorithm"""
    import heapq

    distances = {node: float("infinity") for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_distance, current = heapq.heappop(pq)

        if current_distance > distances[current]:
            continue

        for neighbor, weight in graph[current].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

# Example graph for testing
graph = {
    "A": {"B": 4, "C": 2},
    "B": {"C": 1, "D": 5},
    "C": {"D": 8, "E": 10},
    "D": {"E": 2},
    "E": {}
}

if __name__ == "__main__":
    print(f"Fibonacci(10): {fibonacci_memoized(10)}")
    print(f"Quicksort([3,1,4,1,5,9,2,6]): {quicksort([3,1,4,1,5,9,2,6])}")
    print(f"Dijkstra from A: {dijkstra(graph, 'A')}")

