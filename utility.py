import graphviz as gv
import heapq
import math
from collections import defaultdict

def printArr(parent, n):
    for i in range(1, n):
        print("% d - % d" % (parent[i], i))
        

class DisjointSet:
    def __init__(self, n):
        self.rank = [0] * n
        self.parent = [i for i in range(n)]
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        
        if xroot == yroot:
            return
        
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1
   

class Graph():
    def __init__(self, V):
        self.V = V
        self.graph = defaultdict(list)
    
    def fromList(self, L):
        V = len(L)
        for u, e in enumerate(L):
              for v, w in e:
                 self.addEdge(u, v, w)
    def getList(self):
      result = []
      for key, values in sorted(self.graph.items()):
        temp_set = set()
        for pair in values:
            temp_set.add((pair[0], pair[1]))
        temp_list = list(temp_set)
        result.append(temp_list)
      return result

    def addEdge(self, src, dest, weight):
        newNode = [dest, weight]
        self.graph[src].insert(0, newNode)
        newNode = [src, weight]
        self.graph[dest].insert(0, newNode)

    def PrimMST(self, start = 0):
        V = self.V 
        visited = [False] * V
        key = [float("inf")] * V
        parent = [-1] * V
        minHeap = []

        heapq.heappush(minHeap, (0, start))
        key[start] = 0

        while minHeap:
            _, u = heapq.heappop(minHeap)
            if visited[u]: continue
            visited[u] = True
            for v, weight in self.graph[u]:
                if not visited[v] and weight < key[v]:
                    key[v] = weight
                    parent[v] = u
                    heapq.heappush(minHeap, (weight, v))

        printArr(parent, V)
        return parent
    
    def kruskal_algo(self):
        n = self.V
        ds = DisjointSet(n)
        edges = [(w, u, v) for u in range(n) for v, w in self.graph[u]]
        edges.sort()
        mst = []
        result = []
        links = 0

        for w, u, v in edges:
            if ds.find(u) != ds.find(v):
                ds.union(u, v)
                result.append((u, v, w))
                mst.append((u, v))
                links += 1
            if links == n - 1:
                break
        tot_weight = 0
        for u, v, weight in result:
            tot_weight += weight
            print("%d - %d: %d" % (u, v, weight))

        print("Costo total del MST para el grafo: ", tot_weight)

        return mst
    def johnson(self):
        # Paso 1: Agregar un nuevo vértice y aristas con peso 0
        self.graph[self.V] = []
        for v in range(self.V):
            self.graph[self.V].append((v, 0))

        # Paso 2: Ejecutar el algoritmo de Bellman-Ford
        dist = self.bellmanFord(self.V)

        # Si se detecta un ciclo de peso negativo, el algoritmo termina
        if dist is None:
            return None

        # Paso 3: Calcular los nuevos pesos para las aristas del grafo original
        adjusted_weights = self.adjustWeights(dist)

        # Paso 4: Eliminar el vértice agregado y restaurar los pesos originales
        del self.graph[self.V]
        for u in self.graph:
            self.graph[u] = [(v, adjusted_weights[u][v]) for v in self.graph[u]]

        # Paso 5: Ejecutar el algoritmo de Dijkstra para cada par de vértices
        shortest_paths = []
        for u in range(self.V):
            shortest_path = self.dijkstra(u)
            shortest_paths.append(shortest_path)

        return shortest_paths

    def bellmanFord(self, src):
        dist = [float('inf')] * (self.V + 1)
        dist[src] = 0

        for _ in range(self.V):
            for u in self.graph:
                for v, weight in self.graph[u]:
                    if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight

        # Verificar si hay un ciclo de peso negativo
        for u in self.graph:
            for v, weight in self.graph[u]:
                if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                    return None  # Hay un ciclo de peso negativo

        return dist

    def adjustWeights(self, dist):
        adjusted_weights = defaultdict(dict)
        for u in self.graph:
            for v, weight in self.graph[u]:
                adjusted_weights[u][v] = weight + dist[u] - dist[v]
        return adjusted_weights

    def dijkstra(self, src):
        dist = [float('inf')] * self.V
        dist[src] = 0

        minHeap = [(0, src)]

        while minHeap:
            distance, u = heapq.heappop(minHeap)

            if distance > dist[u]:
                continue

            for v, weight in self.graph[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    heapq.heappush(minHeap, (dist[v], v))

        return dist

def prim(G):
    n = len(G)
    visited = [False]*n
    path = [-1]*n
    cost = [math.inf]*n

    cost[0] = 0
    q = [(0, 0)]
    while q:
        _, u = heapq.heappop(q)
        if visited[u]: continue
        visited[u] = True
        for v, w in G[u]:
            if not visited[v] and w < cost[v]:
                cost[v] = w
                path[v] = u
                heapq.heappush(q, (w, v))

    return path

def kruskal(G):
    n = len(G)
    ds = DisjointSet(n)
    edges = [(w, u, v) for u in range(n) for v, w in G[u]]
    edges.sort()
    mst = []
    result = []
    links = 0
    for w, u, v in edges:
        if ds.find(u) != ds.find(v):
           ds.union(u, v)
           result.append((u, v, w))
           mst.append((u, v))
           links += 1
        if links == n - 1: break
    tot_w = 0
    for u, v, w in result:
      tot_w += w
      print("%d - %d: %d" % (u, v, w))
    print("Costo total del MST para el grafo: ", tot_w)
    return mst
def prim(G, start = 0):
        n = len(G)
        visited = [False] * n
        key = [float("inf")] * n
        parent = [-1] * n
        minHeap = []

        heapq.heappush(minHeap, (0, start))
        key[start] = 0

        while minHeap:
            _, u = heapq.heappop(minHeap)
            if visited[u]: continue
            visited[u] = True
            for v, weight in G[u]:
                if not visited[v] and weight < key[v]:
                    key[v] = weight
                    parent[v] = u
                    heapq.heappush(minHeap, (weight, v))

        printArr(parent, n)
        return parent
def loadGraphAL(fn):
  with open(fn) as f:
    G = []
    for line in f:
      if line == '-\n':
        G.append([])
      else:
        nums = [int(x) for x in line.split()]
        G.append([(nums[i], nums[i+1]) for i in range(0, len(nums), 2)])
    return G

def readAdjl(fn, haslabels=False, weighted=False, sep="|"):
  with open(fn) as f:
    labels = None
    if haslabels:
      labels = f.readline().strip().split()
    L = []
    for line in f:
      if weighted:
        L.append([tuple(map(int, p.split(sep))) for p in line.strip().split()])
      else: 
        L.append(list(map(int, line.strip().split())))
  return L, labels

def adjlShow(L, labels=None, directed=False, weighted=False, path=[],
             simplepath=True,
             layout="sfdp", defaultcolor="gray"):
  g = gv.Digraph("G") if directed else gv.Graph("G")
  g.graph_attr["layout"] = layout
  g.edge_attr["color"] = defaultcolor
  g.node_attr["color"] = "orangered"
  g.node_attr["width"] = "0.1"
  g.node_attr["height"] = "0.1"
  g.node_attr["fontsize"] = "8"
  g.node_attr["fontcolor"] = "mediumslateblue"
  g.node_attr["fontname"] = "monospace"
  g.edge_attr["fontsize"] = "8"
  g.edge_attr["fontname"] = "monospace"
  n = len(L)
  for u in range(n):
    g.node(str(u), labels[u] if labels else str(u))
  added = set()
  path = enumerate(path) if simplepath else path
  for v, u in path:
    if u != -1:
      if weighted:
        for vi, w in L[u]:
          if vi == v:
            break
        g.edge(str(u), str(v), str(w), dir="forward", penwidth="2", color="orange")
      else:
        g.edge(str(u), str(v), dir="forward", penwidth="2", color="orange")
      added.add(f"{u},{v}")
      added.add(f"{v},{u}")
  if weighted:
    for u in range(n):
      for v, w in L[u]:
        if not directed and not f"{u},{v}" in added:
          added.add(f"{u},{v}")
          added.add(f"{v},{u}")
          g.edge(str(u), str(v), str(w))
        elif directed:
          g.edge(str(u), str(v), str(w))
  else:
    for u in range(n):
      for v in L[u]:
        if not directed and not f"{u},{v}" in added:
          added.add(f"{u},{v}")
          added.add(f"{v},{u}")
          g.edge(str(u), str(v))
        elif directed:
          g.edge(str(u), str(v))
  return g