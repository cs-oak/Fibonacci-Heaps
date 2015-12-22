# most of this code was written by Dr. Timo Heister,
# Clemson University for a demonstration of a simple
# implementation of Dijkstra's algorithm. This code
# includes extra code with the Fibonacci Heap logic
# to compare and verify correctness. The original is
# part of a class assignment where we had to fill in
# various functions of the Priority Queue

%matplotlib inline
import matplotlib.pyplot as plt
import math
from collections import deque
from pylab import rcParams

# this is my homework assignment

class PriorityQueue():
    '''
    The arguments passed to a PriorityQueue must consist of
    objects than can be compared using <.
    Use a tuple (priority, item) if necessary.
    '''

    def __init__(self):
        self._pq = []
        self.icount = 0

    def _parent(self,n):
        return (n-1)//2

    def _leftchild(self,n):
        return 2*n + 1

    def _rightchild(self,n):
        return 2*n + 2

    def push(self, obj):
        # append at end and bubble up
        self._pq.append( obj )
        self.icount += 1
        n = len(self._pq)
        self._bubble_up(n-1)

    def _bubble_up(self, index):
        while index>0:
            cur_item = self._pq[index]
            parent_idx = self._parent(index)
            parent_item = self._pq[parent_idx]
            self.icount += 1
            
            if cur_item < parent_item:
                # swap with parent
                self._pq[parent_idx] = cur_item
                self._pq[index] = parent_item
                index = parent_idx
            else:
                break

    def pop(self):
        n = len(self._pq)
        if n==0:
            return None
        if n==1:
            return self._pq.pop()
        
        # replace with last item and sift down:
        obj = self._pq[0]
        self._pq[0] = self._pq.pop()
        self.icount += 1
        self._sift_down(0)
        return obj

    def heapify(self, items):
        """ you can assume that the PQ is empty! """
        self._pq = items
        
        # TODO: restore heap property of _pq
        pass
    
    def _sift_down(self,index):
        n = len(self._pq)
        
        while index<n:
            self.icount += 1
            cur_item = self._pq[index]
            lc = self._leftchild(index)
            if n <= lc:
                break

            # first set small child to left child:
            small_child_item = self._pq[lc]
            small_child_idx = lc
            
            # right exists and is smaller?
            rc = self._rightchild(index)
            if rc < n:
                r_item = self._pq[rc]
                if r_item < small_child_item:
                    # right child is smaller than left child:
                    small_child_item = r_item
                    small_child_idx = rc
            
            # done: we are smaller than both children:
            if cur_item <= small_child_item:
                break
            
            # swap with smallest child:
            self._pq[index] = small_child_item
            self._pq[small_child_idx] = cur_item
            
            # continue with smallest child:
            index = small_child_idx
        
    def size(self):
        return len(self._pq)
    
    def is_empty(self):
        return len(self._pq) == 0
    
# this is the Fibonacci Heap

class FhObject(object):
    '''
    This is a node object for the Fibonacci Heap class
    Data entered here must be comparable with < or >
    '''
    def __init__(self):
        self.parent = None
        self.child = None
        self.data = None
        self.mark = 0
        self.lsib = None
        self.rsib = None
        
    def add_child(self, new):
        '''
        Adds a child node to a node. Maintains min-heap property if child key is greater than parent key
        Very useful for auto-updating the Fibonacci heap for min-heap during the consolidate operation
        '''
        if new.data < self.data:
            self.__balance(self, new)
        if self.child == None:
            self.child = new
            new.parent = self
        else:
            x = self.child
            while x.rsib != None:
                x = x.rsib
            x.rsib = new
            new.lsib = x
            new.parent = self
            if new.data < self.child.data:
                self.child = new
                
    def detach(self):
        '''
        Special delete method for Fibonacci heap implementation - detaches whole branch below the node
        Does not actually delete the node or it's children - that is done by the Fibonacci Heap class
        '''
        if(self.parent != None):
            #if this node was the min-child of parent, fix parent's min-child
            if(self.parent.child == self):
                minx = None
                if(self.lsib != None):
                    x = self.lsib
                    if (minx == None):
                        minx = x
                    else:
                        if(x.data < minx.data):
                            minx = x
                if(self.rsib != None):
                    x = self.rsib
                    if (minx == None):
                        minx = x
                    else:
                        if(x.data < minx.data):
                            minx = x
                self.parent.child = minx
            self.parent = None
        #reset the siblings' connections before exiting the tree
        if(self.lsib != None):
            if(self.rsib != None):
                self.lsib.rsib = self.rsib
                self.rsib.lsib = self.lsib
                self.rsib = None
            else:
                self.lsib.rsib = None
            self.lsib = None
        if(self.rsib != None):
            self.rsib.lsib = None
            self.rsib = None
            
    def find_rank(self):
        ''' Finds the rank of a given node in the heap based on it's position '''
        x = self
        rank = 0
        while x.child != None:
            if x.child != None:
                rank += 1
                x = x.child
            temp = x
            if x.lsib != None:
                while x.lsib != None:
                    x = x.lsib
                    if x.child != None:
                        break
                if x.child != None:
                    continue
            x = temp
            if x.rsib != None:
                while x.rsib != None:
                    x = x.rsib
                    if x.child != None:
                        break
                if x.child != None:
                    continue
        return rank
    
    def __balance(self, parent, child):
        ''' Private method - responsible for maintaining balance of the heap '''
        if child.data < parent.data:
            temp = child.data
            child.data = parent.data
            parent.data = temp
        if(parent.parent != None):
            if parent.data < parent.parent.child.data:
                parent.parent.child = parent
            self.__balance(parent.parent, parent)
        if(child.child != None):
            if child.data > child.child.data:
                self.__balance(child, child.child)
            else:
                x = child.child
                while x.rsib != None:
                    x = x.rsib
                    if x.data < child.data:
                        child.child = x
                        self.__balance(child,x)
                temp = x
                while x.lsib != None:
                    x = x.lsib
                    if x.data < child.data:
                        child.child = x
                        self.__balance(child,x)
            
                
            
        
class FibonacciHeap(object):
    '''
    This is the class for a Fibonacci Heap
    '''
    def __init__(self):
        self.roots = [] #this list will only hold roots
        self.nodes = [] #this list will hold all nodes in the F-heap, useful for keeping find operation simple
        self.minroot = None
        self. icount = 0
    
    def find_min(self):
        '''
        Finds the root with minimum value
        Used in other functions to keep the pointer to the smallest value up to date
        '''
        index = 0
        for i in range (0, len(self.roots)):
            if self.minroot == None:
                self.minroot = self.roots[i]
                index = i
            else:
                if self.roots[i].data < self.minroot.data:
                    self.minroot = self.roots[i]
                    index = i
        return index
    
    def insert(self, item):
        '''
        Inserts new data into the sequence
        Updates the minimum root automatically
        Inserted data must be comparable with < or >
        Tuples of the form (priority, key) accepted
        '''
        
        #algorithm is 'lazy' - we just insert the new item for now, we do not
        #bother updating the heap for balance like we do in classic min-heap
        
        new = FhObject()
        new.data = item
        self.roots.append(new)
        self.icount += 1
        self.nodes.append(new)
        if self.minroot == None:
            self.minroot = new
        else:
            if new.data < self.minroot.data:
                self.minroot = new
        
    def union(self, obj):
        '''
        Concatenates root lists of two Fibonacci Heaps
        Updates minimum root automatically
        '''
        self.roots += obj.roots
        self.nodes += obj.nodes
        self.icount += 1
        self.find_min()
        
    def decrease_key(self, item, value):
        '''
        Decreases the value of a given node in the Fibonacci Heap
        If multiple keys with the same value exists, it will perform this operation on the first node with the given value
        Operation is return error message if key is not found
        '''
        
        #again, we take the 'lazy' approach - just re-attach the decreased nodes to root list
        #we keep the order of the heap in check by marking nodes who've had more than 1 child
        #detached due to this function, and we recursively detach marked parents
        
        x = self.__find(item)
        if x == None:
            print "Error: Could not find a node with the given value"
            return ''
        x.data = value
        if x not in self.roots:
            if x.parent.mark == 0:
                x.parent.mark = 1
                x.detach()
                self.roots.append(x)
            else:
                y = x.parent
                x.detach()
                self.roots.append(x)
                while y.mark != 0:
                    z = y.parent
                    if z == None:
                        break
                    y.mark = 0
                    y.detach()
                    self.roots.append(y)
                    y = z
        self.icount += 1
        self.find_min()
        
    def fh_pop(self):
        '''
        Pops the minimum element from the Fibonacci Heap
        Functions similar to pop in a classic Priority Queue
        '''
        
        #first extarct the min-root and append all it's children to the root list
        
        if self.is_empty():
            print "Underflow",
            return ''
        else:
            x = self.minroot
            y = x.data
            self.roots.remove(self.minroot)
            self.nodes.remove(self.minroot)
            self.icount += 1
            if x.child:
                x = x.child
                self.roots.append(x)
                temp = x
                while x.lsib != None:
                    x = x.lsib
                    self.roots.append(x)
                    self.icount += 1
                x = temp
                while x.rsib != None:
                    x = x.rsib
                    self.roots.append(x)
                    self.icount += 1
        for i in range(0,len(self.roots)):
            if self.roots[i].parent != None:
                self.roots[i].detach()
                
        #consolidate - perform all the steps which the 'lazy' algorithm has been postponing so far
        
        #cur - current node in the loop, checks nodes for same rank, if there is a match, attaches itself
        #to it, and starts over, searching for root nodes matching it's new rank, once this is exhausted,
        #it moves on to the next node
        
        #prev - basically, to check for the algorithm's termination, we check if the root list has nodes with all 
        #unique ranks : this is the only condition which will not reset this variable and cause the loop to break
        
        #We cannot directly loop this using the bound len(roots) because that bound changes during the loop
        
        if self.is_empty():
            self.minroot = None
            return y
        cur = self.roots[0]
        i = self.roots.index(cur)
        prev = 0
        while True:
            if prev == len(self.roots):
                break
            i = (i+1)%(len(self.roots))
            if i == self.roots.index(cur):
                cur = self.roots[(i+1)%(len(self.roots))]
                i = self.roots.index(cur)
                prev += 1
                continue
            if cur.find_rank() == self.roots[i].find_rank():
                cur.add_child(self.roots[i])
                self.icount += 1
                self.roots.remove(self.roots[i])
                i = self.roots.index(cur)
                prev = 0
                continue
        self.minroot = None
        self.find_min()
        return y
                
    def is_empty(self):
        '''
        Returns True if the Fibonacci Heap is empty
        '''
        if len(self.roots) == 0:
            return True
        else:
            return False
    
    def __find(self, item):
        '''
        Private method - find an item in Fibonacci heap which matches a given key
        Used only by decrease key method 
        '''
        for i in range (0, len(self.nodes)):
            if self.nodes[i].data == item:
                return self.nodes[i]
        return None
		
# Dr. Heister's code begins here

class Graph(object):
    '''Represents a graph'''

    def __init__(self, vertices, edges):
        '''A Graph is defined by its set of vertices
           and its set of edges.'''
        self.V = set(vertices) # The set of vertices
        self.E = set([])       # The set of edges
        self.Adj = {}          # A dictionary that will hold the list
                               # of adjacent vertices for each vertex.
        self.Vcoord = {}       # A dictionary that can hold coordinates
                               # for the vertices.
        self.edge_labels = {}

        self.add_edges(edges)  # Note the call to add_edges will also
                               # update the Adj dictionary
        print '(Initializing a graph with %d vertices and %d edges)' % (len(self.V),len(self.E))


    def add_vertices(self,vertex_list):
        ''' This method will add the vertices in the vertex_list
            to the set of vertices for this graph. Since V is a
            set, duplicate vertices will not be added to V. '''
        for v in vertex_list:
            self.V.add(v)
        self.build_Adj()


    def add_edges(self,edge_list):
        ''' This method will add a list of edges to the graph
            It will insure that the vertices of each edge are
            included in the set of vertices (and not duplicated).
            It will also insure that edges are added to the
            list of edges and not duplicated. '''
        for s,t in edge_list:
            if (s,t) not in self.E and (t,s) not in self.E:
                self.V.add(s)
                self.V.add(t)
                self.E.add((s,t))
        self.build_Adj()


    def build_Adj(self):
        self.Adj = {}
        for v in self.V:
            self.Adj[v] = []
        for e in self.E:
            s,t = e
            self.Adj[s].append(t)
            self.Adj[t].append(s)


    def degree_of(self,vertex):
        if vertex in self.V:
            return len(self.Adj[vertex])
        else:
            return None


    def get_a_vertex(self):
        if 0 < len(self.V):
            v = self.V.pop()
            self.V.add(v)
            return v
        else:
            return None


    def plot(self):
        nV = len(self.V)
        if len(self.Vcoord) != nV:
            # Coordinates have not been specified for every vertex
            dTheta = 2*math.pi/nV
            k = 0
            for v in self.V:
                self.Vcoord[v] = (10*math.cos(math.pi/2-k*dTheta),10*math.sin(math.pi/2-k*dTheta))
                k += 1
        px = []
        py = []
        for v in self.V:
            px.append(self.Vcoord[v][0])
            py.append(self.Vcoord[v][1])
        plt.plot(px,py,'bo',hold=True)
        for vertex in self.V:
            p = self.Vcoord[vertex]
            pq = max(0.1,math.sqrt(p[0]**2 + p[1]**2))
            rx = p[0]/pq
            ry = p[1]/pq
            plt.text(p[0]+0.2*rx, p[1]+0.2*ry, str(vertex))
        for s,t in self.E:
            plt.plot([self.Vcoord[s][0], self.Vcoord[t][0]],
                     [self.Vcoord[s][1], self.Vcoord[t][1]],
                     'b',hold=True)
            if (s,t) in self.edge_labels:
                label = self.edge_labels[(s,t)]
                plt.text((self.Vcoord[s][0]+self.Vcoord[t][0])/2-0.1,
                         (self.Vcoord[s][1]+self.Vcoord[t][1])/2-0.1, label)
        plt.xlim(min(px)-1.0,max(px)+1.1)
        plt.ylim(min(py)-1.0,max(py)+1.1)

    def get_a_component_spanning_tree(self, root):
        # This routine uses a breadth-first search
        # to obtain a tree that spans the component
        # containing 
        spanning_tree = []
        visited = {}
        for v in self.V:
            visited[v] = False
        Q = deque()
        visited[root] = True
        Q.append(root)
        while 0 < len(Q):
            v = Q.popleft()
            for u in self.Adj[v]:
                if not visited[u]:
                    visited[u] = True
                    Q.append(u)
                    spanning_tree.append((v,u))
        return spanning_tree


    def is_connected(self):
        # If the graph is connected then if the tree
        # returned by get_a_component_spanning_tree has
        # nV-1 edges - that is, it spans the graph.
        root = self.get_a_vertex()
        tree = self.get_a_component_spanning_tree(root)
        if len(tree) == len(self.V)-1:
            return True
        else:
            return False
			
class Network(Graph):    
    def __init__(self, vertices, edge_weights):
        ''' Initialize the network with a list of vertices
        and weights (a dictionary with keys (E1, E2) and values are the weights)'''

        edges = []
        for e1,e2 in edge_weights:
            edges.append((e1,e2))
        
        Graph.__init__(self, vertices, edges)
        self.weights = {}
        for e1,e2 in edge_weights:
            weight = edge_weights[(e1,e2)]
            self.weights[(e1,e2)] = weight
            self.weights[(e2,e1)] = weight
        self.edge_labels = self.weights

# original dijkstra code plus Fibonacci Heap code

def dijkstra(network, source):
    dist = {source:0}
    prev = {}
    done = {}
    pq = PriorityQueue()
    pq.push((0,source))
    
    while not pq.is_empty():
        dist_u, u = pq.pop()
        if u in done:
            continue
        done[u] = True
        
        for v in network.Adj[u]:
            new_dist_to_v = dist_u + network.weights[(u,v)]
            if not v in dist or dist[v]>new_dist_to_v:
                dist[v] = new_dist_to_v
                prev[v] = u
                pq.push((new_dist_to_v, v))
                
    print pq.icount           
    return dist, prev
            
print "For Normal Priority Queue:"   
dist, prev = dijkstra(G1,'A')           
print "distance:", dist
print "prev:", prev

print ''

def dijkstra1(network, source):
    dist = {source:0}
    prev = {}
    done = {}
    pq = FibonacciHeap()
    pq.insert((0,source))
    
    while not pq.is_empty():
        dist_u, u = pq.fh_pop()
        if u in done:
            continue
        done[u] = True
        
        for v in network.Adj[u]:
            new_dist_to_v = dist_u + network.weights[(u,v)]
            if not v in dist or dist[v]>new_dist_to_v:
                dist[v] = new_dist_to_v
                prev[v] = u
                pq.insert((new_dist_to_v, v))
                
    print pq.icount            
    return dist, prev
            
print "For Fibonacci Heap:"    
dist, prev = dijkstra1(G1,'A')           
print "distance:", dist
print "prev:", prev

