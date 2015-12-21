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
		
import sys, unittest

class fhtest(unittest.TestCase):
    '''
    Performs a coprehensive check of the Fibonacci Heap Class
    Checks if all functions are working correctly
    '''
    def test_fullcheck(self):
        '''
        Function for the coprehensive test
        '''
        #check initialization
        fh = FibonacciHeap()
        self.assertEqual(fh.is_empty(), True)
        #check insertion and single deletion
        x = len(fh.roots)
        for i in range(1,101):
            fh.insert(i)
        self.assertEqual(len(fh.roots),x+100)
        self.assertEqual(fh.minroot.data, 1)
        temp = fh.fh_pop()
        self.assertEqual(temp, 1)
        self.assertEqual(fh.minroot.data, 2)
        #check decrease key for change in min-root and invalid search
        fh.decrease_key(67,-1)
        fh.decrease_key(69,1)
        self.assertEqual(fh.decrease_key(69,0),'')
        self.assertEqual(fh.minroot.data, -1)
        #check union
        fh2 = FibonacciHeap()
        fh2.insert(101)
        fh.union(fh2)
        #check pop
        while not fh.is_empty():
            temp = fh.fh_pop()
        self.assertEqual(temp, 101)
        self.assertEqual(fh.is_empty(), True)
        
suite = unittest.TestLoader().loadTestsFromTestCase(fhtest)
unittest.TextTestRunner(verbosity=1).run(suite)