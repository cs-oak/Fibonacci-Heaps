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