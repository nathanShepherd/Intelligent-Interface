# Simple BST Implimentation
# TODO: Add balancing, upgrade to sum tree

class Node:
    def __init__(self, left, right, datum):
        self.left = left
        self.right = right
        self.datum = datum
        self.idx = id(datum)

    def get_children(self):
        return [self.left, self.right]


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, datum):
        node = Node(None, None, datum)
        done = False
        if self.root == None:
            self.root = node
            done = True
        root = self.root
        while not done:
            if root.idx >= node.idx:
                if root.left == None:
                    root.left = node
                    done = True
                else:
                    root = root.left
            else:
                if root.right == None:
                    root.right = node
                    done = True
                else:
                    root = root.right
            
    def find(self, datum):
        index = id(datum)
        root = self.root
        while root != None:
            if index < root.idx:
                root = root.left
            elif index > root.idx:
                root = root.right
            elif index == root.idx:
                return root.datum
        return False

    def height(self):
        #Credits: Geeks4Geeks @ https://bit.ly/2FbgHkz
        height = 0
        if self.root is None: return height

        # Create a empty queue for level order traversal
        q = [self.root]
        while(True):
            # nodeCount(queue size) indicates number of nodes at current level
            nodeCount = len(q)
            if nodeCount == 0:
                return height 
     
            height += 1
 
            # Dequeue all nodes of current level and Enqueue
            # all nodes of next level
            while(nodeCount > 0):
                node = q[0]
                q.pop(0)
                if node.left is not None:
                    q.append(node.left)
                if node.right is not None:
                    q.append(node.right)
 
                nodeCount -= 1 

    def to_string(self):
        pass

def int_check(oak):
    for i in range(1000):
        num = random.randint(-i, i)
        oak.insert(num)
        if oak.find(num) != num:
            print("Error!")

def array_check(cedar):
    for j in range(1, 12):
        for i in range(100):
            arr = []
            for k in range(j):
                offset = random.randint(1,2)
                arr.append(random.randint(-k, k)+offset)
            cedar.insert(arr)
            #print(arr, cedar.find(arr))
            if cedar.find(arr) != arr:
                print("Error!")

def get_dict():
    return {'0':[random.random()],
            '1':[random.random()]}

def performance_check(num_tests=5000):
    import time;now = time.time
    #%%%%%%% Binary Search Tree %%%%%%%#
    print("Performance of BST:")
    bst = BinarySearchTree()
    start = now()
    for i in range(num_tests):
        d = str(get_dict())
        bst.insert(d)
    print("--> Insertion Runtime:", now() - start)

    del bst
    memory = []
    cherry = BinarySearchTree()
    for i in range(num_tests):
        d = str(get_dict())
        memory.append(d)
        cherry.insert(d)
        
    start = now()
    for i in range(num_tests):
        cherry.find(memory[i])
    print("--> Search Runtime:", now() - start)

    #%%%%%%% List %%%%%%%#
    print("Performance of List:")
    start = now(); arr = []
    for i in range(num_tests):
        d = str(get_dict())
        arr.append( {} )
    print("--> Insertion Runtime:", now() - start)
    
    memory = []
    for i in range(num_tests):
        d = str(get_dict())
        memory.append(d)
        
    start = now()
    for i in range(num_tests):
        memory[i]
    print("--> Search Runtime:", now() - start)

    #%%%%%%% Dictionary %%%%%%%#
    print("Performance of Dictionary:")
    start = now(); table = {}
    for i in range(num_tests):
        d = str(get_dict())
        table[d] = {}
    print("--> Insertion Runtime:", now() - start)
    
    memory = []
    lexio = {}
    for i in range(num_tests):
        d = str(get_dict())
        memory.append(d)
        lexio[d] = {}
        
    start = now()
    for i in range(num_tests):
        lexio[memory[i]]
    print("--> Search Runtime:", now() - start)
    
        

    

if __name__ == "__main__":
    import numpy as np
    import random
    
    bst = BinarySearchTree()
    bst.insert(1)
    bst.insert(10)
    bst.insert(50)
    vect = np.array([1, 2, 3, 4])
    bst.insert(vect)
    bst.insert("Spruce")

    print("1:", bst.find(1))
    print("10:", bst.find(10))
    print("50:", bst.find(50))
    print(str(vect)+":", bst.find(vect))
    print("Spruce:", bst.find("Spruce"))

    int_check(bst)
    array_check(bst)
    performance_check()


