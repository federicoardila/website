from sets import Set
from matplotlib import pyplot as plt
from matplotlib import animation
import copy
import numpy as np

#grid convention - if grid height = 1 then there are 2 rows (the 0th row and the 1st row)

#join irreducible shape consists of string elements describing what your young tableaux looks like, for example snake shape ['u','r','d'] is the young tableaux
#which is made from 1 unit step up, 1 unit step right, and followed by 1 unit step down (note that the first box is assumed)

#note that the 
#first vertical describes says the integer entry in the first box of the young tableaux. because these will be join irreducible elements if the first element
#is join irreducible, then the first entry uniquely determines the rest of the integer entries


""" JOIN IRREDUCIBLE """

class JoinIrreducible:

    join_irreducible_collection = Set()
    arm_length = 0
    grid_height = 0

    def __init__(self,snake_shape,first_vertical):

        self.snake_shape = snake_shape
        self.first_vertical = first_vertical

        self.parents = []

        """
        self.related_irreducibles = Set()
        self.inconsisten_pairs = []
        """

    def __hash__(self):
        return hash((self.snake_shape,self.first_vertical))

    def __eq__(self,other):
        return ((self.snake_shape == other.snake_shape) and (self.first_vertical == other.first_vertical))
    
    @classmethod
    def set_arm_length(cls,n):
        cls.arm_length = n

    @classmethod
    def set_grid_height(cls,n):
        cls.grid_height = n
        
    @classmethod
    def add_join_irreducible(cls,join_irreducible):
        cls.join_irreducible_collection.add(join_irreducible)




""" PIP """


class PIP:

    
    def __init__(self,join_irreducibles):

        #turns our set of join irreducibles into a list which we will use to index the matrix
        self.class_index = list(join_irreducibles)

        #creates a (# join irreducibles)x(# join irreducibles) matrix with all entries 0
        pip_matrix = [[0 for x in join_irreducibles] for x in join_irreducibles]
        pip_matrix = self.generate_inconsistent_pairs(pip_matrix)
        
        for row, join_irreducible_less_than in enumerate(self.class_index):
            for column, join_irreducible_greater_than in enumerate(self.class_index):
                if pip_matrix[row][column] == 0:
                    if self.is_less_than(join_irreducible_less_than,join_irreducible_greater_than):
                
                        pip_matrix[row][column] = 1

        self.pip = pip_matrix

        self.altered_pip = copy.deepcopy(pip_matrix)


    def is_less_than(self,irreducible_one,irreducible_two):

        smallest_length = min(len(irreducible_one.snake_shape), len(irreducible_two.snake_shape))
        is_less_than = False
        equiv_boxes = True

        if len(irreducible_one.snake_shape) <= len(irreducible_two.snake_shape):
            if irreducible_one.first_vertical >= irreducible_two.first_vertical:
                for box in range(smallest_length):

                    if irreducible_one.snake_shape[box] != irreducible_two.snake_shape[box]:
                        equiv_boxes = False
                        break
                if equiv_boxes == True:
                    is_less_than = True

        return is_less_than


    def generate_inconsistent_pairs(self,pip_matrix):

        #iterate over the upper triangle of pip_matrix
        for row in range(len(pip_matrix)):
            for col in range(row,len(pip_matrix)):

                is_consistent = True

                #gets the smaller length of the two elements
                smallest_length = min(len(self.class_index[row].snake_shape), len(self.class_index[col].snake_shape))

                #compare whether one of the tableaux is contained in the other
                for box in range(smallest_length):
                    if self.class_index[row].snake_shape[box] != self.class_index[col].snake_shape[box]:
                        is_consistent = False
                        break

                #if not then label the pair inconsistent
                if is_consistent == False:

                    pip_matrix[row][col] = -1
                    pip_matrix[col][row] = -1

        return pip_matrix

    #cerates the pip corresponding to the cubical complex rooted at a new state
    def reroot(self,order_ideal):

        altered_set = Set()
    
        for join_irreducible in order_ideal:

            index_join_irreducible = self.class_index.index(join_irreducible)

            for col_index,row in enumerate(self.pip):

                #if the elements being examined are being compared for the first time and are both in the order ideal
                #swap the order. i.e. if a > b now b > a.
                if (self.class_index[col_index] in order_ideal) and ((index_join_irreducible,col_index) not in altered_set):

                    self.altered_pip[index_join_irreducible][col_index] = self.pip[col_index][index_join_irreducible]
                    self.altered_pip[col_index][index_join_irreducible] = self.pip[index_join_irreducible][col_index]

                    altered_set.add((col_index,index_join_irreducible))

                if self.class_index[col_index] not in order_ideal:

                    #if the col index is comparable to the join irreducible element in our order ideal
                    #but is not itself in the order ideal, make inconsistent
                    if self.pip[index_join_irreducible][col_index] == 1:

                        self.altered_pip[index_join_irreducible][col_index] = -1
                        self.altered_pip[col_index][index_join_irreducible] = -1

                    #if the col index is inconsistent to the join irreducible element in our order ideal
                    #make the element outside our order ideal, the col index, comparable and larger
                    if self.pip[index_join_irreducible][col_index] == -1:

                        self.altered_pip[index_join_irreducible][col_index] = 1
                        self.altered_pip[col_index][index_join_irreducible] = 0

                    altered_set.add((col_index,index_join_irreducible))



#parent pip element found by decrementing every integer by 1 in the young tableaux
#this parent maintains the exact same shape
def same_shape_parent(join_irreducible):


    if join_irreducible.first_vertical != 0:
        same_shape_parent = JoinIrreducible(join_irreducible.snake_shape,join_irreducible.first_vertical - 1)
        join_irreducible.parents.append(same_shape_parent)




#parent pip element found by appending vertical step in the most tucked in position
#this parent has the same shape but with an additional box appended to the right of the last box
def right_parent(join_irreducible):


    #sum of number of right steps and the number of boxes with the first entry
    #determines whether we can add an additional box or not in the young tableaux
    if join_irreducible.snake_shape.count('E') + join_irreducible.first_vertical + (len(join_irreducible.snake_shape)+ 1) < (JoinIrreducible.arm_length - 1):
        
        old_shape = tuple(i for i in join_irreducible.snake_shape)
        right_parent = JoinIrreducible(old_shape + ('E',),join_irreducible.first_vertical)
        join_irreducible.parents.append(right_parent)


#parent pip element found by adding a box above or below the last box if legitimate
def vertical_parent(join_irreducible):

    #sum of number of right steps and the number of boxes with the first entry is one less than the arm length (so that we are allowed to add one more box)
    #determines whether we can add an additional box or not in the young tableaux
    if join_irreducible.snake_shape.count('E') + join_irreducible.first_vertical + (len(join_irreducible.snake_shape)+1) < JoinIrreducible.arm_length:

        last_vertical = ' '
        right_count = 0

        #from the end counts the number of right steps until the
        #first not right step it comes across in the young tableaux
        for item in reversed(join_irreducible.snake_shape):
            if item == 'E':
                right_count += 1
            
            if (item == 'N') or (item == 'S'):
                last_vertical = item
                break
            
        parity = right_count % 2
        old_shape = tuple(i for i in join_irreducible.snake_shape)

        #if number of consecutive last right boxes has odd parity, append a box in the opposite direction of the last vertical box
        #if even parity, append a box in the same direction as the last vertical box
        if (last_vertical == 'N' and parity == 0) or (last_vertical == ' ' and parity == 0) or (last_vertical == 'S' and parity == 1):
            if join_irreducible.snake_shape.count('N') - join_irreducible.snake_shape.count('S') < JoinIrreducible.grid_height - 1:
                vertical_parent = JoinIrreducible(old_shape + ('N',),join_irreducible.first_vertical)
                join_irreducible.parents.append(vertical_parent)

        if (last_vertical == 'S' and parity == 0) or (last_vertical == 'N' and parity == 1):
            #if parity == 1:    WAS HERE WHILE THERE WAS AN ISSUE, TESTING COMMENTING OUT
            if join_irreducible.snake_shape.count('N') - join_irreducible.snake_shape.count('S') != 0:
                vertical_parent = JoinIrreducible(old_shape + ('S',),join_irreducible.first_vertical)
                join_irreducible.parents.append(vertical_parent)
                

def generate_ancestry(join_irreducible):

    #keeps track of previously made join irreducibles
    JoinIrreducible.add_join_irreducible(join_irreducible)

    #creates all possible parent elements of the join irreducible
    same_shape_parent(join_irreducible)
    right_parent(join_irreducible)
    vertical_parent(join_irreducible)

    #runs through the list of parents of the join irreducible finds their parents
    for parent in join_irreducible.parents:
        
        if parent not in JoinIrreducible.join_irreducible_collection:
            generate_ancestry(parent)

    
    """ideally I would like to make it so that we generated every relation here but for whatever reason
    #what I have commented out in these few lines isn't thorough enough
        
    #join_irreducible.related_irreducibles = join_irreducible.related_irreducibles | Set(join_irreducible.parents)

    #for parent in join_irreducible.parents:
    #    join_irreducible.related_irreducibles = join_irreducible.related_irreducibles | parent.related_irreducibles
    """
    

#finds the order of moves from ideal1 to ideal2 sequenced by argest minimal antichains
def normal_cube_path(pip,ideal1,ideal2):

    pip.reroot(ideal1)
    symmetric_difference = ideal1 ^ ideal2

    #list of join irreducibles ordered by available moves
    antichain_list = []    

    minimals,non_minimals = minimal_remaining_antichain(pip,symmetric_difference)
    antichain_list.append(minimals)

    while(non_minimals):
        minimals,non_minimals = minimal_remaining_antichain(pip,non_minimals)
        antichain_list.append(minimals)

    return antichain_list
    

#breaks up a set into its largest minimal component (via PIP) and everything else
def minimal_remaining_antichain(pip,remainder):

    minimals = Set()
    non_minimals = Set()

    #compares an element in remainder will other elements in remainder
    for is_parent in remainder: 
        minimal = True
        for is_child in remainder:

            if is_parent != is_child:
                
                #if parent is greater than child (ie PIP_(child,parent) == 1
                #add parent to non minimal antichain
                is_parent_index = pip.class_index.index(is_parent)
                is_child_index = pip.class_index.index(is_child)
                if pip.altered_pip[is_child_index][is_parent_index] == 1:

                    non_minimals.add(is_parent)
                    break

        minimals = remainder ^ non_minimals

    return minimals,non_minimals
        

def compute_state_sequence(pip,order_ideal,antichain_list):
    
    #updates the sequence of order ideals to match the sequence of antichains
    order_ideal_sequence = [(order_ideal)]
    for previous_ideal,antichain in enumerate(antichain_list):
        next_order_ideal = copy.deepcopy(order_ideal_sequence[previous_ideal])
        for join_irreducible in antichain:

            #if the join irreducible is contained in the order ideal and the antichain
            #remove the join irreducible (undo the move)
            if join_irreducible in order_ideal_sequence[previous_ideal]:
                next_order_ideal.remove(join_irreducible)

            #otherwise add the join irreducible and apply the movement
            else:
                next_order_ideal.add(join_irreducible)
        
        order_ideal_sequence.append(next_order_ideal)

    state_sequence = []

    for ideal in order_ideal_sequence:

        #for the sake of making sure this is an order ideal
        ideal_closure = compute_order_ideal(pip,ideal)
            
        max_antichain = order_ideal_to_antichain(pip,ideal_closure)
        tableaux = antichain_to_tableaux(max_antichain)
        next_state = tableaux_to_state(tableaux)
        state_sequence.append(next_state)

        


    return state_sequence



""" USER INPUT FUNCTION """

#tests whether the user input is valid, i.e. fits inside the grid and is non-self intersecting
def enter_valid_state(grid_height):

    valid_state = False
    
    while(not valid_state):

        valid_state = True
        state_attempt = list(raw_input("Enter a valid state:  "))
        token_list = ['u','r','d']
        previous_token = ' '

        up_count = 0
        
        for token in state_attempt:
            if token not in token_list:
                valid_state = False
                print("Undefined step unit, please use 'u', 'd', or 'r'.")
                break
            if token == 'u':
                up_count += 1
                if int(up_count) > int(grid_height):
                    valid_state = False
                    print("State exceeds the given grid height")
                    break
            if token == 'd':
                up_count -= 1
                if up_count < 0:
                    valid_state = False
                    print("The given state falls below the grid. Please enter a valid state.")
                    break
    
            if (previous_token == 'u' and token == 'd') or (previous_token == 'd' and token == 'u'):
                valid_state = False
                print("The given state has an overlapping step sequence u,d or d,u.")
                break
            previous_token = token
            
    return state_attempt



""" TRANSITION FUNCTIONS ON SET TYPES (STATES/TABLEAUX/ORDERIDEALS/ANTICHAINS """


#takes in a list of unit strings 'u', 'r', or 'd' and turns it into a tableux with units 'N', 'E', and 'S' respectively (NOT JOIN IRREDUCIBLE)
def state_to_tableaux(state):

    tableaux_direction = ()
    tableaux_values = ()
    
    right_count = 0
    last_direction = ' '
    for step in state:
        if step == 'r':
            right_count += 1
        if step == 'u':
            tableaux_values += (right_count,)
            if last_direction == ('u' or ' '):
                tableaux_direction += ('N',)
            if last_direction == 'd':
                tableaux_direction += ('E',)

            last_direction = 'u'
            
        if step == 'd':
            tableaux_values += (right_count,)
            if last_direction == 'u':
                tableaux_direction += ('E',)
            if last_direction == 'd':
                tableaux_direction += ('S',)

            last_direction = 'd'

    return tableaux_direction,tableaux_values



#decomposes young tableaux of the state into a set of join irreducible young tableaux
def decompose_tableaux(tableaux):

    decomposition_list = []

    #tableaux[0] corresponds to box directions
    #tableaux[1] corresponds to box values
    for index,box in enumerate(tableaux[1]):
        snake_shape = tableaux[0][0:index]
        east_count = snake_shape.count('E')
        first_vertical = box - east_count

        decomposition_list.append(JoinIrreducible(snake_shape,first_vertical))

    return decomposition_list


#takes a PIP matrix and a list of join irreducible elements (decomposition list)
#and creates a set which contains all children of every element in your list
def compute_order_ideal(pip,decomposition_list):

    order_ideal = Set()

    #runs through the list of join irreducibles from the decomposed entered state
    for decomposed_irreducible in decomposition_list:

        #grabs the index in the matrix corresponding to the join irreducible (an identical tableaux)
        index_decomposed = pip.class_index.index(decomposed_irreducible)

        #runs through matrix and checks which elements are beneath the current decomposed join irreducible being checked
        #then adds to the set of order ideals
        for row_index,row in enumerate(pip.pip):
            if row[index_decomposed] == 1:
                order_ideal.add(pip.class_index[row_index])

    return order_ideal


#order ideal is a SET of join irreducible elements that also includes any join irreducibles that have parents in the set
#given an order ideal (works for just sets of join irreducibles) determines the maximal antichain that can be made
def order_ideal_to_antichain(pip,order_ideal):

    maximal_antichain = copy.deepcopy(order_ideal)

    if len(order_ideal) != 0:
        for join_irreducible in order_ideal:

            #creates a list of elements that are children of 
            index_join_irreducible = pip.class_index.index(join_irreducible)

            for row_index,row in enumerate(pip.altered_pip):

                if pip.class_index[row_index] != join_irreducible and pip.class_index[row_index] in maximal_antichain:
                    if pip.pip[row_index][index_join_irreducible] == 1:
                        maximal_antichain.remove(pip.class_index[row_index])

    else:
        maximal_antichain = set()
    
    return maximal_antichain

#stacks join irreducible elements in the antichain together and chooses the smallest integer in each box of the tableaux
def antichain_to_tableaux(antichain):

    tableaux_direction = []
    tableaux_values = []

    if len(antichain) != 0:
        for join_irreducible in antichain:

            if len(tableaux_direction) < len(join_irreducible.snake_shape):
                tableaux_direction = join_irreducible.snake_shape

            last_tableaux_value = join_irreducible.first_vertical + join_irreducible.snake_shape.count('E')

            #compute the integer tableaux of join irreducible
            current_entry = join_irreducible.first_vertical
            join_irreducible_integer_tableaux = [current_entry]
            for box in join_irreducible.snake_shape:
                if box == 'E':
                    current_entry += 1
                join_irreducible_integer_tableaux.append(current_entry)
        
            if len(tableaux_values) < len(join_irreducible.snake_shape)+ 1:
                tableaux_values.extend(join_irreducible_integer_tableaux[len(tableaux_values):len(join_irreducible_integer_tableaux)])


            for index in range(min(len(tableaux_values),len(join_irreducible_integer_tableaux))):
                if join_irreducible_integer_tableaux[index] < tableaux_values[index]:
                    tableaux_values[index] = join_irreducible_integer_tableaux[index]


    return tuple(tableaux_direction),tuple(tableaux_values)


def tableaux_to_state(tableaux):

    #tableaux[0] is the direction of the blocks
    #tableaux[1] are the integers in the respective blocks
    state = []

    #if the tableaux is empty, i.e. no join irreducibles characterize the state then the state is all right steps
    if len(tableaux[1]) == 0:
        for i in range(JoinIrreducible.arm_length):
            state.append('r')

    else:
        first_vertical = tableaux[1][0]
        right_count = 0
        for index in range(first_vertical):
            state.append('r')
            right_count += 1
        state.append('u')
        previous_vertical = 'u'
        for index,block_direction in enumerate(tableaux[0]):

            for difference in range(tableaux[1][index+1] - state.count('r')):
                state.append('r')

            if block_direction == 'N':
                state.append('u')

            if block_direction == 'S':
                state.append('d')

            if block_direction == 'E':
                if previous_vertical == 'u':
                    state.append('d')
                    previous_vertical = 'd'
                else:
                    state.append('u')
                    previous_vertical = 'u'
        for difference in range(JoinIrreducible.arm_length - len(state)):
            state.append('r')

    return state

#reorganizes the vertices of the states into a list carrying the x coordinates and a list carrying the y coordinates
def states_to_graphs(state_sequence):

    state_x_collection = []
    state_y_collection = []

    for state in state_sequence:

        state_x = [0]
        state_y = [0]

        for index,step_direction in enumerate(state):

            if step_direction == 'u':
                state_x.append(state_x[index])
                state_y.append(state_y[index]+1)
            if step_direction == 'd':
                state_x.append(state_x[index])
                state_y.append(state_y[index]-1)
            if step_direction == 'r':
                state_x.append(state_x[index]+1)
                state_y.append(state_y[index])

        state_x_collection.append(state_x)
        state_y_collection.append(state_y)

    return state_x_collection,state_y_collection

#goes from step direction to ordered pairs of vertices
def states_to_graph(state_sequence):

    state_graph_collection = []
    
    for state in state_sequence:

        state_graph = [(0,0)]
        for index,step_direction in enumerate(state):

            if step_direction == 'u':
                next_joint = (state_graph[index][0], state_graph[index][1] + 1)

            if step_direction == 'd':
                next_joint = (state_graph[index][0], state_graph[index][1] - 1)

            if step_direction == 'r':
                next_joint = (state_graph[index][0] + 1, state_graph[index][1])

            state_graph.append(next_joint)

        state_graph_collection.append(state_graph)


    return state_graph_collection

#animation code
#resets the plot with the updates information
#does so mod the length of the state so that the animation loops
def update_state(i,start_state,max_rights,grid_height,x_data,y_data):

    plt.gcf().clear()
    ax = fig.add_subplot(111)
    axes = plt.gca()


    axes.set_xlim([-1,max_rights+1])
    axes.set_ylim([-1,int(grid_height)+1])
    vertices = plt.plot(x_data[i%len(x_data)],y_data[i%len(y_data)], 'ro')
    line = plt.plot(x_data[i%len(x_data)],y_data[i%len(y_data)])

    plt.setp(line, color = 'r', linewidth = 3.0)

    

    return vertices,line




""" USER INPUT SEQUENCE """


print("This program takes a starting and ending state of a robotic arm of equal lengths and determines the normal cube path (optimal path).")
print("We define a state by a sequence of steps 'u', 'd', and 'r' where u - up, d - down, and r - right.")

while(True):
    
    while(True):

        #make sure grid height is an integer
        grid_height = raw_input("Enter the number of rows in the grid (grid height): ")
        while(type(grid_height) != int and grid_height <= 0):
            print "Invalid grid height. Please enter a positive integer value."
            grid_height = raw_input("enter the number of rows in ithe grid (grid height): ")

        grid_height = int(grid_height)
        
        start_state = enter_valid_state(grid_height)
        end_state = enter_valid_state(grid_height)

        if len(start_state) != len(end_state):
            print("Length of robotic arms are incongruent. Enter both states again.")
        else:
            break

    #sets conditions to recursively generate the join irreducible elements
    JoinIrreducible.set_arm_length(len(start_state))
    JoinIrreducible.set_grid_height(grid_height)
    generate_ancestry(JoinIrreducible((),JoinIrreducible.arm_length - 1))
    pip = PIP(JoinIrreducible.join_irreducible_collection)

    
    start_order_ideal = compute_order_ideal(pip,decompose_tableaux(state_to_tableaux(start_state)))
    end_order_ideal = compute_order_ideal(pip,decompose_tableaux(state_to_tableaux(end_state)))

    pip.reroot(start_order_ideal)
    antichain_layers = normal_cube_path(pip,start_order_ideal,end_order_ideal)
    state_sequence = compute_state_sequence(pip,start_order_ideal,antichain_layers)

    x_coordinates,y_coordinates = states_to_graphs(state_sequence)

    print "The minimum number of steps is %d ." %len(antichain_layers)
    print "The minimum number of individual moves is %d.\n\n\n" %(len(start_order_ideal ^ end_order_ideal))

    graphs = states_to_graph(state_sequence)

    #determines the necessary horizontal axis for the animation
    #prints each state
    max_rights = 0
    for state in state_sequence:
        current_right_count = 0
        print ''.join(state)
        for step in state:
            if step == 'r':
                current_right_count += 1
        if max_rights < current_right_count:
            max_rights = current_right_count

    
    fig = plt.figure(figsize = (7,5))
    ax = fig.add_subplot(111)
    axes = plt.gca()

    axis_length = max_rights +1
    axes.set_xlim([-1,max_rights + 1])
    axes.set_ylim([-1,int(grid_height)+1])

    ani = animation.FuncAnimation(fig, update_state, frames = 1000, interval = 300, blit = False, fargs = (start_state,max_rights,grid_height,x_coordinates,y_coordinates), repeat = True)
    
    plt.show()

    #end of use, clearing join irreducibles collection
    JoinIrreducible.join_irreducible_collection.clear()




