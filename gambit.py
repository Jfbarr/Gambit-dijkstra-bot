import numpy as np
import random
import time

class Agent:
    
    def __init__(self):
        self.name="Dijkstra-bot"
        self.desire ={'safe':3, #8 is super safe, 1 is super risky
                      'flee':0.35,
                      'reward':2,
                      'ammo':1,
                      'hunt':1,
                      'treasure':0,
                      'player-aversion':0, 
                      'run':0}
        self.reward_dijkstra = DijkstraMap(12,10,)
        self.previous_move = None
        self.previous_location = None
        self.safe_bomb_dijkstra = DijkstraMap(12,10,None)
        self.safe_bomb_dijkstra_high = DijkstraMap(12,10,None)
        self.tick =0
        self.current_bombs = []
        self.all_bombs = []
        self.all_explosions = []
        self.previous_bombs = set()
        
        
    def generate_one_reward_map(self, state):
        state_pad = np.zeros((14,12))
        state_pad[1:-1, 1:-1] = state
        score_conv = [[0,1,0],[1,0,1],[0,1,0]]
        #state_pad = np.where(state_pad ==np.nan, 0,state_pad)
        reward_map = np.zeros((12,10))
        state_pad[np.isnan(state_pad)]=0
        for x in range(12):
            for y in range(10):
                sub_matrix = state_pad[x:x+3, y:y+3]
                reward_map[x,y] = np.dot(sub_matrix[0], score_conv[0]) \
                                + np.dot(sub_matrix[1], score_conv[1]) \
                                + np.dot(sub_matrix[2], score_conv[2])
        return reward_map
    
    
    def generate_reward_map(self, state):
        def evaluate_sub_matrix(mat):
            directions = [
                np.array([ 1, 0]),  # right
                np.array([-1, 0]),  # left
                np.array([ 0, 1]),  # up
                np.array([ 0,-1]),  # down
            ]

            origin = np.array([2,2])
            val = 0

            for direction in directions:
                for k in range(1, 3):
                    v = origin + k*direction
                    if np.isnan(mat[v[0], v[1]]):
                        break
                    elif mat[v[0], v[1]] == 0:
                        pass
                    else:
                        val += mat[v[0], v[1]]*(1 if k==1 else 0.95)
                        break
            return val

        state_pad = np.zeros((16,14))
        state_pad[2:-2, 2:-2] = state
        score_conv = [[0,1,0],[1,0,1],[0,1,0]]
        reward_map = np.zeros((12,10))
        for x in range(12):
            for y in range(10):
                sub_matrix = state_pad[x:x+5, y:y+5]
                reward_map[x,y] = evaluate_sub_matrix(sub_matrix)
        return reward_map

    def generate_walls(self, game_state, player_state):
        return game_state.all_blocks + game_state.bombs + game_state.opponents(player_state.id)
        
    def generate_reward_dijkstra(self,game_state, player_state):
        """
        Generate a dijkstra map representing the agents desire
        to get points
        :param game_board: Current game board
        :return DijkstraMap: Dijstra map for reward seeking
        """

        directions = [
            np.array([ 0, 1]),
            np.array([ 0,-1]),
            np.array([ 1, 0]),
            np.array([-1, 0]),
        ]
        explosion_tiles = []
        
        # Populate list of tiles that will explode
        for bomb in game_state.bombs:
            explosion_tiles.append(bomb)
            for direction in directions:
                for k in range(1,3):
                    e_tile = (bomb[0]+k*direction[0], bomb[1]+k*direction[1])
                    if e_tile in game_state.all_blocks + game_state.bombs:
                        explosion_tiles.append(e_tile)
                        break
                    else:
                        explosion_tiles.append(e_tile)
        
        game_board = np.zeros((12,10))
        for soft_block_idx in game_state.soft_blocks:
            if soft_block_idx in explosion_tiles:
                game_board[soft_block_idx] = 0
            else:
                game_board[soft_block_idx] = 2
        for ore_block_idx in game_state.ore_blocks:
            ore_ex = [e for e in self.all_explosions if e == ore_block_idx] 
            if len(ore_ex)==0:
                game_board[ore_block_idx]  = 0.5
            if len(ore_ex)==1:
                game_board[ore_block_idx]  = 3
            if len(ore_ex)>=2:
                game_board[ore_block_idx]  = 11
        for indestructible_block_idx in game_state.indestructible_blocks:
            game_board[indestructible_block_idx] = np.nan
            
        wall_indices = self.generate_walls(game_state, player_state)
        reward_dijkstra = DijkstraMap(12, 10, wall_indices)
        if len(game_state.ore_blocks + game_state.soft_blocks)<5:
            score_map = self.generate_one_reward_map(game_board)
        else:
            score_map = self.generate_reward_map(game_board)
        nonzero_ind = np.nonzero(score_map)
        goal_indices = [(x,y) for x,y in zip(nonzero_ind[0], nonzero_ind[1])]
        for goal in goal_indices:
            reward_dijkstra.add_goal(int(goal[0]), int(goal[1]),-1*score_map[goal[0], goal[1]])
        reward_dijkstra.recalculate_map()
        return reward_dijkstra

    def generate_bomb_flee_dijkstra(self, game_state, player_state):
        """
        Generate a dijkstra map representing the agents desire
        to flee from bombs
        :param state: Current game board
        :return DijkstraMap: Dijstra map for reward seeking
        """
        
        wall_indices = game_state.all_blocks
        bomb_dijkstra = DijkstraMap(12,10,wall_indices)            
        goal_indices = game_state.bombs
        
        for goal in goal_indices:
            bomb_dijkstra.add_goal(int(goal[0]), int(goal[1]),-1)
        bomb_dijkstra.recalculate_map()
        bomb_dijkstra = -1.3*bomb_dijkstra
        bomb_dijkstra.recalculate_map(clear=False)
        return bomb_dijkstra
    
    
    
    def generate_bomb_safe_dijkstra_high_timer(self, game_state, player_state):

        wall_indices = self.generate_walls(game_state, player_state)
        bomb_dijkstra = DijkstraMap(12,10,wall_indices) 
        bomb_indices = []

        directions = [
            np.array([ 0, 1]),
            np.array([ 0,-1]),
            np.array([ 1, 0]),
            np.array([-1, 0]),
        ]
        explosion_tiles = []
        
        # Populate list of tiles that will explode
        bombs = [(b[0], b[1]) for b in self.current_bombs if self.tick - b[2] >30]
        for bomb in bombs:
            explosion_tiles.append(bomb)
            for direction in directions:
                for k in range(1,3):
                    e_tile = (bomb[0]+k*direction[0], bomb[1]+k*direction[1])
                    if e_tile in game_state.all_blocks + game_state.bombs:
                        break
                    else:
                        explosion_tiles.append(e_tile)
        for x in range(12):
            for y in range(10):
                if (x,y) in game_state.all_blocks+game_state.bombs+explosion_tiles:
                    continue
                else:
                    bomb_dijkstra.add_goal(x,y,0)
        bomb_dijkstra.recalculate_map()
        return bomb_dijkstra     
    
    
    
    def generate_bomb_safe_dijkstra(self, game_state, player_state):
       
        wall_indices = self.generate_walls(game_state, player_state)
        bomb_dijkstra = DijkstraMap(12,10,wall_indices) 
        bomb_indices = []
       
        directions = [
            np.array([ 0, 1]),
            np.array([ 0,-1]),
            np.array([ 1, 0]),
            np.array([-1, 0]),
        ]
        explosion_tiles = []
        
        # Populate list of tiles that will explode
        for bomb in game_state.bombs:
            explosion_tiles.append(bomb)
            for direction in directions:
                for k in range(1,3):
                    e_tile = (bomb[0]+k*direction[0], bomb[1]+k*direction[1])
                    if e_tile in game_state.all_blocks + game_state.bombs:
                        break
                    else:
                        explosion_tiles.append(e_tile)
        for x in range(12):
            for y in range(10):
                if (x,y) in game_state.all_blocks+game_state.bombs+explosion_tiles:
                    continue
                else:
                    bomb_dijkstra.add_goal(x,y,0)
        bomb_dijkstra.recalculate_map()
        return bomb_dijkstra   


    def generate_ammo_dijkstra(self, game_state, player_state):
        
        wall_indices = self.generate_walls(game_state, player_state)
        ammo_dijkstra = DijkstraMap(12,10,wall_indices) 
        for ammo in game_state.ammo:
            ammo_dijkstra.add_goal(ammo[0], ammo[1], -3)
        for treasure in game_state.treasure:
            ammo_dijkstra.add_goal(treasure[0], treasure[1], -2)
        ammo_dijkstra.recalculate_map()
        return ammo_dijkstra
    
    def generate_treasure_dijkstra(self, game_state, player_state):
        
        wall_indices = self.generate_walls(game_state, player_state)
        treasure_dijkstra = DijkstraMap(12,10,wall_indices) 
        for treasure in game_state.treasure:
            treasure_dijkstra.add_goal(treasure[0], treasure[1], -2)
        treasure_dijkstra.recalculate_map()
        return treasure_dijkstra

    def check_bomb_location(self, game_state, player_state):
 
        directions = [
            np.array([ 0, 1]),
            np.array([ 0,-1]),
            np.array([ 1, 0]),
            np.array([-1, 0]),
        ]
        game_board = np.zeros((12,10))
        for soft_block_idx in game_state.soft_blocks:
            game_board[soft_block_idx] = 2
        for ore_block_idx in game_state.ore_blocks:
            game_board[ore_block_idx]  = 10
        for indestructible_block_idx in game_state.indestructible_blocks:
            game_board[indestructible_block_idx] = np.nan

        score_map = self.generate_reward_map(game_board)
        walls = self.generate_walls(game_state, player_state)
        safety_dijkstra = DijkstraMap(12,10,walls)
        explosion_tiles = [player_state.location]
        # Populate list of tiles that will explode
        for bomb in game_state.bombs+[player_state.location]:
            for direction in directions:
                for k in range(1,3):
                    e_tile = (bomb[0]+k*direction[0], bomb[1]+k*direction[1])
                    if e_tile in walls:
                        break
                    else:
                        explosion_tiles.append(e_tile)
        for x in range(12):
            for y in range(10):
                if (x,y) in walls+explosion_tiles:
                    continue
                else:
                    safety_dijkstra.add_goal(x,y,0)
        safety_dijkstra.recalculate_map()

        if score_map[player_state.location[0]][player_state.location[1]] >0:
            if  safety_dijkstra._get_lowest_neighbor_value(player_state.location[0], player_state.location[1]) < 50:
                return True
        return False   

    def generate_enemy_flee_dijkstra(self, game_state, player_state):    
        """
        Generate a dijkstra map representing the agents desire
        to flee from bombs
        :param state: Current game board
        :return DijkstraMap: Dijstra map for reward seeking
        """
        
        wall_indices = game_state.all_blocks+game_state.bombs
        enemy_dijkstra = DijkstraMap(12,10,wall_indices)            
        
        for x in range(3,8):
            for y in range(3,6):
                enemy_dijkstra.add_goal(x, y, 0)
        enemy_dijkstra.recalculate_map()

        return enemy_dijkstra

    def generate_run_dijkstra(self, game_state, player_state):
        """
        Generate a dijkstra map representing the agents desire
        to flee from bombs
        :param state: Current game board
        :return DijkstraMap: Dijstra map for reward seeking
        """
        
        wall_indices = game_state.all_blocks
        run_dijkstra = DijkstraMap(12,10,wall_indices)            
        
        for goal in game_state.opponents(player_state.id):
            run_dijkstra.add_goal(int(goal[0]), int(goal[1]),0)
        run_dijkstra.recalculate_map()
        run_dijkstra = -1.2*run_dijkstra
        run_dijkstra.recalculate_map(clear=False)
        return run_dijkstra
    
    def next_move(self, game_state, player_state):
        
        self.tick+=1
        if self.previous_location:
            if self.previous_location == player_state.location and self.previous_move in ('l','r','d','u') and (self.tick - self.previous_tick)<=2:
                return None

        for bomb in game_state.bombs:
            if bomb not in [(b[0], b[1]) for b in self.current_bombs]:
                self.current_bombs.append((bomb[0], bomb[1], self.tick))
        filtered = [bomb for bomb in self.current_bombs if (bomb[0], bomb[1]) in game_state.bombs]
        self.current_bombs = filtered

        if set(game_state.bombs) != self.previous_bombs:
            diff = set(game_state.bombs) - self.previous_bombs
            if len(diff)>0:
                self.all_bombs.extend(list(diff))
                self.all_explosions.append(list(diff))
                directions = [
                    np.array([ 0, 1]),
                    np.array([ 0,-1]),
                    np.array([ 1, 0]),
                    np.array([-1, 0]),
                ]
                for bomb in diff:
                    for direction in directions:
                        for k in range(1,3):
                            e_tile = (bomb[0]+k*direction[0], bomb[1]+k*direction[1])
                            if e_tile in game_state.all_blocks + game_state.bombs:
                                self.all_explosions.append(e_tile)
                                break
                            else:
                                self.all_explosions.append(e_tile)

        if len(game_state.ore_blocks + game_state.soft_blocks)==0:
            self.desire['player-aversion']=1
            self.desire['run'] = 0.25
            self.desire['treasure'] = 5
            self.desire['reward'] = 0
        
        if player_state.ammo <=1:
            self.desire['ammo']=9
            if player_state.ammo == 0:
                self.desire['reward']=0.5
            else:
                self.desire['reward']=2
        else:
            self.desire['ammo']= 1.4 
        if player_state.hp ==1:
            self.desire['safe']=8
        else:
            self.desire['safe']=1 
        
        bomb_dijkstra = self.generate_bomb_flee_dijkstra(game_state, player_state)
        reward_dijkstra = self.generate_reward_dijkstra(game_state, player_state)
        ammo_dijkstra = self.generate_ammo_dijkstra(game_state, player_state)
        self.safe_bomb_dijkstra = self.generate_bomb_safe_dijkstra(game_state, player_state)
        flee_enemy_dijkstra = self.generate_enemy_flee_dijkstra(game_state, player_state)
        run_dijkstra = self.generate_run_dijkstra(game_state, player_state)
        treasure_dijkstra = self.generate_treasure_dijkstra(game_state, player_state)
        
        goal_dijkstra = self.desire['reward'] * reward_dijkstra \
                      + self.desire['safe']   * self.safe_bomb_dijkstra \
                      + self.desire['flee']   * bomb_dijkstra \
                      + self.desire['ammo']   * ammo_dijkstra \
                      + 25 * self.safe_bomb_dijkstra_high \
                      + self.desire['player-aversion'] * flee_enemy_dijkstra \
                      + self.desire['run'] * run_dijkstra \
                      + self.desire['treasure'] * treasure_dijkstra

        self.safe_bomb_dijkstra_high = self.generate_bomb_safe_dijkstra_high_timer(game_state, player_state)
        player_location = player_state.location
        
        move_dict={
        (0,0):'',
        (-1,0):'l',
        (1,0):'r',
        (0,1):'u',
        (0,-1):'d'
         }
        
        moves = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
        move_list = []
        best = 1000
        for move in moves:
            if goal_dijkstra.point_in_map(player_location[0] + move[0],player_location[1] + move[1]):
                value = goal_dijkstra.tiles[player_location[0] + move[0]][player_location[1]+move[1]]
                if value <= best:
                    best = goal_dijkstra.tiles[player_location[0]+move[0]][player_location[1]+move[1]]
        for move in moves: 
            if goal_dijkstra.point_in_map(player_location[0] + move[0],player_location[1] + move[1]):
                if goal_dijkstra.tiles[player_location[0] + move[0]][player_location[1] + move[1]] == best:
                    move_list.append(move)
        
        
        if move_list:
            move = random.choice(move_list)
        else:
            move = (0,0)    
        action = move_dict[move]
        self.previous_bombs = set(game_state.bombs)
        if game_state.entity_at((player_location[0]+move[0], player_location[1]+move[1])) ==1-player_state.id:
            action = ''
            self.previous_move = action
            self.previous_location = player_location
            return action
        
                
        if action == '':
            if self.check_bomb_location(game_state, player_state) and player_state.ammo>0:
                action = 'b'
        
        self.previous_move = action
        self.previous_location = player_location
        self.previous_tick = self.tick

        return action


class DijkstraMap:
    """
    Python implementation of Dijkstra Maps  
    Source: http://www.roguebasin.com/index.php?title=The_Incredible_Power_of_Dijkstra_Maps
            http://www.roguebasin.com/index.php?title=Dijkstra
    """

    # X, Y Transitions to the 8 neighboring cells
    neighbors = [(0, -1), (-1, 0), (1, 0),(0, 1),(0,0)]

    def __init__(self, width, height, walls = None):
        """
        Create a Map  showing the movement score of various tiles
        :param int width: Map size in tiles
        :param int height: Map size in tiles
        """
        self.width = width
        self.height = height
        if walls:
            self.walls = walls
        else:
            self.walls = []
        self.goals = []
        self.tiles = []
        self._clear_map()

    def add_goal(self, x, y, score=0):
        """
        Add a goal tile to the map
        :param int x: Tile X coordinate
        :param int y: Tile Y coordinate
        :param int score: Desirability of this location (default: 0)
        """
        #print("WALLS:")
        #print(self.walls)
        #print(x,y)
        if (x,y) in self.walls:
            z=1
        else:
            self.goals.append((x, y, score))

    def recalculate_map(self, clear=True):
        """
        Use Dijkstra's Algorithm to calculate the movement score towards
        goals in this map
        """
        if clear:
            self._clear_map()
        changed = True
        while changed:
            changed = False
            for x in range(0, self.width):
                for y in range(0, self.height):
                    if (x,y) in self.walls:
                        continue
                    lowest_neighbor = self._get_lowest_neighbor_value(x, y)
                    if self.tiles[x][y] > lowest_neighbor + 1:
                        self.tiles[x][y] = lowest_neighbor + 1
                        changed = True

    def get_move_options(self, x, y):
        """
        Return a list of ideal moves from a given point
        :param x: Entity X Coordinate
        :param y: Entity Y Coordinate
        :return list: Recommended moves
        """
        best = self._get_lowest_neighbor_value(x, y)
        moves = []
        for dx, dy in DijkstraMap.neighbors:
            tx, ty = x + dx, y + dy
            if self.point_in_map(tx, ty) and self.tiles[tx][ty] == best:
                moves.append( (dx, dy))
        return moves

    def point_in_map(self, x, y):
        """
        Checks if a given point falls within the current map
        :param x: Target X position
        :param y: Target Y position
        :return: True if desired location is within map bounds
        """
        return 0 <= x < self.width and 0 <= y < self.height and (x,y) not in self.walls

    def _clear_map(self, default=100):
        """
        Reset the map scores to an arbitrary value and populate goals
        :param int default: the initial value to set for each cell
        """
        self.tiles = [
            [default
             for _ in range(self.height)]
            for _ in range(self.width)]

        for (x, y, score) in self.goals:
            self.tiles[x][y] = score

        for (x,y) in self.walls:
            self.tiles[x][y] = np.nan

    def _get_lowest_neighbor_value(self, x, y):
        """
        Get the score in the current lowest-valued neighbor cell
        :param x: Current X Coordinate
        :param y: Current Y Coordinate
        :return int: Lowest neighboring value
        """
        lowest = 100
        for dx, dy in DijkstraMap.neighbors:
            tx, ty = x + dx, y + dy
            if self.point_in_map(tx, ty):
                lowest = min(lowest, self.tiles[tx][ty])
        return lowest

    def __repr__(self):
        """
        Output the current map in a printable fashion
        :return string: Printable form of map
        """
        out = ""
        for x in range(0, self.width):
            for y in range(0, self.height):
                out += str("{:.1f}".format(self.tiles[x][y])).rjust(6,' ')
            out += "\n"
        return out
    
    def __add__(self, other):
        if self.width != other.width or self.height != other.height:
            return None
            print("ERROR: Different sizes")
        result_map = DijkstraMap(self.width, self.height, self.walls)
        result_map.tiles = [
            [self.tiles[x][y] +other.tiles[x][y]
                for y in range(0, self.height)]
                for x in range(0, self.width)]
        return result_map
    
    def __mul__(self, num):
        result_map = DijkstraMap(self.width, self.height, self.walls)
        result_map.tiles =[
            [float(self.tiles[x][y]) * num
                for y in range(self.height)]
                for x in range(self.width)]
            
        return result_map
    
    def __rmul__(self,num):
        return self.__mul__(num)
