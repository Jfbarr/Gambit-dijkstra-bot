# Gambit-dijkstra-bot
This repository contains the source code for my submission to the CoderOne AI challenge in December 2020.

Our bot uses a method that has been applied to control AI in other games such as Brogue, known as Dijkstra maps within the gaming space. These models have also been referred to as _potential_ or _flow_ methods in other domains. For a write-up demonstrating the theory and the usefulness of these methods, see [here](http://www.roguebasin.com/index.php?title=The_Incredible_Power_of_Dijkstra_Maps). Our code makes use of a Python implemention written by [Kehvarl](https://github.com/Kehvarl/Dijkstra). 

# Dijkstra maps
## Overview

To populate a Dijkstra map, the following algorithm is applied:

1. Set each grid cell to a default value of 100
2. Given a set of goals, set the value of each goal to 0
3. For each cell `c` in the grid:
    - If there is an adjacent cell `c_a` with a difference greater than 1, set `c= c_a + 1`
    - Else do nothing
4. Repeat 1-3 until no changes are recorded

The end result will be a gridmap with 0s at our goal locations, and increasing values as we move away from the goal. For example, a 5x5 grid with a goal at the center would return:

```
43234
32123
21012
32123
43234
```

The same grid with goals at each corner would look like:

```
01210
12321
23432
12321
01210
```

To navigate a Dijkstra map, our agent decides to move to the lowest adjacent value. If the lowest value is the current position, then the agent will remain in the same location. If there are multiple options, the agent randomly chooses between them. In this way, the pathfinding problem has been reduced to "rolling downhill". Note that the default value of 0 can be replaced with any other chosen value. Negative goal values will have the agent seek out the goal from further away.

A key observation is in the second example, an agent placed at the centre can navigate to any of the four goals. This demonstrates that the agent already has some capacity to decide between different goals (even if our decision process is random at this point). 

## Combining maps
Given two Dijkstra maps `D_1` and `D_2`, we can define the sum elemntwise by `D_3[i][j] = D_1[i][j] + D_2[i][j]`. Taking our previous examples, this yields

```
43234   01210   44444
32123   12321   44444 
21012 + 23432 = 44444 
32123   12321   44444 
43234   01210   44444
```

The resulting uniform map is not particularly useful, as the agent will simply wander the field randomly. To remedy this, we define multiplication of a Dijkstra map `D` by a scalar `a` by `aD[i][j] = D[i][j] * a`. Multiplying a map by a constant can be thought of multiplying the height of the hills in the Dijkstra map, so that the agent will be more likely to navigate towards the goals in maps with higher coefficients. 

# Agent
## Overview

Our agent relies on the following logic to decide it's action for each turn.

1. If the previous action hasn't been completed, skip its turn (this accounts for desync conditions)
2. Update a list of bombs in play and the tiles that will be hit with an explosion
2. Update an internal list of 'desires' based on game state
3. Generate a list of Dijkstra maps based on game state
4. Create a final 'goal' Dijkstra map a linear combination of the maps from step 3, with coefficients calculated in step 2
5. Check which surrounding tile the agent can move to based on its current position
    * If the action returned is to do nothing, the agent checks if it should place a bomb
    * If the action returned would move the agent into the opponent, the agent does nothing

## Dijkstra map generation
The following functions are used to generate the maps in step 4:

### generate_reward_map
This map has the goals set to be the tiles on the map that will yield points if a bomb were to be placed there. The function takes into account that explosions can't travel through walls, and the health of blocks. If there are less than 6 blocks remaining on the map, only tiles one block away will have goal tiles set.

### generate_bomb_flee_dijkstra

This map is used to give our agent a general aversion to being near bombs. Using just this map, the agent would try and flee from all bombs on the map.

If we were to simply set the positions of bombs as goals and multiply the map by a negative coefficient, our agent would simply pathfind into corners of rooms. By taking the resulting map, multiplying it by a number in the vicinity of -1.2 and passing it through the algorithm again we achieve a map with smarter fleeing behaviour. Passing the map through the algorithm again essentially places goals at parts of the map farther away so that our agent won't get trapped between bombs and walls.

### generate_bomb_safe_dijkstra
This map sets every game tile that isn't in the explosion radius of a bomb as a goal. This gives our agent a notion of the tiles that are safe to walk on.

### generate_bomb_safe_dijkstra_high_timer

Similar to the previous function, this map only sets tiles to be safe if they're outside the explosion radius of a bomb that is close to exploding. This will allow our agent to avoid being in the explosion of a bomb that is close to expiring. 

This map is calculated after the goal Dijkstra map is created. This is to account for the fact that explosions last a tick after the bomb explodes.

### generate_ammo_dijkstra

This map places goals at the locations of ammo and treasure pickups. Ammo is weighted slightly higher than treasure.

### generate_treasure_dijkstra

Similar to the ammo map, this map only places goals on treasure drops.

### generate_enemy_flee_dijkstra
This map places goals in a square ranging from (3,3) to (8,6). This is used in the late game to keep our agent near the center of the map.

### generate_run_dijkstra
This map works the same was as our bomb flee map, but flees from the opponent agent.

## Bomb check logic
Before placing a bomb, our agent checks whether any of the following conditions fail:

1. The agent will receive points if it were to place a bomb, or it will damage an ore block
2. There is a path to a safe tile 


