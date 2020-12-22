# Gambit-dijkstra-bot
This repository contains the source code for my submission to the CoderOne AI challenge in December 2020.

Our bot uses a method that has been applied to direct AI in other games such as Brogue, known as Dijkstra maps within the gaming space. These models have also been referred to as _potential_ or _flow_ methods in more scientific domains. For a write-up demonstrating the theory and the usefulness of these methods, see [here](http://www.roguebasin.com/index.php?title=The_Incredible_Power_of_Dijkstra_Maps). Our code makes use of a Python implemention written by [Kehvarl](https://github.com/Kehvarl/Dijkstra). 

## What is a Dijsktra map?

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

To navigate a Dijkstra map, our agent decides to move to the lowest adjacent value. If the lowest value is the current position, then the agent will remain in the same location. If there are multiple options, the agent randomly chooses between them.

A key observation is in the second example, an agent placed at the centre can navigate to any of the four goals. This demonstrates that the agent already has some capacity to decide between different goals (even if our decision process is random at this point). 

## Combining maps
Given two Dijkstra maps `D_1` and `D_2`, we can define the sum elemntwise by `D_3[i][j] = D_1[i][j] + D_2[i][j]`. 
