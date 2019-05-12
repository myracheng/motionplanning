"""
Path Planning Sample Code with RRT for car like robot.
author: AtsushiSakai(@Atsushi_twi)
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import random
import sys
import os

try:
    import dubins_path_planning
except:
    raise


show_animation = False


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea,
                 goalSampleRate=10, maxIter=1000):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start[0], start[1], start[2])
        self.end = Node(goal[0], goal[1], goal[2])
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self, animation=False):
        """
        Pathplanning
        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        for i in range(self.maxIter):
            rnd = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            newNode = self.steer(rnd, nind)

            if self.__CollisionCheck(newNode, self.obstacleList):
                self.nodeList.append(newNode)

            if animation and i % 5 == 0:
                self.DrawGraph(rnd=rnd)

        # generate course
        lastIndex = self.get_best_last_index()
        #  print(lastIndex)

        if lastIndex is None:
            return None

        path = self.gen_final_course(lastIndex)
        # print(len(self.nodeList))
        return path

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def steer(self, rnd, nind):
        #  print(rnd)
        curvature = 1.0

        nearestNode = self.nodeList[nind]

        px, py, pyaw, mode, clen = dubins_path_planning.dubins_path_planning(
            nearestNode.x, nearestNode.y, nearestNode.yaw, rnd[0], rnd[1], rnd[2], curvature)

        newNode = copy.deepcopy(nearestNode)
        newNode.x = px[-1]
        newNode.y = py[-1]
        newNode.yaw = pyaw[-1]

        newNode.path_x = px
        newNode.path_y = py
        newNode.path_yaw = pyaw
        newNode.cost += clen
        newNode.parent = nind

        return newNode

    def get_random_point(self):

        if random.randint(0, 100) > self.goalSampleRate:
            rnd = [random.uniform(self.minrand, self.maxrand),
                   random.uniform(self.minrand, self.maxrand),
                   random.uniform(-math.pi, math.pi)
                   ]
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y, self.end.yaw]

        return rnd

    def get_best_last_index(self):
        #  print("get_best_last_index")

        disglist = [self.calc_dist_to_goal(
            node.x, node.y) for node in self.nodeList]
        goalinds = [disglist.index(i) for i in disglist if i <= 0.1]

        if len(goalinds) > 0:
            mincost = min([self.nodeList[i].cost for i in goalinds])
            for i in goalinds:
                if self.nodeList[i].cost == mincost:
                    return i
            # print("bug1 here") 
            # print(self.nodeList)
            sys.exit()

            return None
        # print("bug 2 here")
        # print(self.nodeList)
        # sys.exit()
        return None

#returns list of nodes.
    def gen_final_course(self, goalind):
        path = []
        # path = [[self.end.x, self.end.y, self.end.yaw]]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            for count, (ix, iy, iyaw) in enumerate(zip(reversed(node.path_x), reversed(node.path_y), reversed(node.path_yaw))):
                if count % 20 == 0:
                    path.append([ix, iy, iyaw])
            # path.append([node.x, node.y, node.yaw])
            goalind = node.parent
        # path.append([self.start.x, self.start.y, self.start.yaw])
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])

    def DrawGraph(self, rnd=None):  # pragma: no cover
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "ok", ms=30 * size)

        dubins_path_planning.plot_arrow(
            self.start.x, self.start.y, self.start.yaw)
        dubins_path_planning.plot_arrow(
            self.end.x, self.end.y, self.end.yaw)

        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2
                 + (node.y - rnd[1]) ** 2
                 + (node.yaw - rnd[2] ** 2) for node in nodeList]
        minind = dlist.index(min(dlist))

        return minind

    def __CollisionCheck(self, node, obstacleList):

        for (ox, oy, size) in obstacleList:
            for (ix, iy) in zip(node.path_x, node.path_y):
                dx = ox - ix
                dy = oy - iy
                d = dx * dx + dy * dy
                if d <= size ** 2:
                    return False  # collision

        return True  # safe


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.path_x = []
        self.path_y = []
        self.path_yaw = []
        self.cost = 0.0
        self.parent = None


def main():
    print("Start " + __file__)
    # ====Search Path with RRT====
    # obstacleList = [
        # (5, 5, 1),
        # (3, 6, 2),
        # (3, 8, 2),
        # (3, 10, 2),
        # (7, 5, 2),
        # (9, 5, 2)
    # ]  # [x,y,size(radius)]
    obstacleList = [
        (36, 43, 2),
        (12, 16, 2),
        (24, 34, 2)
    ] 
    # Set Initial parameters
    # for i in range (50):
        # for j in range(50):
    start = [7, 42, np.deg2rad(0.0)] #starting at obstacle.
    # goal = [10.0, 10.0, np.deg2rad(0.0)]
    goal = [27.0, 13.0, np.deg2rad(0.0)]
    print(start)
    rrt = RRT(start, goal, randArea=[0, 50.0], obstacleList=obstacleList)
    
    path = rrt.Planning(animation=False) #returns list of nodes
    print(path)
            # path = rrt.Planning(animation=show_animation)
            # print(len(path))
            # print(self.nodeList)

            # Draw final path
            # if show_animation:  # pragma: no cover
            # rrt.DrawGraph()
            # plt.plot([x for (x, y, z) in path], [y for (x, y, z) in path], '-o')
            # plt.grid(True)
            # plt.pause(10)
            # plt.show()


if __name__ == '__main__':
    main()