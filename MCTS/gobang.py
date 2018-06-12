# -*- coding:utf-8 -*-

from math import *
import random
import numpy as np
import tkinter as tk
from tkinter import Canvas, messagebox
import threading
from timeit import timeit

class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """

    def __init__(self, board_sz, win_num):
        self.playerJustMoved = 2  # At the root pretend the player just moved is player 2 - player 1 has the first move 上一个移动的player
        self.checkerboard = np.full(shape=(board_sz,board_sz), fill_value=0) # 0表示没用过
        self.win_num = win_num


    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState(self.checkerboard.shape[0], self.win_num)
        st.playerJustMoved = self.playerJustMoved
        st.checkerboard = self.checkerboard.copy()
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
            move = (row, col)
        """
        self.playerJustMoved = 3 - self.playerJustMoved
        self.checkerboard[move[0], move[1]] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        moves = list()
        rows, cols = self.checkerboard.shape
        for row in range(rows):
            for col in range(cols):
                if self.checkerboard[row, col] == 0:
                    moves.append((row, col))
        return moves

    def GetResult(self, playerjm, lastmove):
        # 先判断最后这一步放进来能不能赢
        row, col = lastmove
        rows, cols = self.checkerboard.shape
        player_id = self.checkerboard[row, col]

        # 行
        for i in range(row, -1, -1): #(-1,row]
            if self.checkerboard[i][col] != player_id:
                break
        pre_len = row - i
        for i in range(row, rows):
            if self.checkerboard[i][col] != player_id:
                break
        sufix_len = i - row
        len =  pre_len + sufix_len - 1
        if len >= self.win_num:
            if player_id == playerjm:
                return 1.0
            else:
                return 0.0

        # 列
        for i in range(col, -1, -1):
            if self.checkerboard[row][i] != player_id:
                break
        pre_len = col - i
        for i in range(col, cols):
            if self.checkerboard[row][i] != player_id:
                break
        sufix_len = i - col
        len = pre_len + sufix_len - 1
        if len >= self.win_num:
            if player_id == playerjm:
                return 1.0
            else:
                return 0.0

        # 左上到右下
        i = row; j = col
        pre_len = 0
        while i >= 0 and j >= 0:
            if board.checkerboard[i, j] == player_id:
                pre_len += 1
                i -= 1
                j -= 1
            else:
                break

        sufix_len = 0
        i = row; j = col
        while i < self.rows and j < cols:
            if self.checkerboard[i, j] == player_id:
                sufix_len += 1
                i += 1
                j += 1
            else:
                break
        len = pre_len + sufix_len - 1
        if len >= self.win_num:
            if player_id == playerjm:
                return 1.0
            else:
                return 0.0

        # 左下到右上
        i = row; j = col
        pre_len = 0
        while i < rows and j > 0:
            if self.checkerboard[i, j] == player_id:
                pre_len += 1
                i += 1
                j -= 1
            else:
                break

        sufix_len = 0
        i = row; j = col
        while i > 0 and j < cols:
            if self.checkerboard[i, j] == player_id:
                sufix_len += 1
                i -= 1
                j += 1
            else:
                break
        len = pre_len + sufix_len - 1
        if len >= self.win_num:
            if player_id == playerjm:
                return 1.0
            else:
                return 0.0

        # 如果没赢，那看后面还有位置没
        # ids = map(np.unique, board.checkerboard)
        ids = np.unique(self.checkerboard)
        if 0 not in list(ids):
            # 没有位置了
            return 0.5

        # 还能玩
        return 0.5


    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        n = self.win_num #几个子连起来就赢了
        rows, cols = self.checkerboard.shape
        for row in range(rows):
            for col in range(cols):
                player = self.checkerboard[row, col]
                if player == 0:
                    continue

                # 行
                if (col + n - 1< cols) and (len(set(self.checkerboard[row, j] for j in range(col ,col + n))) == 1):
                    if playerjm == player:
                        return 1.0
                    else:
                        return 0.0

                # 列
                if (row + n - 1< rows) and (len(set(self.checkerboard[i, col] for i in range(row, row + n))) == 1):
                    if playerjm == player:
                        return 1.0
                    else:
                        return 0.0

                # 斜线--右斜向上 /
                if (n-1 <= row and col + n - 1 < cols) and (len(set(self.checkerboard[row - i, col + i] for i in range(n))) == 1):
                    if playerjm == player:
                        return 1.0
                    else:
                        return 0.0

                # 斜线--左斜向下 \
                if (row + n - 1 < rows and col + n-1 < cols) and (len(set(self.checkerboard[row + i, col + i] for i in range(n))) == 1):
                    if playerjm == player:
                        return 1.0
                    else:
                        return 0.0

        # 到这还没返回，说明没有分出胜负, 每个人赢一半
        return 0.5


    def __repr__(self):
        """ Don't need this - but good style.
        """
        s = "AvaMoves:" + str(len(self.GetMoves())) + " JustPlayed:" + str(self.playerJustMoved)
        return s


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()  # future child nodes
        self.playerJustMoved = state.playerJustMoved  # the only part of the state that the Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(
            self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose=False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []:  # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []:  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []:  # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(
                node.playerJustMoved))  # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose):
        print(rootnode.TreeToString(0))
    else:
        print(rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move  # return the move that was most visited

class Graph(object):
    def __init__(self, title="GoBang15", gem_sz=600, board_sz=15):
        self.title = title
        self.sz = gem_sz
        self.board_cols = board_sz
        self.board_rows = board_sz

        self.init_root()
        self.init_canvas()
        self.init_bind()

    def init_root(self):
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry(str(self.sz) + "x" + str(self.sz))

    def init_canvas(self):
        self.canvas = Canvas(self.root,width=600,height=600,borderwidth=3,background='white')
        self.canvas.pack()
        # CheckerBoard
        self.startX = (600 - 15 * 30) / 2
        self.startY = (600 - 15 * 30) / 2
        for num in range(0, self.board_cols + 1):
            self.canvas.create_line(self.startX + num * 30, 0 + self.startY, self.startX + num * 30, self.startY + 450, width=2)
        for num in range(0, self.board_rows + 1):
            self.canvas.create_line(self.startX + 0, self.startY + num * 30, self.startX + 450, self.startY + num * 30, width=2)

    def init_bind(self):
        global msg
        global player_id
        global player_move
        self.root.bind("<<showmsg>>", lambda event: self.show_msg(event, msg))
        self.canvas.bind("<<update>>", lambda event: self.update_graph(event,player_id, player_move))

    def show(self):
        self.root.mainloop()

    def cross(self, X, Y):
        """
        :param X:  小方格左上角的坐标
        :param Y:
        """
        cross_sz = 10
        self.canvas.create_line(X + cross_sz, Y + cross_sz, X - cross_sz, Y - cross_sz, width=4, fill="red")
        self.canvas.create_line(X - cross_sz, Y + cross_sz, X + cross_sz, Y - cross_sz, width=4, fill="red")
    def circle(self, X, Y):
        d = 20
        self.canvas.create_oval(X - d / 2, Y - d / 2, X + d / 2, Y + d / 2, width=4, outline='green')

    def update_graph(self, event, player_id, move):
        # AI确定落子位置后，在tkinter上显示出来
        row, col = move
        boardX = col * 30 + self.startX
        boardY = row * 30 + self.startY
        if player_id == 1:
            self.cross(boardX + 15, boardY + 15) #AI用cross
        else:
            self.circle(boardX + 15, boardY + 15)


    def show_msg(self, event, msg):
        tk.messagebox.showinfo(title="Game Over", message=msg)

def UCTPlayGame(window):
    """ Play a sample game between two UCT players where each player gets a different number
        of UCT iterations (= simulations = tree nodes).
    """
    global player_id
    global player_move
    global msg
    state = GameState(7, 5) # 设置棋盘大小
    while (state.GetMoves() != []):
        print(str(state))
        if state.playerJustMoved == 1:
            m = UCT(rootstate=state, itermax=3000, verbose=False)  # play with values for itermax and verbose = False
        else:
            m = UCT(rootstate=state, itermax=3000, verbose=False)
        print("Best Move: " + str(m) + "\n")
        state.DoMove(m)

        player_id = state.playerJustMoved
        player_move = m
        window.canvas.event_generate("<<update>>")

        if state.GetResult(state.playerJustMoved) == 1.0:
            msg = "Player " + str(state.playerJustMoved) + " wins!"
            if player_id == 1:
                msg = msg + "cross"
            else:
                msg = msg + "circle"
            print(msg)
            window.root.event_generate("<<showmsg>>")
            return

        elif state.GetResult(state.playerJustMoved) == 0.0:
            msg = "Player " + str(3 - state.playerJustMoved) + " wins!"
            if player_id == 1:
                msg = msg + "cross"
            else:
                msg = msg + "circle"
            print(msg)
            window.root.event_generate("<<showmsg>>")
            return

    print("Nobody wins!")
    msg = "No Wins!"
    window.root.event_generate("<<showmsg>>")


if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    window = Graph()
    global player_id
    global player_move
    global msg

    game = threading.Thread(target=UCTPlayGame, args=(window,))
    game.start()

    window.show()

    # state = GameState(board_sz=7, win_num=5)
    # state.checkerboard[0, :5] = 1
    # assert state.GetResult(1) == 1.0
    #
    # state.checkerboard[:,:] = 0
    # state.checkerboard[:5,1] = 1
    # assert state.GetResult(1) == 1.0
    #
    # state.checkerboard[:,:] = 0
    # for i in range(5):
    #     state.checkerboard[i, i] = 1
    # print(state.checkerboard)
    # print(state.GetResult(1))
    # assert state.GetResult(1) == 1.0
    #
    # state.checkerboard[:,:] = 0
    # for i in range(5):#刨除每一行的最后一列
    #     state.checkerboard[i, 5-i] = 1
    # print(state.checkerboard)
    # assert state.GetResult(1) == 1.0
    #
    # state = GameState(board_sz=10, win_num=5)
    # state.checkerboard[0, 5:] = 1
    # print(state.checkerboard[0])
    # assert state.GetResult(1) == 1.0





