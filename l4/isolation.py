from abc import abstractmethod
from enum import Enum
import itertools
import random
import time
from typing import Optional, Protocol
import math


class Colour(Enum):
    RED = 'R'
    BLUE = 'B'

    def flip(self) -> 'Colour':
        return Colour.RED if self == Colour.BLUE else Colour.BLUE

class Board:
    def __init__(self, width: int, height: int):
        self.width: int = width
        self.height: int = height
        self.positions: dict[tuple[int, int], str] = dict()
        self.red_position: Optional[tuple[int, int]] = None
        self.blue_position: Optional[tuple[int, int]] = None
        self._prepare_board()

    def _prepare_board(self):
        for i in range(self.width):
            for j in range(self.height):
                self.positions[(i, j)] = '.'

    def __str__(self):
        representation = '\\ ' + ' '.join([str(i + 1) for i in range(self.width)]) + '\n'
        for j in range(self.height):
            representation += (chr(ord('A') + j) + ' ' + ' '.join([self.positions[i, j] for i in range(self.width)]))
            if j < self.height - 1:
                representation += '\n'
        return representation

    def moves_for(self, current_player: Colour) -> list[tuple[int, int]]:
        result = []
        player_position = self._player_position(current_player)
        if player_position is None:
            for position in self.positions:
                if self.positions[position] == '.':
                    result.append(position)
        else:
            directions = list(itertools.product([-1, 0, 1], repeat=2))
            directions.remove((0, 0))
            for dx, dy in directions:
                px, py = player_position
                px, py = px + dx, py + dy
                while 0 <= px < self.width and 0 <= py < self.height:
                    potential_position = px, py
                    if self.positions[potential_position] == '.':
                        result.append(potential_position)
                        px, py = px + dx, py + dy
                    else:
                        break
        return result

    def apply_move(self, current_player: Colour, move: tuple[int, int]) -> None:
        player_position = self._player_position(current_player)
        if player_position is not None:
            self.positions[player_position] = '#'
        self.positions[move] = current_player.value
        self._update_player_position(current_player, move)

    def _player_position(self, current_player: Colour) -> tuple[int, int]:
        return self.red_position if current_player == Colour.RED else self.blue_position

    def _update_player_position(self, current_player: Colour, new_position: tuple[int, int]) -> None:
        if current_player == Colour.RED:
            self.red_position = new_position
        else:
            self.blue_position = new_position

    def to_state_str(self) -> str:
        positions_in_order = []
        for j in range(self.height):
            for i in range(self.width):
                positions_in_order.append(self.positions[(i, j)])
        return f"{self.width}_{self.height}_{''.join(positions_in_order)}"

    @staticmethod
    def from_state_str(state_str: str) -> 'Board':
        width, height, positions = state_str.split('_')
        width, height = int(width), int(height)
        board = Board(width, height)
        for j in range(height):
            for i in range(width):
                position = positions[j * width + i]
                board.positions[(i, j)] = position
                if position == Colour.RED.value:
                    board.red_position = (i, j)
                elif position == Colour.BLUE.value:
                    board.blue_position = (i, j)
        return board

    def duplicate(self) -> 'Board':
        return self.from_state_str(self.to_state_str())

class Player(Protocol):
    @abstractmethod
    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def register_opponent_action(self, action: tuple[int, int]) -> None:
        raise NotImplementedError

class Game:
    # - tutaj poznasz zasady tego wariantu gry w izolację, są bardzo proste
    # zasady:
    #  * jest dwóch graczy, czerwony i niebieski, czerwony porusza się pierwszy
    #  * każdy gracz ma dokładnie jeden pionek w swoim kolorze ('R' lub 'B')
    #  * plansza jest prostokątem, w swoim pierwszym ruchu każdy gracz może położyć pionek na jej dowolnym pustym polu
    #  * w kolejnych ruchach gracze naprzemiennie przesuwają swoje pionki
    #     * pionki poruszają się jak hetmany szachowe (dowolna liczba pól w poziomie, pionie, lub po skosie)
    #     * pole, z którego pionek startował jest usuwane z planszy ('.' zastępuje '#') i trwale zablokowane
    #     * zarówno pionek innego gracza jak i zablokowane pola uniemożliwiają dalszy ruch (nie da się ich przeskoczyć)
    #  * jeżeli gracz musi wykonać ruch pionkiem, a nie jest to możliwe (każdy z ośmiu kierunków zablokowany)...
    #  * ...to taki gracz przegrywa (a jego przeciwnik wygrywa ;])
    def __init__(self, red: Player, blue: Player, board: Board, current_player: Colour = Colour.RED):
        self.red: Player = red
        self.blue: Player = blue
        self.board: Board = board
        self.current_player: Colour = current_player
        self.finished: bool = False
        self.winner: Optional[Colour] = None

    def run(self, verbose=False):
        if verbose:
            print()
            print(self.board)

        while not self.finished:
            legal_moves = self.board.moves_for(self.current_player)
            if len(legal_moves) == 0:
                self.finished = True
                self.winner = Colour.BLUE if self.current_player == Colour.RED else Colour.RED
                break

            player = self.red if self.current_player == Colour.RED else self.blue
            opponent = self.red if self.current_player == Colour.BLUE else self.blue
            move = player.choose_action(self.board, self.current_player)
            opponent.register_opponent_action(move)
            self.board.apply_move(self.current_player, move)
            self.current_player = self.current_player.flip()

            if verbose:
                print()
                print(self.board)

        if verbose:
            print()
            print(f"WINNER: {self.winner.value}")

class RandomPlayer(Player):
    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        legal_moves = board.moves_for(current_player)
        return random.sample(legal_moves, 1)[0]

    def register_opponent_action(self, action: tuple[int, int]) -> None:
        pass

    def clear_tree(self) -> None:
        pass

class MCTSNode:
    def __init__(self, board: Board, current_player: Colour, c_coefficient: float):
        self.parent: Optional[MCTSNode] = None
        self.leaf: bool = True
        self.terminal: bool = False
        self.times_chosen: int = 0
        self.value: float = 0.5
        self.children: dict[tuple[int, int], MCTSNode] = dict()
        self.board: Board = board
        self.current_player: Colour = current_player
        self.c_coefficient: float = c_coefficient

    def ucb_score(self, child: "MCTSNode") -> float:
        n = child.times_chosen
        if n == 0:
            return float("inf")  # niewypróbowane — najpierw je

        # child.value = P(gracz przy ruchu w dziecku wygrywa) = P(przeciwnik rodzica wygrywa);
        # rodzic maksymalizuje własną szansę: eksploatacja = 1 - child.value
        Q = 1.0 - child.value
        N = self.times_chosen
        exploration = self.c_coefficient * math.sqrt(math.log(max(1, N)) / n)
        return Q + exploration

    def select(self, final=False) -> tuple[int, int]:  # musimy zwrócić ruch na jaki się decydujemy (pole na które wchodzimy)
        if final:
            if not self.children:
                raise RuntimeError("select(final=True): brak dzieci — rozważ reset MCTS lub expand przed wyborem")
            max_visits = max(self.children[move].times_chosen for move in self.children)
            candidates = [move for move in self.children if self.children[move].times_chosen == max_visits]
            return random.choice(candidates)

        best_score = float("-inf")
        candidates: list[tuple[int, int]] = []
        for move, child in self.children.items():
            score = self.ucb_score(child)
            if score > best_score:
                best_score = score
                candidates = [move]
            elif score == best_score:
                candidates.append(move)
        return random.choice(candidates)

    def expand(self) -> None:
        if not self.terminal and self.leaf:
            legal_moves = self.board.moves_for(self.current_player)
            if len(legal_moves) > 0:
                self.leaf = False
                opponent = self.current_player.flip()
                for move in legal_moves:
                    child_board = self.board.duplicate()
                    child_board.apply_move(self.current_player, move)
                    child = MCTSNode(child_board, opponent, self.c_coefficient)
                    child.parent = self
                    self.children[move] = child
            else:
                self.terminal = True

    def simulate(self) -> Colour:
        if self.terminal:
            return self.current_player.flip()   # z automatu wygrywa drugi, bo my nie mamy ruchów
        else:
            starting_board = self.board.duplicate()
            current_player = self.current_player

            while True:
                legal_moves = starting_board.moves_for(current_player)
                if len(legal_moves) == 0:
                    return current_player.flip()

                move = random.choice(legal_moves)
                starting_board.apply_move(current_player, move)
                current_player = current_player.flip()

            return None

    def backpropagate(self, winner: Colour) -> None:

        # czyli idziemy do góry w parentach i aktualizujemy wartości
        current_node = self
        while current_node is not None:

            # getting N
            old_N = current_node.times_chosen
            new_N = old_N + 1

            # old
            old_Q_avg = current_node.value
            old_Q_sum = old_Q_avg * old_N

            # new
            new_Q_sum = old_Q_sum + (1 if winner == current_node.current_player else 0)
            new_Q_avg = new_Q_sum / new_N

            # updating
            current_node.value = new_Q_avg
            current_node.times_chosen = new_N

            current_node = current_node.parent # going UP

class MCTSPlayer(Player):
    def __init__(self, time_limit: float, c_coefficient: float):
        self.time_limit: float = time_limit
        self.root_node: Optional[MCTSNode] = None
        self.c_coefficient: float = c_coefficient

    def clear_tree(self) -> None:
        """Odrzuć drzewo MCTS (np. przed nową partią). Następny ruch zbuduje korzeń od zera."""
        self.root_node = None

    def copy(self) -> "MCTSPlayer":
        """Nowy agent z tymi samymi parametrami (time_limit, c_coefficient), bez drzewa."""
        return MCTSPlayer(self.time_limit, self.c_coefficient)

    def __del__(self) -> None:
        # Nie polegaj na tym między kolejnymi partiami — ten sam Player żyje przez wiele gier.
        # Przy usuwaniu obiektu puść referencję do dużego drzewa, żeby GC mógł je zebrać.
        self.root_node = None

    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        stale = (
            self.root_node is None
            or self.root_node.board.to_state_str() != board.to_state_str()
            or self.root_node.current_player != current_player
        )
        if stale:
            self.root_node = MCTSNode(board.duplicate(), current_player, self.c_coefficient)

        start_time = time.time()
        while True:
            self._mcts_iteration()

            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= self.time_limit:
                break

        action = self.root_node.select(final=True)
        self._step_down(action)
        return action

    def register_opponent_action(self, action: tuple[int, int]) -> None:
        if self.root_node is not None:
            self.root_node.expand()
            self._step_down(action)

    def _mcts_iteration(self):
        # print("MCTS iteration")
        node = self.root_node
        while not node.leaf:
            action = node.select()
            node = node.children[action]
        node.expand()
        winner = node.simulate()
        node.backpropagate(winner)

    def _step_down(self, action: tuple[int, int]) -> None:
        new_root = self.root_node.children[action]
        new_root.parent = None
        self.root_node = new_root

def main(player: Player) -> int:
    red_wins = 0
    blue_wins = 0

    for _ in range(100):
        board = Board(7, 5)
        red_player = player.copy()
        blue_player = RandomPlayer()
        game = Game(red_player, blue_player, board)
        game.run(verbose=False)

        if game.winner == Colour.RED:
            red_wins += 1
        else:
            blue_wins += 1

    print(f"{red_wins} - {blue_wins}", end="", flush=True)
    return red_wins

if __name__ == '__main__':
    import csv
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    c_coefficients = [
                        1.0,
                        0.9,
                        0.8,
                        0.75,
                        0.7,
                        0.6,
                        0.5,
                        0.4,
                        0.3,
                        0.25,
                        0.2,
                        0.1,
                        0.0,
                    ]

    time_limits = [ 
                    0.2,           # 200   ms
                    0.1,           # 100   ms
                    0.05,          # 50    ms
                    0.025,         # 25    ms
                    0.01,          # 10    ms

                    0.005,         # 5     ms
                    0.004,         # 4     ms
                    0.003,         # 3     ms
                    0.0025,        # 2.5   ms
                    0.002,         # 2     ms
                    0.0015,        # 1.5   ms
                    0.001,         # 1     ms
                    0.0005,        # 0.5   ms

                    0.00025,       # 0.25  ms
                    0.0001,        # 0.1   ms
                    0.00005,       # 0.05  ms
                    0.000025,      # 0.025 ms
                    0.00001,       # 0.01  ms
                    0.000005,      # 0.005 ms
                    0.0000025,     # 0.0025 ms

                    0.000001,      # 0.001  ms
                    0.0000005,     # 0.0005 ms
                ]

    _here = os.path.dirname(os.path.abspath(__file__))
    _plots_dir = os.path.join(_here, 'plots')
    os.makedirs(_plots_dir, exist_ok=True)

    num_outer_runs = 10

    matrix: list[list[float]] = []
    run_rows: list[list[list[int]]] = []
    csv_rows: list[tuple[float, float, float, float, list[int]]] = []

    for t in time_limits:
        row: list[float] = []
        row_runs: list[list[int]] = []
        for c in c_coefficients:
            print(f'time={t}, c={c}:')
            run_scores: list[int] = []
            for r in range(num_outer_runs):
                # Świeży agent na każdy run (brak stanu między runami).
                red_wins = main(MCTSPlayer(t, c))
                print()
                run_scores.append(red_wins)
                print(f'  run {r + 1}/{num_outer_runs}: {red_wins} wygranych RED (100 gier)')
            mean_rw = float(np.mean(run_scores))
            std_rw = float(np.std(run_scores, ddof=1)) if num_outer_runs > 1 else 0.0
            print(f'  -> średnia {mean_rw:.2f} (σ={std_rw:.2f})\n')
            row.append(mean_rw)
            row_runs.append(run_scores)
            csv_rows.append((t, c, mean_rw, std_rw, run_scores))
        matrix.append(row)
        run_rows.append(row_runs)

    mat = np.array(matrix, dtype=float)
    run_tensor = np.array(run_rows, dtype=int)
    csv_path = os.path.join(_plots_dir, 'mcts_param_study.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(
            [
                'time_limit_s',
                'c_coefficient',
                'red_wins_mean',
                'red_wins_std',
                'win_rate_mean',
            ]
            + [f'run{k + 1}' for k in range(num_outer_runs)],
        )
        for t, c, mean_rw, std_rw, runs in csv_rows:
            w.writerow([t, c, mean_rw, std_rw, mean_rw / 100.0] + runs)

    # Heatmap: wiersze = time limit (góra = dłuższy czas), kolumny = c
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(
        mat,
        origin='upper',
        aspect='auto',
        cmap='viridis',
        interpolation='nearest',
        vmin=0,
        vmax=100,
    )
    ax.set_xticks(np.arange(len(c_coefficients)))
    ax.set_xticklabels([str(x) for x in c_coefficients])
    ax.set_yticks(np.arange(len(time_limits)))
    ax.set_yticklabels([f'{x * 1000:g}' for x in time_limits])
    ax.set_xlabel('c (eksploracja UCB)')
    ax.set_ylabel('Limit czasu na ruch (ms)')
    ax.set_title(
        'MCTS (czerwony) vs losowy (niebieski) — średnia wygranych RED / 100 gier\n'
        f'(średnia z {num_outer_runs} niezależnych runów parametru)'
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f'Średnia wygranych RED (max 100), n={num_outer_runs} runów')
    fig.tight_layout()
    fig.savefig(os.path.join(_plots_dir, 'mcts_heatmap.png'), dpi=150)
    plt.close(fig)

    # Linie: stałe c, oś X = czas (log)
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    for j, c in enumerate(c_coefficients):
        ax2.semilogx(
            time_limits,
            mat[:, j],
            marker='o',
            ms=3,
            lw=1.2,
            label=f'c = {c}',
        )
    ax2.set_xlabel('Limit czasu na ruch (s), skala log')
    ax2.set_ylabel(f'Średnia wygranych RED / 100 gier (n={num_outer_runs} runów)')
    ax2.set_title('MCTS vs losowy — zależność od czasu i c (uśrednione runy)')
    ax2.grid(True, which='both', ls=':', alpha=0.6)
    ax2.legend(ncol=2, fontsize=8, loc='best')
    fig2.tight_layout()
    fig2.savefig(os.path.join(_plots_dir, 'mcts_lines_by_c.png'), dpi=150)
    plt.close(fig2)

    # Linie: stały czas, oś X = c
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    for i, t in enumerate(time_limits):
        ax3.plot(
            c_coefficients,
            mat[i, :],
            marker='o',
            ms=3,
            lw=1.0,
            label=f'{t * 1000:g} ms',
        )
    ax3.set_xlabel('c (eksploracja UCB)')
    ax3.set_ylabel(f'Średnia wygranych RED / 100 gier (n={num_outer_runs} runów)')
    ax3.set_title('MCTS vs losowy — zależność od c (uśrednione; krzywe = limity czasu)')
    ax3.grid(True, ls=':', alpha=0.6)
    ax3.legend(ncol=3, fontsize=7, loc='best')
    fig3.tight_layout()
    fig3.savefig(os.path.join(_plots_dir, 'mcts_lines_by_time.png'), dpi=150)
    plt.close(fig3)

    n_t, n_c, _ = run_tensor.shape

    # Box ploty: stałe c, na osi X kolejne limity czasu (każde pudełko = num_outer_runs runów)
    ncols_b1 = min(2, max(1, n_c))
    nrows_b1 = max(1, (n_c + ncols_b1 - 1) // ncols_b1)
    fig_b1, axes_b1 = plt.subplots(nrows_b1, ncols_b1, figsize=(7 * ncols_b1, 4.2 * nrows_b1), squeeze=False)
    for j, c in enumerate(c_coefficients):
        rr, cc = divmod(j, ncols_b1)
        axb = axes_b1[rr][cc]
        data = [run_tensor[i, j, :].astype(float) for i in range(n_t)]
        axb.boxplot(data)
        axb.set_xticklabels([f'{time_limits[i] * 1000:g}' for i in range(n_t)])
        axb.set_xlabel('Limit czasu (ms)')
        axb.set_ylabel('Wygrane RED / 100 gier')
        axb.set_title(f'c = {c}')
        axb.grid(True, axis='y', ls=':', alpha=0.5)
        axb.tick_params(axis='x', rotation=45)
    for j in range(n_c, nrows_b1 * ncols_b1):
        rr, cc = divmod(j, ncols_b1)
        axes_b1[rr][cc].set_visible(False)
    fig_b1.suptitle(
        f'Rozkład {num_outer_runs} runów na punkt — stałe c, zmienny limit czasu',
        fontsize=12,
    )
    fig_b1.tight_layout()
    fig_b1.savefig(os.path.join(_plots_dir, 'mcts_boxplot_by_c.png'), dpi=150)
    plt.close(fig_b1)

    # Box ploty: stały limit czasu, na osi X kolejne c
    ncols_b2 = min(3, max(1, n_t))
    nrows_b2 = max(1, (n_t + ncols_b2 - 1) // ncols_b2)
    fig_b2, axes_b2 = plt.subplots(nrows_b2, ncols_b2, figsize=(5.5 * ncols_b2, 3.8 * nrows_b2), squeeze=False)
    for i, t in enumerate(time_limits):
        rr, cc = divmod(i, ncols_b2)
        axb = axes_b2[rr][cc]
        data = [run_tensor[i, j, :].astype(float) for j in range(n_c)]
        axb.boxplot(data)
        axb.set_xticklabels([str(x) for x in c_coefficients])
        axb.set_xlabel('c')
        axb.set_ylabel('Wygrane RED / 100 gier')
        axb.set_title(f'{t * 1000:g} ms')
        axb.grid(True, axis='y', ls=':', alpha=0.5)
    for k in range(n_t, nrows_b2 * ncols_b2):
        rr, cc = divmod(k, ncols_b2)
        axes_b2[rr][cc].set_visible(False)
    fig_b2.suptitle(
        f'Rozkład {num_outer_runs} runów na punkt — stały limit czasu, zmienne c',
        fontsize=12,
    )
    fig_b2.tight_layout()
    fig_b2.savefig(os.path.join(_plots_dir, 'mcts_boxplot_by_time.png'), dpi=150)
    plt.close(fig_b2)

    print(
        f'Zapisano: {csv_path}, wykresy (średnia + box) w {_plots_dir}',
    )