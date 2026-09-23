import random

import numpy as np
import pygame
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# Snake AI using scikit-learn
# ------------------------------------------------------------

WIDTH = 600
HEIGHT = 600
GRID = 20
CELL = WIDTH // GRID

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (255, 0, 0)
BLUE = (0, 120, 255)
YELLOW = (255, 255, 0)
GRAY = (70, 70, 70)

DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
DIR_ORDER = ["up", "right", "down", "left"]


# ------------------------------------------------------------
# 1. Create a simple AI model using scikit-learn
# ------------------------------------------------------------
def distance_to_wall(head, direction):
    x, y = head
    dx, dy = DIRECTIONS[direction]
    steps = 0
    while True:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < GRID and 0 <= ny < GRID:
            x, y = nx, ny
            steps += 1
        else:
            return steps


def cell_is_blocked(head, body, direction):
    dx, dy = DIRECTIONS[direction]
    nx = head[0] + dx
    ny = head[1] + dy
    if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
        return True
    if (nx, ny) in body:
        return True
    return False


def heuristic_best_move(head, body, food):
    best_move = None
    best_score = -999999

    for direction in DIR_ORDER:
        dx, dy = DIRECTIONS[direction]
        nx = head[0] + dx
        ny = head[1] + dy

        if nx < 0 or nx >= GRID or ny < 0 or ny >= GRID:
            score = -1000
        elif (nx, ny) in body:
            score = -1000
        else:
            # prefer moves that move toward food
            food_dx = food[0] - nx
            food_dy = food[1] - ny
            distance_now = abs(food[0] - head[0]) + abs(food[1] - head[1])
            distance_after = abs(food_dx) + abs(food_dy)
            score = 50 - distance_after

            # prefer open space
            score += distance_to_wall((nx, ny), direction) * 2

            # avoid being trapped
            if cell_is_blocked((nx, ny), body, direction):
                score -= 100

            # slight preference for moving toward food
            if distance_after < distance_now:
                score += 10

        if score > best_score:
            best_score = score
            best_move = direction

    return best_move


def make_training_data(samples=5000):
    X = []
    y = []

    for _ in range(samples):
        # random board layout for training
        snake = [(random.randint(2, GRID - 3), random.randint(2, GRID - 3))]
        length = random.randint(1, 5)
        for _ in range(length):
            snake.append((snake[-1][0] + 1, snake[-1][1]))

        head = snake[0]
        food = (random.randint(0, GRID - 1), random.randint(0, GRID - 1))
        body = set(snake[1:])

        for direction in DIR_ORDER:
            dx, dy = DIRECTIONS[direction]
            nx = head[0] + dx
            ny = head[1] + dy
            dist_to_food = abs(nx - food[0]) + abs(ny - food[1])
            wall_distance = distance_to_wall((nx, ny), direction)
            blocked = 1 if (nx < 0 or ny < 0 or nx >= GRID or ny >= GRID or (nx, ny) in body) else 0

            feature = [
                head[0],
                head[1],
                food[0],
                food[1],
                dx,
                dy,
                dist_to_food,
                wall_distance,
                blocked,
                len(body),
            ]
            label = heuristic_best_move(head, list(body), food)
            X.append(feature)
            y.append(label)

    return np.array(X), np.array(y)


# Train scikit-learn model
X_train, y_train = make_training_data(800)
model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)


# ------------------------------------------------------------
# 2. Snake game logic
# ------------------------------------------------------------
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = "right"
        self.next_direction = "right"
        self.snake = [(GRID // 2, GRID // 2), (GRID // 2 - 1, GRID // 2), (GRID // 2 - 2, GRID // 2)]
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False

    def spawn_food(self):
        while True:
            pos = (random.randint(0, GRID - 1), random.randint(0, GRID - 1))
            if pos not in self.snake:
                return pos

    def get_ai_move(self):
        head = self.snake[0]
        body = self.snake[1:]
        food = self.food

        feature = [
            head[0],
            head[1],
            food[0],
            food[1],
            DIRECTIONS[self.next_direction][0],
            DIRECTIONS[self.next_direction][1],
            abs(head[0] - food[0]) + abs(head[1] - food[1]),
            distance_to_wall(head, self.next_direction),
            1 if cell_is_blocked(head, body, self.next_direction) else 0,
            len(body),
        ]

        prediction = model.predict([feature])[0]
        return prediction

    def update(self):
        if self.game_over:
            return

        self.direction = self.next_direction
        dx, dy = DIRECTIONS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        if new_head[0] < 0 or new_head[0] >= GRID or new_head[1] < 0 or new_head[1] >= GRID:
            self.game_over = True
            return

        if new_head in self.snake:
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop()

    def draw(self, screen):
        screen.fill(BLACK)

        for x in range(GRID):
            for y in range(GRID):
                rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
                pygame.draw.rect(screen, GRAY, rect, 1)

        for segment in self.snake:
            x, y = segment
            pygame.draw.rect(screen, GREEN, (x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2))

        fx, fy = self.food
        pygame.draw.rect(screen, RED, (fx * CELL + 2, fy * CELL + 2, CELL - 4, CELL - 4))

        score_text = pygame.font.SysFont(None, 30).render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        if self.game_over:
            over_text = pygame.font.SysFont(None, 60).render("Game Over", True, WHITE)
            screen.blit(over_text, (WIDTH // 2 - 120, HEIGHT // 2 - 20))


# ------------------------------------------------------------
# 3. Game loop
# ------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake AI with scikit-learn")
    clock = pygame.time.Clock()

    game = SnakeGame()
    running = True
    ai_mode = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    ai_mode = not ai_mode
                elif event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    mapping = {
                        pygame.K_UP: "up",
                        pygame.K_DOWN: "down",
                        pygame.K_LEFT: "left",
                        pygame.K_RIGHT: "right",
                    }
                    new_dir = mapping[event.key]
                    current = game.direction
                    dx, dy = DIRECTIONS[current]
                    new_dx, new_dy = DIRECTIONS[new_dir]

                    if (dx + new_dx, dy + new_dy) != (0, 0):
                        game.next_direction = new_dir
                        ai_mode = False

        if not game.game_over:
            if ai_mode:
                ai_move = game.get_ai_move()
                if ai_move in DIRECTIONS:
                    current = game.direction
                    dx, dy = DIRECTIONS[current]
                    new_dx, new_dy = DIRECTIONS[ai_move]

                    if (dx + new_dx, dy + new_dy) != (0, 0):
                        game.next_direction = ai_move
            game.update()

        game.draw(screen)
        pygame.display.flip()
        clock.tick(8)

    pygame.quit()


if __name__ == "__main__":
    main()
