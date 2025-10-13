import tensorflow as tf
import keras
import os
import sys
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import chess
import chess.pgn
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import pygame
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# board encoding function ---
def fen_to_8x8x12(fen):
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            tensor[row, col, piece_to_channel[piece.symbol()]] = 1.0
    return tensor


# PGN to HDF5 batch parser
def parse_pgn_to_h5(pgn_path, h5_path="chess_dataset.h5", batch_size=10000):
    with h5py.File(h5_path, "w") as hf:
        boards_ds = hf.create_dataset(
            "boards", shape=(0,8,8,12), maxshape=(None,8,8,12),
            dtype=np.float32, chunks=True, compression="gzip"
        )
        labels_ds = hf.create_dataset(
            "labels", shape=(0,), maxshape=(None,), dtype=np.int8,
            chunks=True, compression="gzip"
        )
        x_batch, y_batch = [], []
        game_count, pos_count = 0, 0

        with open(pgn_path) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if not game:
                    break
                result = game.headers.get("Result", "")
                if result not in ["1-0", "0-1"]:
                    continue
                label = 1 if result == "1-0" else 0
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    x_batch.append(fen_to_8x8x12(board.fen()))
                    y_batch.append(label)
                    pos_count += 1
                game_count += 1

                # save batch
                if len(x_batch) >= batch_size:
                    new_size = boards_ds.shape[0] + len(x_batch)
                    boards_ds.resize(new_size, axis=0)
                    labels_ds.resize(new_size, axis=0)
                    boards_ds[-len(x_batch):] = np.array(x_batch)
                    labels_ds[-len(y_batch):] = np.array(y_batch)
                    print(f"Saved {new_size} positions from {game_count} games")
                    x_batch.clear()
                    y_batch.clear()

            # save any remaining data
            if x_batch:
                new_size = boards_ds.shape[0] + len(x_batch)
                boards_ds.resize(new_size, axis=0)
                labels_ds.resize(new_size, axis=0)
                boards_ds[-len(x_batch):] = np.array(x_batch)
                labels_ds[-len(y_batch):] = np.array(y_batch)
                print(f"Final save: {new_size} positions total.")

checkpoint = ModelCheckpoint(
    filepath='weights_epoch_{epoch:02d}.weights.h5',
    save_weights_only=True,
    save_best_only=False,      # save after every epoch
    verbose=1
)

'''
parse_pgn_to_h5("chess dataset.pgn")


with h5py.File("chess_dataset.h5", "r") as hf:
    total_samples = hf["boards"].shape[0]
batch_size = 64
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size

def train_generator():
    with h5py.File("chess_dataset.h5", "r") as hf:
        boards = hf["boards"]
        labels = hf["labels"]
        total_batches = (train_size + batch_size - 1) // batch_size
        batch_num = 0
        while True:
            indices = np.random.permutation(train_size)
            sorted_indices = np.sort(indices)
            for i in range(0, train_size, batch_size):
                batch_start = i
                batch_end = min(i+batch_size, train_size)
                batch_indices = sorted_indices[batch_start:batch_end]
                # Print debug info
                if batch_num % 1000 == 0 or batch_num == total_batches - 1:
                    remaining = total_batches - batch_num
                    print(f"Batch {batch_num+1}/{total_batches} (remaining: {remaining})")
                    print(f"Sample indices: {batch_indices[:5]}...{batch_indices[-5:]}")
                batch_boards = boards[batch_indices]
                batch_labels = labels[batch_indices]
                perm = np.random.permutation(len(batch_boards))
                yield batch_boards[perm], batch_labels[perm]
                batch_num += 1
            batch_num = 0  

def val_generator():
    with h5py.File("chess_dataset.h5", "r") as hf:
        boards = hf["boards"][train_size:]
        labels = hf["labels"][train_size:]
        while True:
            indices = np.arange(len(boards))
            np.random.shuffle(indices)
            for i in range(0, len(boards), batch_size):
                yield boards[indices[i:i+batch_size]], labels[indices[i:i+batch_size]]
model.fit(
    train_generator(),
    steps_per_epoch=train_size // batch_size,
    validation_data=val_generator(),
    validation_steps=val_size // batch_size,
    epochs=3,
    callbacks=[checkpoint],
    verbose=3
)'''



def get_best_move(board, model):
    """
    Args:
        board: chess.Board object - current position (must be white's turn to move)
        model: Your trained Keras model

    Returns:
        best_move: chess.Move object with highest predicted win probability
        best_prob: float - predicted win probability for white
    """
    if not board.is_valid() or board.is_game_over():
        return None, 0.0

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0.0

    # generate all possible resulting positions
    boards_after_moves = []
    for move in legal_moves:
        temp_board = board.copy()
        temp_board.push(move)
        boards_after_moves.append(fen_to_8x8x12(temp_board.fen()))

    # convert to numpy array and predict
    input_tensor = np.array(boards_after_moves)
    predictions = model.predict(input_tensor, verbose=0).flatten()

    # find best move
    best_idx = np.argmin(predictions)
    return legal_moves[best_idx], predictions[best_idx]


pygame.init()
PANEL_WIDTH = 160
WIDTH = HEIGHT = 512
TOTAL_WIDTH = WIDTH + 2 * PANEL_WIDTH
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
IMAGES = {}
PLAYER_IMAGES = {}


def load_images():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wQ', 'wK',
              'bp', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        try:
            IMAGES[piece] = pygame.transform.scale(
                pygame.image.load(resource_path(f"images/{piece}.png")),
                (SQ_SIZE, SQ_SIZE))
        except Exception as e:
            print(f"Failed to load {piece}.png: {e}")
            IMAGES[piece] = pygame.Surface((SQ_SIZE, SQ_SIZE))

    try:
        PLAYER_IMAGES['engine'] = pygame.transform.smoothscale(
            pygame.image.load(resource_path("images/mahmoud.jpg")), (100, 100))
    except Exception as e:
        print(f"Failed to load engine photo: {e}")
        PLAYER_IMAGES['engine'] = pygame.Surface((100, 100))
        PLAYER_IMAGES['engine'].fill((200, 0, 0))

    try:
        PLAYER_IMAGES['player'] = pygame.transform.scale(
            pygame.image.load(resource_path("images/player.jpg")), (100, 100))
    except Exception as e:
        print(f"Failed to load player photo: {e}")
        PLAYER_IMAGES['player'] = pygame.Surface((100, 100))
        PLAYER_IMAGES['player'].fill((0, 0, 200))
def draw_board(screen):
    colors = [pygame.Color("white"), pygame.Color("gray")]
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color,
                             pygame.Rect(PANEL_WIDTH + col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            square = chess.square(col, 7 - row)
            piece = board.piece_at(square)
            if piece:
                color = 'w' if piece.color == chess.WHITE else 'b'
                symbol = piece.symbol()
                key = color + ('p' if symbol.lower() == 'p' else symbol.upper())
                screen.blit(IMAGES[key],
                            pygame.Rect(PANEL_WIDTH + col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))
def highlight_squares(screen, board, selected_square):
    if selected_square is not None:
        row, col = 7 - (selected_square // 8), selected_square % 8
        s = pygame.Surface((SQ_SIZE, SQ_SIZE))
        s.set_alpha(100)
        s.fill(pygame.Color('blue'))
        screen.blit(s, (PANEL_WIDTH + col * SQ_SIZE, row * SQ_SIZE))
        for move in board.legal_moves:
            if move.from_square == selected_square:
                to_row = 7 - (move.to_square // 8)
                to_col = move.to_square % 8
                s.fill(pygame.Color('yellow'))
                screen.blit(s, (PANEL_WIDTH + to_col * SQ_SIZE, to_row * SQ_SIZE))

def draw_left_panel(screen, human_turn):
    panel_rect = pygame.Rect(0, 0, PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(screen, pygame.Color("white"), panel_rect)
    pygame.draw.rect(screen, pygame.Color("black"), panel_rect, 2)
    font_title = pygame.font.SysFont("Arial", 20, bold=True)
    font_small = pygame.font.SysFont("Arial", 16)

    # mahmoud Engine v1.0 (top)
    screen.blit(PLAYER_IMAGES['engine'], (30, 20))
    eng_text = font_title.render("Mahmoud", True, pygame.Color("black"))
    screen.blit(eng_text, (20, 130))
    eng_ver = font_small.render("Engine v1.0", True, pygame.Color("black"))
    screen.blit(eng_ver, (20, 155))
    col_text = font_small.render("(Black)", True, pygame.Color("black"))
    screen.blit(col_text, (20, 175))
    if not human_turn:
        pygame.draw.rect(screen, pygame.Color("green"), pygame.Rect(20, 20, 100, 100), 3)

    # you (bottom)
    you_label_y = HEIGHT - 180   # <<-- Move this value higher for more space
    you_text = font_title.render("You", True, pygame.Color("black"))
    screen.blit(you_text, (20, you_label_y))
    col2_text = font_small.render("(White)", True, pygame.Color("black"))
    screen.blit(col2_text, (20, you_label_y + 25))
    screen.blit(PLAYER_IMAGES['player'], (30, HEIGHT - 120))
    if human_turn:
        pygame.draw.rect(screen, pygame.Color("green"), pygame.Rect(20, HEIGHT - 120, 100, 100), 3)

def draw_right_panel(screen, board, model):
    panel_rect = pygame.Rect(PANEL_WIDTH + WIDTH, 0, PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(screen, pygame.Color("white"), panel_rect)
    pygame.draw.rect(screen, pygame.Color("black"), panel_rect, 2)
    font_title = pygame.font.SysFont("Arial", 18, bold=True)
    font_medium = pygame.font.SysFont("Arial", 16)
    # Title
    title_text = font_title.render("Expected Winner", True, pygame.Color("black"))
    screen.blit(title_text, (PANEL_WIDTH + WIDTH + 10, 30))
    # Get current position evaluation
    try:
        current_fen = board.fen()
        position_tensor = fen_to_8x8x12(current_fen)
        prediction = model.predict(np.array([position_tensor]), verbose=0)[0][0]
        white_prob = prediction * 100
        black_prob = (1 - prediction) * 100
        # display probabilitiees
        white_text = font_medium.render(f"White: {white_prob:.1f}%", True, pygame.Color("black"))
        black_text = font_medium.render(f"Black: {black_prob:.1f}%", True, pygame.Color("black"))
        screen.blit(white_text, (PANEL_WIDTH + WIDTH + 10, 70))
        screen.blit(black_text, (PANEL_WIDTH + WIDTH + 10, 100))
        # winner prediction
        if white_prob > black_prob:
            winner_text = font_medium.render("Predicted: White", True, pygame.Color("black"))
        elif black_prob > white_prob:
            winner_text = font_medium.render("Predicted: Black", True, pygame.Color("black"))
        else:
            winner_text = font_medium.render("Predicted: Draw", True, pygame.Color("black"))
        screen.blit(winner_text, (PANEL_WIDTH + WIDTH + 10, 140))
    except Exception:
        error_text = font_medium.render("Prediction Error", True, pygame.Color("black"))
        screen.blit(error_text, (PANEL_WIDTH + WIDTH + 10, 70))

def draw_promotion_menu(screen, color):
    # Show Q, R, B, N vertically on the right panel
    menu_pieces = ['Q', 'R', 'B', 'N']
    for i, piece in enumerate(menu_pieces):
        key = ('w' if color else 'b') + piece
        rect = pygame.Rect(PANEL_WIDTH + WIDTH + 30, 200 + i * (SQ_SIZE + 10), SQ_SIZE, SQ_SIZE)
        pygame.draw.rect(screen, pygame.Color("lightgray"), rect)
        pygame.draw.rect(screen, pygame.Color("black"), rect, 2)
        screen.blit(IMAGES[key], rect)
    font = pygame.font.SysFont("Arial", 16)
    choose_text = font.render("Choose promotion:", True, pygame.Color("black"))
    screen.blit(choose_text, (PANEL_WIDTH + WIDTH + 10, 170))

def main(model_path):
    model = models.Sequential()
    model.add(layers.Input(shape=(8, 8, 12)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    model.load_weights(resource_path('weights_epoch_03.weights.h5'))
    screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
    pygame.display.set_caption("Chess vs Mahmoud Engine v1.0")
    clock = pygame.time.Clock()
    board = chess.Board()
    load_images()

    selected_square = None
    human_turn = True
    promotion_pending = False
    promotion_move = None
    promotion_color = None

    running = True
    while running:
        if not promotion_pending:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN and human_turn:
                    location = pygame.mouse.get_pos()
                    if PANEL_WIDTH <= location[0] < PANEL_WIDTH + WIDTH:
                        col = (location[0] - PANEL_WIDTH) // SQ_SIZE
                        row = 7 - (location[1] // SQ_SIZE)
                        square = chess.square(col, row)
                        if selected_square == square:
                            selected_square = None
                        elif board.piece_at(square) and board.piece_at(square).color == chess.WHITE:
                            selected_square = square
                        elif selected_square is not None:
                            move = chess.Move(selected_square, square)
                            # Promotion detection
                            if (board.piece_at(selected_square).piece_type == chess.PAWN and
                                    (chess.square_rank(square) == 0 or chess.square_rank(square) == 7)):
                                promotion_pending = True
                                promotion_move = move
                                promotion_color = board.turn
                            elif move in board.legal_moves:
                                board.push(move)
                                human_turn = False
                            selected_square = None

        # promotion menu event handling
        if promotion_pending:
            draw_promotion_menu(screen, promotion_color)
            pygame.display.flip()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                    promotion_pending = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    for i in range(4):
                        rect = pygame.Rect(PANEL_WIDTH + WIDTH + 30, 200 + i * (SQ_SIZE + 10), SQ_SIZE, SQ_SIZE)
                        if rect.collidepoint(x, y):
                            promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                            move = chess.Move(
                                promotion_move.from_square,
                                promotion_move.to_square,
                                promotion=promotion_pieces[i]
                            )
                            if move in board.legal_moves:
                                board.push(move)
                                human_turn = False
                            promotion_pending = False
                            promotion_move = None
                            promotion_color = None

        # AI Move (Black)
        if not human_turn and not board.is_game_over() and not promotion_pending:
            ai_move, _ = get_best_move(board, model)
            if ai_move:
                board.push(ai_move)
            human_turn = True

        # draw everything
        screen.fill(pygame.Color("white"))
        draw_left_panel(screen, human_turn)
        draw_board(screen)
        highlight_squares(screen, board, selected_square)
        draw_pieces(screen, board)
        draw_right_panel(screen, board, model)
        if promotion_pending:
            draw_promotion_menu(screen, promotion_color)

        if board.is_game_over():
            font = pygame.font.SysFont("Arial", 32)
            text = font.render("Game Over - " + board.result(), True, pygame.Color("red"))
            screen.blit(text, (TOTAL_WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))

        pygame.display.flip()
        clock.tick(15)


if __name__ == "__main__":
    import tensorflow as tf  # Ensure TensorFlow is imported

    main("weights_epoch_03.weights.h5")
