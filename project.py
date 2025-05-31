import tensorflow as tf
import keras
import os
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import chess
import chess.pgn

def fen_to_8x8x12(fen):
    """Convert FEN string to 8x8x12 tensor representation"""
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)  # Convert to 0-7 from top
            col = square % 8
            tensor[row, col, piece_to_channel[piece.symbol()]] = 1.0

    return tensor


def pgn_to_dataset():
    """Convert PGN file to (samples, 8, 8, 12) tensors and binary labels"""
    print("wait")
    pgn_path="chess dataset.pgn"
    x = []
    y = []
    i=0
    with open(pgn_path) as pgn:
        while True:
            i=i+1
            print(i)
            game = chess.pgn.read_game(pgn)
            if not game or i>5000:
                break

            result = game.headers.get("Result", "")
            if result not in ["1-0", "0-1"]:
                continue  # Skip draws and unknown results

            label = 1 if result == "1-0" else 0
            board = game.board()

            for move in game.mainline_moves():
                board.push(move)
                x.append(fen_to_8x8x12(board.fen()))
                y.append(label)
    print("done")
    return np.array(x), np.array(y)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8,8,12)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss=tf.keras.losses.binary_crossentropy,metrics=['accuracy'])
boards, labels = pgn_to_dataset()
model.fit(boards, labels, epochs=7,verbose=2,validation_split=0.2)
