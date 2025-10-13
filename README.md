<p align="center">
  <a href="https://mahmoudhossamws.github.io/Mahmoud_Chess_Engine_Web/">
    <img src="https://img.shields.io/badge/🌐_Visit_Website-1e90ff?style=for-the-badge&logoColor=white" alt="Visit Website">
  </a>
  &nbsp;&nbsp;
  <a href="https://drive.google.com/file/d/17PBX1-uNQafle0HKlBocnyvw1g0re3ry/view?usp=sharing">
    <img src="https://img.shields.io/badge/💾_Download_Windows_App-28a745?style=for-the-badge&logoColor=white" alt="Download Windows App">
  </a>
</p>

# Mahmoud Chess Engine v1.0

Mahmoud Chess Engine v1.0 is a neural network chess engine that selects moves using a convolutional neural network trained on game outcomes. The engine uses supervised learning and is open for collaboration.

## Approach

Data Preparation: Positions are taken from chess games (PGN files) and encoded as 8x8x12 tensors. Each position is labeled with the final game result (1 for White win, 0 for Black win)

Model Architecture: A CNN predicts the probability that White will win from a position. The model is trained with binary cross-entropy loss using only the final outcome

Move Selection: For every legal move, the engine simulates the move, encodes the resulting board, and predicts the win probability. The move with the highest probability (or lowest for Black) is chosen

No Search or Lookahead: The engine does not perform tree search or tactical lookahead. It evaluates positions using the neural network only

## Performance

The engine shows intermediate performance. It avoids basic blunders, develops pieces sensibly, and keeps reasonable material balance and king safety. It is suitable for casual and intermediate players but does not reach the depth of search-based engines

## Open for Collaboration

The project is open source and welcomes contributors. You can improve the neural network, add search, enhance training, or experiment with ideas

How to contribute:  
- Fork the repository  
- Open issues or pull requests  
- Discuss ideas and directions  

For collaboration, [open an issue](https://github.com/mahmoudhossamws/Mahmoud-Chess-engine/issues) or contact me directly

## Website

The project website uses HTML, CSS, and JavaScript and is hosted on GitHub Pages. The model is hosted on Hugging Face Spaces and accessed via its API

![Web Screenshot](Web_Screenshot.PNG)

## Desktop App

The Windows desktop app is a standalone build with a Pygame interface

- UI library: Pygame  
- Run in development: install Python 3.8+ and Pygame, then run `project.py`  
- Distribution: packaged Windows executable

![Interface](screenshot.PNG)

## Contact

For questions, feedback, or collaboration requests:  
mahmoudhossam@aucegypt.edu
