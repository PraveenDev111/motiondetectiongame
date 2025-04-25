# Rock Paper Scissors Game

This project is a Rock-Paper-Scissors game that uses computer vision to detect hand gestures. It was created for the CS402.3 Computer Graphics and Visualization course at NSBM Green University.

## Features

- Real-time hand gesture recognition using OpenCV and YOLO
- Web-based interface built with Flask
- Visualization of image processing steps (grayscale, thresholding, binarization, etc.)
- Interactive gameplay against the computer

## Requirements

This project requires the following dependencies:
- Python 3.6+
- Flask
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- Pillow

All dependencies are listed in the `requirements.txt` file.

## Installation

1. Clone the repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Download the YOLO model (will be done automatically on first run)

## Usage

1. Run the application:
```
python app.py
```

2. Open a web browser and navigate to `http://127.0.0.1:5000/`

3. Allow camera access when prompted

4. Play the game:
   - Position your hand in front of the camera
   - Say "Rock, Paper, Scissors, Shoot!" and make your gesture
   - Click the "Capture Gesture" button (or press spacebar)
   - See the results and try again!

## Project Structure

- `app.py`: Main application file with the Flask server and image processing logic
- `templates/`: Contains HTML templates
  - `index.html`: Main game interface
- `static/`: Contains static files
  - `css/style.css`: Styling for the web interface
  - `js/script.js`: JavaScript for the game functionality

## Image Processing Steps

1. Capture image from webcam
2. Convert to grayscale
3. Apply Gaussian blur
4. Perform thresholding
5. Find contours
6. Detect hand gesture
7. Determine game result

## Game Rules

- Rock beats Scissors
- Scissors beats Paper
- Paper beats Rock
- Same gesture is a tie

## Extension

The game can be extended to include "Rock, Paper, Scissors, Lizard, Spock" from the TV series "The Big Bang Theory" as suggested in the coursework requirements.

## Contributors

- [Team Member 1] - [Role]
- [Team Member 2] - [Role]
- [Team Member 3] - [Role]
- [Team Member 4] - [Role]
- [Team Member 5] - [Role]

## License

[Specify license]
