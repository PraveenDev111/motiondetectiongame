// Main JavaScript for Rock Paper Scissors Game

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const captureBtn = document.getElementById('captureBtn');
    const userGesture = document.getElementById('userGesture');
    const computerGesture = document.getElementById('computerGesture');
    const gameResult = document.getElementById('gameResult');
    
    // Function to capitalize first letter
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Function to update the UI with game results
    function updateGameUI(result) {
        // Update user gesture
        userGesture.textContent = result.user_gesture === 'unknown' 
            ? 'Not detected' 
            : capitalizeFirstLetter(result.user_gesture);
        
        // Update computer gesture
        computerGesture.textContent = capitalizeFirstLetter(result.computer_gesture);
        
        // Update game result
        gameResult.textContent = result.result;
        
        // Change background color based on result
        if (result.result.includes('win')) {
            gameResult.parentElement.style.backgroundColor = '#27ae60'; // Green for win
        } else if (result.result.includes('Computer wins')) {
            gameResult.parentElement.style.backgroundColor = '#e74c3c'; // Red for loss
        } else if (result.result.includes('Tie')) {
            gameResult.parentElement.style.backgroundColor = '#f39c12'; // Orange for tie
        } else {
            // Error or other cases
            gameResult.parentElement.style.backgroundColor = '#7f8c8d'; // Gray for error
        }
    }
    
    // Event listener for capture button
    captureBtn.addEventListener('click', function() {
        // Change button text and disable temporarily
        captureBtn.textContent = 'Processing...';
        captureBtn.disabled = true;
        
        // Visual countdown
        let count = 3;
        gameResult.textContent = `Capturing in ${count}...`;
        
        const countdownInterval = setInterval(() => {
            count--;
            if (count > 0) {
                gameResult.textContent = `Capturing in ${count}...`;
            } else {
                clearInterval(countdownInterval);
                gameResult.textContent = 'Analyzing gesture...';
                
                // Call the capture endpoint
                fetch('/capture')
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('Capture response:', data);
                        // Update the UI with the results
                        updateGameUI(data);
                        
                        // Re-enable the button
                        captureBtn.textContent = 'Capture Gesture';
                        captureBtn.disabled = false;
                    })
                    .catch(error => {
                        console.error('Error capturing gesture:', error);
                        // Update UI with error
                        updateGameUI({
                            user_gesture: 'unknown',
                            computer_gesture: 'unknown',
                            result: 'Error: Could not capture gesture. Try again.'
                        });
                        
                        // Re-enable the button
                        captureBtn.textContent = 'Capture Gesture';
                        captureBtn.disabled = false;
                    });
            }
        }, 1000);
    });
    
    // Add keyboard support (spacebar to capture)
    document.addEventListener('keydown', function(event) {
        if (event.code === 'Space' && !captureBtn.disabled) {
            captureBtn.click();
        }
    });
    
    // Function to create animations for the gestures
    function createGestureAnimations() {
        const gestures = document.querySelectorAll('.gesture-display');
        
        gestures.forEach(gesture => {
            gesture.addEventListener('transitionend', () => {
                gesture.classList.remove('animate');
            });
        });
    }
    
    // Initialize gesture animations
    createGestureAnimations();
});
